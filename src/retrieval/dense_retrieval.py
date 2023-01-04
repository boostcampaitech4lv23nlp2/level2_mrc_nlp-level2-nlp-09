from typing import List, Optional, Tuple, Union

import json
import os
import pickle
import random
import re
import time
from collections import defaultdict
from contextlib import contextmanager

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class DenseRetrieval:
    def __init__(
        self,
        args,
        dataset,
        tokenizer,
        num_neg,
        data_path: Optional[str] = "./data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ):

        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg
        self.batch_size = args.per_device_train_batch_size
        self.tokenizer = tokenizer
        self.num_neg = num_neg

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            self.wiki = json.load(f)

        new_wiki = {}
        for i in range(len(self.wiki)):
            key = str(i)
            context = self.wiki[key]["text"]
            self.wiki[key]["text"] = self.preprocess(context)
            new_wiki[key] = self.wiki[key]

        self.contexts = list(dict.fromkeys([v["text"] for v in new_wiki.values()]))
        self.wiki_context_id_dict = {v["text"]: v["document_id"] for v in new_wiki.values()}
        self.wiki_id_context_dict = {v["document_id"]: v["text"] for v in new_wiki.values()}

        self.tokenized_examples = defaultdict(list)

    def preprocess(self, text):
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\\n", " ", text)  # remove newline character
        text = re.sub(r"\s+", " ", text)  # remove continuous spaces
        text = re.sub(r"#", " ", text)

        return text

    def train(self, train_dataset, valid_dataset, p_encoder, q_encoder):

        args = self.args
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        train_sampler = RandomSampler(self.train_dataset)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=args.per_device_train_batch_size,
            drop_last=True,
        )
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=args.per_device_eval_batch_size)
        num_neg = self.num_neg

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Start training!
        global_step = 0

        p_encoder.zero_grad()
        q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):

        best_loss = 9999  # valid_loss를 저장하는 변수
        num_epoch = 0
        for _ in train_iterator:

            train_loss = 0
            train_acc = 0
            train_step = 0

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    train_step += 1
                    p_encoder.train()
                    q_encoder.train()

                    neg_batch_ids = []
                    neg_batch_att = []
                    neg_batch_tti = []
                    random_sampling_idx = random.randrange(0, num_neg)
                    for batch_in_sample_idx in range(args.per_device_train_batch_size):
                        """
                        question과 pos passage는 1대1로 매칭이 되지만
                        hard negative sample들은 해당 question에 대해 num_neg의 수만큼 매칭이 되기 때문에
                        매 학습 루프마다 한개를 랜덤하게 뽑아서 pos passage와 concat을 하여 사용하게 됩니다.
                        """
                        neg_batch_ids.append(batch[3][:][batch_in_sample_idx][random_sampling_idx].unsqueeze(0))
                        neg_batch_att.append(batch[4][:][batch_in_sample_idx][random_sampling_idx].unsqueeze(0))
                        neg_batch_tti.append(batch[5][:][batch_in_sample_idx][random_sampling_idx].unsqueeze(0))
                    neg_batch_ids = torch.cat(neg_batch_ids)
                    neg_batch_att = torch.cat(neg_batch_att)
                    neg_batch_tti = torch.cat(neg_batch_tti)
                    p_inputs = {
                        "input_ids": torch.cat((batch[0], neg_batch_ids), 0).cuda(),
                        "attention_mask": torch.cat((batch[1], neg_batch_att), 0).cuda(),
                        "token_type_ids": torch.cat((batch[2], neg_batch_tti), 0).cuda(),
                    }

                    q_inputs = {
                        "input_ids": batch[6].cuda(),
                        "attention_mask": batch[7].cuda(),
                        "token_type_ids": batch[8].cuda(),
                    }

                    p_outputs = p_encoder(**p_inputs)  # (batch_size * 2, emb_dim)
                    q_outputs = q_encoder(**q_inputs)  # (batch_size, emb_dim)

                    # Calculate similarity score & loss
                    sim_scores = torch.matmul(
                        q_outputs, torch.transpose(p_outputs, 0, 1)
                    )  # (batch_size, emb_dim) x (emb_dim, batch_size * 2) = (batch_size, batch_size * 2)

                    # 정답은 대각선의 성분들 -> 0 1 2 ... batch_size - 1
                    targets = torch.arange(0, args.per_device_train_batch_size).long()
                    if torch.cuda.is_available():
                        targets = targets.to("cuda")

                    sim_scores = F.log_softmax(sim_scores, dim=1)
                    loss = F.nll_loss(sim_scores, targets)
                    train_loss += loss.item()

                    _, preds = torch.max(sim_scores, 1)  #
                    train_acc += torch.sum(preds.cpu() == targets.cpu()) / args.per_device_train_batch_size

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    q_encoder.zero_grad()
                    p_encoder.zero_grad()
                    global_step += 1
                    # validation
                    if train_step % 247 == 0:
                        valid_loss = 0
                        valid_acc = 0
                        v_epoch_iterator = tqdm(self.valid_dataloader, desc="Iteration")
                        for step, batch in enumerate(v_epoch_iterator):
                            with torch.no_grad():
                                q_encoder.eval()
                                p_encoder.eval()

                                cur_batch_size = batch[0].size()[0]
                                # 마지막 배치의 drop last를 안하기 때문에 단순 batch_size를 사용하면 에러발생
                                if torch.cuda.is_available():
                                    batch = tuple(t.cuda() for t in batch)
                                p_inputs = {
                                    "input_ids": batch[0],
                                    "attention_mask": batch[1],
                                    "token_type_ids": batch[2],
                                }

                                q_inputs = {
                                    "input_ids": batch[6],
                                    "attention_mask": batch[7],
                                    "token_type_ids": batch[8],
                                }
                                p_outputs = p_encoder(**p_inputs)
                                q_outputs = q_encoder(**q_inputs)

                                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
                                targets = torch.arange(0, cur_batch_size).long()
                                if torch.cuda.is_available():
                                    targets = targets.to("cuda")

                                sim_scores = F.log_softmax(sim_scores, dim=1)
                                loss = F.nll_loss(sim_scores, targets)

                                _, preds = torch.max(sim_scores, 1)  #
                                valid_acc += torch.sum(preds.cpu() == targets.cpu()) / cur_batch_size

                                valid_loss += loss
                        valid_loss = valid_loss / len(self.valid_dataloader)
                        valid_acc = valid_acc / len(self.valid_dataloader)

                        print()
                        print(f"valid loss: {valid_loss}")
                        print(f"valid acc: {valid_acc}")
                        if best_loss > valid_loss:
                            # valid_loss가 작아질 때만 저장하고 best_loss와 best_acc를 업데이트
                            # acc에 대해서도 가능합니다.
                            print("best model save")
                            p_encoder.save_pretrained(args.output_dir + "/p_encoder")
                            q_encoder.save_pretrained(args.output_dir + "/q_encoder")
                            best_loss = valid_loss

                num_epoch += 1
                train_loss = train_loss / len(self.train_dataloader)
                train_acc = train_acc / len(self.train_dataloader)

                print(f"train loss: {train_loss}")
                print(f"train acc: {train_acc}")

    def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None):

        eval_batch_size = 32

        p_encoder.cuda()
        q_encoder.cuda()

        if os.path.isfile("./data/embedding.bin"):
            with open("./data/embedding.bin", "rb") as f:
                p_embs = pickle.load(f)
        else:
            self.tokenized_example = self.tokenizer(
                self.contexts, truncation=True, max_length=512, padding="max_length", return_tensors="pt"
            )

            p_seqs = TensorDataset(
                self.tokenized_example["input_ids"],
                self.tokenized_example["token_type_ids"],
                self.tokenized_example["attention_mask"],
            )
            sampler = SequentialSampler(p_seqs)
            dataloader = DataLoader(p_seqs, sampler=sampler, batch_size=eval_batch_size)

            p_embs = []

            with torch.no_grad():

                epoch_iterator = tqdm(dataloader, desc="Iteration", position=0, leave=True)
                p_encoder.eval()
                for _, batch in enumerate(tqdm(epoch_iterator)):
                    batch = tuple(t.cuda() for t in batch)
                    p_inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                    }
                    outputs = p_encoder(**p_inputs).to("cpu").numpy()
                    p_embs.extend(outputs)

            torch.cuda.empty_cache()
            p_embs = np.array(p_embs)

            with open("./data/embedding.bin", "wb") as f:
                pickle.dump(p_embs, f)

        q_seqs = self.tokenizer(
            query,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dataset = TensorDataset(q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"])
        query_sampler = SequentialSampler(dataset)
        query_dataloader = DataLoader(dataset, sampler=query_sampler, batch_size=32)
        q_embs = []

        with torch.no_grad():

            epoch_iterator = tqdm(query_dataloader, desc="Iteration", position=0, leave=True)
            q_encoder.eval()

            for _, batch in enumerate(epoch_iterator):
                batch = tuple(t.cuda() for t in batch)

                q_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                outputs = q_encoder(**q_inputs).to("cpu").numpy()
                q_embs.extend(outputs)
        q_embs = np.array(q_embs)

        if torch.cuda.is_available():
            p_embs_cuda = torch.Tensor(p_embs).to("cuda")
            q_embs_cuda = torch.Tensor(q_embs).to("cuda")
        dot_prod_scores = torch.matmul(q_embs_cuda, torch.transpose(p_embs_cuda, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

        query_ids = {}
        query_scores = {}
        idx = 0
        for i in tqdm(range(len(query))):
            p_list = []
            scores = []
            q = query[i]
            for j in range(k):
                p_list.append(self.wiki_context_id_dict[self.contexts[rank[idx][j]]])
                scores.append(dot_prod_scores[idx][rank[idx][j]].item())
            query_ids[q] = p_list
            query_scores[q] = scores
            idx += 1

        return query_scores, query_ids

    def retrieve(
        self, p_encoder, q_encoder, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc(
                    query_or_dataset["question"], k=topk, p_encoder=p_encoder, q_encoder=q_encoder
                )
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join([self.wiki_id_context_dict[pid] for pid in doc_indices[example["question"]]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def build_faiss(self, num_clusters=64) -> None:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(quantizer, quantizer.d, num_clusters, faiss.METRIC_L2)
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(queries, k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform([query])
        assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vecs = self.tfidfv.transform(queries)
        assert np.sum(query_vecs) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()
