from typing import List, Optional, Tuple, Union

import json
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
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
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ):

        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg
        self.batch_size = args.per_device_train_batch_size
        self.tokenizer = tokenizer

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

        self.tokenized_examples = defaultdict(list)

    def train(self, p_encoder, q_encoder):
        args = self.args
        batch_size = self.batch_size
        num_neg = self.num_neg

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)],
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
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

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

                    targets = torch.zeros(batch_size).long()  # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "attention_mask": batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "token_type_ids": batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                    }
                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device),
                    }

                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(
                        q_outputs, torch.transpose(p_outputs, 1, 2)
                    ).squeeze()  # (batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

            self.p_encoder.eval()
            self.q_encoder.eval()
            outputs = torch.zeros((batch_size, 768), device=args.device)

            with torch.no_grad():

                with tqdm(self.valid_dataloader, unit="batch") as tepoch:

                    for batch in tepoch:

                        p_inputs = {
                            "input_ids": batch[0].to(args.device),
                            "attention_mask": batch[1].to(args.device),
                            "token_type_ids": batch[2].to(args.device),
                        }

                        outputs = torch.cat([outputs, self.p_encoder(**p_inputs)])

                    outputs = outputs[batch_size:]

                    corrected_prediction = 0
                    for idx, batch in enumerate(tepoch):

                        q_inputs = {
                            "input_ids": batch[3].to(args.device),
                            "attention_mask": batch[4].to(args.device),
                            "token_type_ids": batch[5].to(args.device),
                        }

                        if batch[0].size(0) == batch_size:
                            targets = torch.LongTensor(
                                list(range(batch_size * idx + batch_size - batch_size, batch_size * idx + batch_size))
                            )
                        else:
                            targets = torch.LongTensor(
                                list(
                                    range(
                                        batch_size * idx + batch_size - batch_size, len(self.valid_dataloader.dataset)
                                    )
                                )
                            )

                        q_logits = self.q_encoder(**q_inputs)
                        scores = torch.mm(q_logits, outputs.permute(1, 0))
                        corrected_prediction += sum(torch.argmax(scores, dim=1).cpu() == targets).item()

                    print(f"valid 정확도 : {corrected_prediction / len(self.valid_dataloader.dataset) * 100}%")

        torch.save(self.p_encoder, "../encoder/p_encoder.pt")
        torch.save(self.q_encoder, "../encoder/q_encoder.pt")

    def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None):

        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            if os.path.isfile("./data/preprocessed.json"):
                with open("./data/preprocessed.json", "r") as f:
                    self.tokenized_examples = json.load(f)

                    self.tokenized_examples["input_ids"] = torch.tensor(self.tokenized_examples["input_ids"])
                    self.tokenized_examples["token_type_ids"] = torch.tensor(self.tokenized_examples["token_type_ids"])
                    self.tokenized_examples["attention_mask"] = torch.tensor(self.tokenized_examples["attention_mask"])
            else:

                for context in self.contexts:
                    tokenized_example = self.tokenizer(
                        context,
                        truncation=True,
                        max_length=512,
                        padding="max_length",
                    )
                    self.tokenized_examples["input_ids"].append(tokenized_example["input_ids"])
                    self.tokenized_examples["token_type_ids"].append(tokenized_example["token_type_ids"])
                    self.tokenized_examples["attention_mask"].append(tokenized_example["attention_mask"])

                with open("./data/preprocessed.json", "w") as f:
                    json.dump(self.tokenized_examples, f)

                with open("./data/preprocessed.json", "r") as f:
                    self.tokenized_examples = json.load(f)
                    self.tokenized_examples["input_ids"] = torch.tensor(self.tokenized_examples["input_ids"])
                    self.tokenized_examples["token_type_ids"] = torch.tensor(self.tokenized_examples["token_type_ids"])
                    self.tokenized_examples["attention_mask"] = torch.tensor(self.tokenized_examples["attention_mask"])

            q_seqs = self.tokenizer(query, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

            p_seqs = TensorDataset(
                self.tokenized_examples["input_ids"],
                self.tokenized_examples["token_type_ids"],
                self.tokenized_examples["attention_mask"],
            )
            p_seqs = DataLoader(p_seqs, batch_size=32, shuffle=False)

            q_seqs = TensorDataset(q_seqs["input_ids"], q_seqs["token_type_ids"], q_seqs["attention_mask"])
            q_seqs = DataLoader(q_seqs, batch_size=32, shuffle=False)

            p_embs = []
            q_embs = []

            p_encoder = p_encoder.to(args.device)
            q_encoder = q_encoder.to(args.device)

            if os.path.isfile("./data/embedding.npy"):
                p_embs = np.load("./data/embedding.npy")
                p_embs = torch.from_numpy(p_embs)
            else:
                p_embs = torch.zeros((32, 768))
                p_embs = p_embs.to(args.device)
                for batch in tqdm(p_seqs):
                    batch = tuple(t.to(args.device) for t in batch)
                    p_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
                    p_emb = p_encoder(**p_inputs)
                    p_embs = torch.cat([p_embs, p_emb])

                p_embs = p_embs.cpu().numpy()
                p_embs = p_embs[32:]
                np.save("./data/embedding.npy", p_embs)

                p_embs = np.load("./data/embedding.npy")
                p_embs = torch.from_numpy(p_embs)

            if os.path.isfile("./data/embeddings.npy"):
                q_embs = np.load("./data/embeddings.npy")
                q_embs = torch.from_numpy(q_embs)
            else:
                q_embs = torch.zeros((32, 768))
                for batch in tqdm(q_seqs):
                    batch = tuple(t.to(args.device) for t in batch)
                    q_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
                    q_emb = q_encoder(**q_inputs).to("cpu")
                    q_embs = torch.cat([q_embs, q_emb])

                q_embs = q_embs.cpu().numpy()
                q_embs = q_embs[32:]
                np.save("./data/embeddings", q_embs)

                q_embs = np.load("./data/embeddings.npy")
                q_embs = torch.from_numpy(q_embs)

        dot_prod_scores = torch.matmul(q_embs, torch.transpose(p_embs, 0, 1))
        dot_prod_scores = np.array(dot_prod_scores)
        doc_scores = []
        doc_indices = []
        for i in range(dot_prod_scores.shape[0]):
            sorted_result = np.argsort(dot_prod_scores[i, :])[::-1]
            doc_scores.append(dot_prod_scores[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def retrieve(
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
                doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset["question"], k=topk)
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
