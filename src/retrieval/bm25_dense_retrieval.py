from typing import List, Optional, Tuple, Union

import json
import os
import pickle
import time
from contextlib import contextmanager

import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm

from .bm25_retrieval import BM25Retrieval
from .dense_retrieval import DenseRetrieval


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BM25DenseRetrieval:
    def __init__(
        self,
        args,
        datasets,
        tokenizer,
        num_neg,
        q_encoder,
        data_path="./data/",
        caching_path="caching/",
        context_path="wikipedia_documents.json",
    ):

        self.args = args
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.num_neg = num_neg

        self.sparse_retrieval = BM25Retrieval(tokenize_fn=self.tokenizer.tokenize)
        self.dense_retrieval = DenseRetrieval(args, datasets, tokenizer=tokenizer, num_neg=self.num_neg)

        self.q_encoder = q_encoder
        with open("./data/embedding.bin", "rb") as f:
            self.p_embs = pickle.load(f)
        if torch.cuda.is_available():
            self.p_embs = torch.Tensor(self.p_embs).to("cuda")

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            self.wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in self.wiki.values()]))
        self.wiki_context_id_dict = {v["text"]: v["document_id"] for v in self.wiki.values()}
        self.wiki_id_context_dict = {v["document_id"]: v["text"] for v in self.wiki.values()}

    def get_topk_doc_id_and_score(self, query, top_k):
        es_score, es_id = self.sparse_retrieval.get_relevant_doc(query=query, k=top_k)
        return self.__rerank(query, es_id, es_score)

    def get_topk_doc_id_and_score_for_querys(self, querys, top_k):
        hybrid_ids = {}
        hybrid_scores = {}
        for i in tqdm(range(len(querys))):
            query = querys[i]
            doc_ids, scores = self.get_topk_doc_id_and_score(query, top_k)
            hybrid_ids[query] = doc_ids
            hybrid_scores[query] = scores

        return hybrid_ids, hybrid_scores

    def __rerank(self, query, es_id, es_score):
        p_embs = self.p_embs
        with torch.no_grad():
            self.q_encoder.cuda()
            self.q_encoder.eval()
            q_seqs_val = self.tokenizer(query, padding="max_length", truncation=True, return_tensors="pt").to("cuda")
            q_emb = self.q_encoder(**q_seqs_val).to("cuda")
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        rank = rank.cpu().numpy().tolist()

        es_id_score = {self.wiki_context_id_dict[self.contexts[k]]: v for k, v in zip(es_id, es_score)}

        hybrid_id_score = dict()

        for i in rank:
            dense_id = self.wiki_context_id_dict[self.contexts[i]]
            if dense_id in es_id_score:
                lin_score = dot_prod_scores[0][i].item() + es_id_score[dense_id]
                hybrid_id_score[dense_id] = lin_score

        hybrid_id_score = list(hybrid_id_score.items())
        hybrid_id_score.sort(key=lambda x: x[1], reverse=True)
        hybrid_ids = list(map(lambda x: x[0], hybrid_id_score))
        hybrid_scores = list(map(lambda x: x[1], hybrid_id_score))

        return hybrid_ids, hybrid_scores

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
            doc_scores, doc_indices = self.get_topk_doc_id_and_score_for_querys(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_indices, doc_scores = self.get_topk_doc_id_and_score_for_querys(
                    query_or_dataset["question"], top_k=topk
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

    def evaluate(self, queries, top_k):
        hybrid_ids, hybrid_scores = self.get_topk_doc_id_and_score_for_querys(queries, top_k)

        for example in tqdm(self.datasets["validation"]):

            question = example["question"]
            text = example["context"]
            corrected_prediction = 0
            for texts in hybrid_ids[question]:
                if text == self.wiki_id_context_dict[texts]:
                    print("성공")

                    corrected_prediction += 1

        print(f"top_k : {top_k} 정확도 : {corrected_prediction / 240  * 100}")
