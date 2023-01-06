from typing import List, Optional, Tuple, Union

import json
import os
import pickle
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoTokenizer


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BM25Retrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "./data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            self.wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in self.wiki.values()]))  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.bm25 = BM25Okapi(self.contexts, tokenize_fn)
        self.tokenize_fn = tokenize_fn

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
                doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset["question"], k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join([self.contexts[pid] for pid in doc_indices["question"]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        if os.path.isfile("../../data/train_doc_scores.bin"):

            with open("../../data/train_doc_indices.bin") as f:
                doc_indices = pickle.load(f)

            with open("../../data/train_doc_scores.bin") as f:
                doc_scores = pickle.load(f)

        else:
            doc_scores = {}
            doc_indices = {}
            for query in queries:
                scores = self.bm25.get_scores(self.tokenize_fn(query))
                sorted_scores = np.argsort(scores)[::-1]
                doc_scores[query] = scores[sorted_scores][:k].tolist()
                doc_indices[query] = sorted_scores.tolist()[:k]

            with open("../../data/train_doc_scores.bin") as f:
                pickle.dump(doc_scores)
            with open("../../data/train_doc_indices.bin") as f:
                pickle.dump(doc_indices)

        return doc_scores, doc_indices

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        scores = self.bm25.get_scores(self.tokenize_fn(query))
        sorted_scores = np.argsort(scores)[::-1]
        doc_scores = scores[sorted_scores][:k].tolist()
        doc_indices = sorted_scores.tolist()[:k]

        return doc_scores, doc_indices

    def evaluate(self, datasets, top_k):
        for k in top_k:

            doc_scores, doc_indices = self.get_relevant_doc_bulk(datasets["validation"]["question"], k=k)
            corrected_prediction = 0
            for row, example in enumerate(tqdm(datasets["validation"])):

                text = example["context"]
                for texts in set(doc_indices[row][:k]):
                    if text == self.contexts[texts]:

                        corrected_prediction += 1

            print(f"top_k : {k} 정확도 : {corrected_prediction / 240  * 100:.2f}%")


if __name__ == "__main__":

    datasets = load_from_disk("../../data/train_dataset")
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
    retriever = BM25Retrieval(
        tokenize_fn=tokenizer.tokenize, data_path="../../data/", context_path="wikipedia_documents.json"
    )
    retriever.get_relevant_doc_bulk(datasets["train"]["question"], k=100)
