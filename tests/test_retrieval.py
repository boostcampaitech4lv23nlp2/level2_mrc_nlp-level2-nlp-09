import os
import shutil
import tempfile
import unittest

import pytest
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

from src.retrieval import SparseRetrieval
from src.utils import DataTrainingArguments, ModelArguments, get_training_args

test_dataset = {
    "title": ["테스트"],
    "context": ["서울은 국제적인 도시이다. 서울의 GDP는 세계 4위이다. 서울에는 국밥 맛집이 많다."],
    "question": ["서울의 GDP는 세계 몇 위인가?"],
    "id": ["1"],
    "answers": [{"answer_start": [25], "text": ["세계 4위"]}],
}

test_sentence = "서울의 GDP는 세계 몇 위인가?"


class TestRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
        self.model_args, self.data_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
        self.training_args = get_training_args()
        self.data_args.dataset_name = "data/train_dataset"
        self.datasets = load_from_disk(self.data_args.dataset_name)
        self.config = AutoConfig.from_pretrained(
            self.model_args.model_name_or_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            use_fast=True,
        )
        self.data_path = "data"
        self.context_path = "wikipedia_documents.json"
        self.retriever = SparseRetrieval(
            tokenize_fn=self.tokenizer, data_path=self.data_path, context_path=self.context_path
        )

    def test_get_sparse_embedding(self):
        retriever = self.retriever
        retriever.get_sparse_embedding()
        self.assertEqual(retriever.p_embedding.shape, (len(retriever.contexts), retriever.max_features))

    def test_retrieve(self):

        self.data_args.top_k_retrieval = 100
        retriever = self.retriever
        retriever.get_sparse_embedding()
        df = retriever.retrieve(self.datasets["validation"], topk=self.data_args.top_k_retrieval)
        # print(len(df.loc[0]["context"].split("\n\n")))
        self.assertEqual(len(self.datasets["validation"]), df.shape[0])

        # To Testing query_or_dataset is str
        df = retriever.retrieve(test_sentence, topk=self.data_args.top_k_retrieval)

    @pytest.mark.skip()
    def test_retrieve_faiss(self):
        retriever = self.retriever
        retriever.get_sparse_embedding()
        retriever.build_faiss()
        df = retriever.retrieve_faiss(self.datasets["validation"], topk=self.data_args.top_k_retrieval)
        self.assertEqual(len(self.datasets["validation"]), df.shape[0])

        # To Testing query_or_dataset is str
        df = retriever.retrieve_faiss(test_sentence, topk=self.data_args.top_k_retrieval)

    def test_get_relevant_doc(self):
        test_sentence = test_dataset["question"][0]
        topk = 10
        retriever = self.retriever
        retriever.get_sparse_embedding()
        scores, indices = retriever.get_relevant_doc(test_sentence, topk)
        self.assertEqual((topk, topk), (len(scores), len(indices)))

    def test_get_relevant_doc_bulk(self):
        test_sentences = test_dataset["question"]
        topk = 10

        retriever = self.retriever
        retriever.get_sparse_embedding()
        scores, indices = retriever.get_relevant_doc_bulk(test_sentences, topk)
        self.assertEqual(len(test_dataset["question"]), len(scores))
        self.assertEqual(topk, len(scores[0]))

    def test_get_faiss(self):
        retriever = self.retriever
        retriever.get_sparse_embedding()
        retriever.build_faiss()
        self.assertIsNotNone(retriever.indexer)

    def test_get_relevant_doc_faiss(self):
        test_sentence = test_dataset["question"][0]
        topk = 10
        retriever = self.retriever
        retriever.get_sparse_embedding()
        retriever.build_faiss()

        D, I = retriever.get_relevant_doc_faiss(test_sentence, topk)
        self.assertEqual((topk, topk), (len(D), len(I)))

    def test_get_relevant_doc_bulk_faiss(self):
        test_sentences = test_dataset["question"]
        topk = 10
        retriever = self.retriever
        retriever.get_sparse_embedding()
        retriever.build_faiss()
        D, I = retriever.get_relevant_doc_bulk_faiss(test_sentences, topk)
        self.assertEqual(len(test_dataset["question"]), len(D))
        self.assertEqual(topk, len(D[0]))

    @pytest.mark.skip()
    def test_build_passage_embedding(self):
        wiki_path = os.path.join(self.data_path, self.context_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            if os.path.exists(temp_dir):
                shutil.copyfile(wiki_path, os.path.join(temp_dir, "wikipedia_documents.json"))
                retriever = SparseRetrieval(
                    tokenize_fn=self.tokenizer.tokenize, data_path=temp_dir, context_path="wikipedia_documents.json"
                )
                retriever.get_sparse_embedding()
                self.assertEqual(retriever.p_embedding.shape, (len(retriever.contexts), retriever.max_features))

    @pytest.mark.skip()
    def test_build_faiss(self):
        wiki_path = os.path.join(self.data_path, self.context_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            if os.path.exists(temp_dir):
                shutil.copyfile(wiki_path, os.path.join(temp_dir, "wikipedia_documents.json"))
                retriever = SparseRetrieval(
                    tokenize_fn=self.tokenizer.tokenize, data_path=temp_dir, context_path="wikipedia_documents.json"
                )
                retriever.get_sparse_embedding()
                retriever.build_faiss()
                self.assertIsNotNone(retriever.indexer)
