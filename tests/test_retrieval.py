import unittest

from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

from src.retrieval import SparseRetrieval
from src.utils import DataTrainingArguments, ModelArguments, get_training_args


class TestRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
        self.model_args, self.data_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
        self.training_args = get_training_args()
        self.data_args.dataset_name = "data/test_dataset"
        self.datasets = load_from_disk(self.data_args.dataset_name)
        self.config = AutoConfig.from_pretrained(
            self.model_args.model_name_or_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            use_fast=True,
        )

    def test_get_sparse_embedding(self):
        retriever = SparseRetrieval(
            tokenize_fn=self.tokenizer, data_path="data", context_path="wikipedia_documents.json"
        )
        retriever.get_sparse_embedding()
        print(len(retriever.contexts))
        self.assertEqual(retriever.p_embedding.shape, (len(retriever.contexts), retriever.max_features))

    def test_retrieve(self):

        self.data_args.top_k_retrieval = 100
        retriever = SparseRetrieval(
            tokenize_fn=self.tokenizer, data_path="data", context_path="wikipedia_documents.json"
        )
        retriever.get_sparse_embedding()
        df = retriever.retrieve(self.datasets["validation"], topk=self.data_args.top_k_retrieval)
        # print(len(df.loc[0]["context"].split("\n\n")))
        self.assertEqual(
            (len(self.datasets["validation"]), len(self.datasets["validation"].column_names) + 1), df.shape
        )
