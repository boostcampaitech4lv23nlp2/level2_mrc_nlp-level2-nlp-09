import unittest

from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

from src.utils import DataTrainingArguments, ModelArguments, check_no_error, get_training_args, set_seed


class TestUtils(unittest.TestCase):
    def test_set_seed(self):
        set_seed(400)

    def test_check_no_error(self):
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

        check_no_error(self.data_args, self.training_args, self.datasets, self.tokenizer)
