import unittest

import pandas as pd
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

from src.postprocess import post_processing_function, postprocess_qa_predictions
from src.utils import DataTrainingArguments, ModelArguments, get_training_args


class TestPostProcess(unittest.TestCase):
    def setUp(self):
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
        self.model_args, self.data_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
        self.training_args = get_training_args()
        self.datasets = load_from_disk("data/train_dataset")
        self.test_datasets = load_from_disk("data/test_dataset")
        self.config = AutoConfig.from_pretrained(
            self.model_args.model_name_or_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            use_fast=True,
        )

    def test_post_processing_function(self):
        self.training_args.do_predict = True
        self.training_args.do_eval = False

        examples = load_from_disk("tests/test_data/examples")
        features = load_from_disk("tests/test_data/features")
        start_logits = pd.read_csv("tests/test_start_logits.csv", index_col=False).to_numpy()
        end_logits = pd.read_csv("tests/test_end_logits.csv", index_col=False).to_numpy()
        predictions = [start_logits, end_logits]
        output = post_processing_function(
            examples=examples,
            features=features,
            predictions=predictions,
            training_args=self.training_args,
            data_args=self.data_args,
        )
        self.assertEqual("스탠더드 오일", output[0]["prediction_text"])

        self.training_args.do_predict = False
        self.training_args.do_eval = True
        output = post_processing_function(
            examples=examples,
            features=features,
            predictions=predictions,
            training_args=self.training_args,
            data_args=self.data_args,
        )
        self.assertIsNotNone(output)

    def test_postprocess_version_2(self):
        examples = load_from_disk("tests/test_data/examples")
        features = load_from_disk("tests/test_data/features")
        start_logits = pd.read_csv("tests/test_start_logits.csv", index_col=False).to_numpy()
        end_logits = pd.read_csv("tests/test_end_logits.csv", index_col=False).to_numpy()
        predictions = [start_logits, end_logits]

        output = postprocess_qa_predictions(examples, features, predictions, version_2_with_negative=True)
        self.assertIsNotNone(output)
