import unittest

from transformers import HfArgumentParser

from src import inference, train
from src.utils import DataTrainingArguments, ModelArguments, get_training_args


class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
        self.model_args, self.data_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
        self.training_args = get_training_args()

    def test_train(self):
        self.training_args.do_train = True
        self.training_args.do_predict = False
        self.data_args.dataset_name = "tests/test_data/train_dataset"
        train(self.model_args, self.data_args, self.training_args)

    def test_predict(self):
        self.training_args.do_train = False
        self.training_args.do_predict = True
        self.data_args.dataset_name = "tests/test_data/test_dataset"
        inference(self.model_args, self.data_args, self.training_args)
