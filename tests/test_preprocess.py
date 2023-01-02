import copy
import unittest

from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

from src.preprocess import PreProcessor
from src.utils import DataTrainingArguments, ModelArguments, get_training_args

test_dataset = {
    "title": ["테스트"],
    "context": ["서울은 국제적인 도시이다. 서울의 GDP는 세계 4위이다. 서울에는 국밥 맛집이 많다."],
    "question": ["서울의 GDP는 세계 몇 위인가?"],
    "id": ["1"],
    "answers": [{"answer_start": [25], "text": ["세계 4위"]}],
}


class TestPreProcess(unittest.TestCase):
    def setUp(self) -> None:
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

    def test_prepare_train_features(self):

        test_data_args = copy.deepcopy(self.data_args)

        test_data_args.max_seq_length = 32
        test_data_args.doc_stride = 8

        preprocessor = PreProcessor(
            datasets=self.datasets, tokenizer=self.tokenizer, data_args=test_data_args, training_args=self.training_args
        )

        cls_id = self.tokenizer("[CLS]", add_special_tokens=False)["input_ids"]
        sep_id = self.tokenizer("[SEP]", add_special_tokens=False)["input_ids"]
        expect_context_id = self.tokenizer(test_dataset["context"], add_special_tokens=False)["input_ids"][0]
        expect_question_id = self.tokenizer(test_dataset["question"], add_special_tokens=False)["input_ids"][0]

        trunc_context_id_0 = expect_context_id[: test_data_args.max_seq_length - len(expect_question_id) - 3]
        trunc_context_id_1 = expect_context_id[len(trunc_context_id_0) - test_data_args.doc_stride :]
        expect_output = [cls_id + expect_question_id + sep_id + trunc_context_id_0 + sep_id] + [
            cls_id + expect_question_id + sep_id + trunc_context_id_1 + sep_id
        ]

        test_output = preprocessor.prepare_train_features(test_dataset)
        self.assertEqual(expect_output, test_output["input_ids"])

    def test_prepare_validation_features(self):
        preprocessor = PreProcessor(
            datasets=self.datasets, tokenizer=self.tokenizer, data_args=self.data_args, training_args=self.training_args
        )
        test_output = preprocessor.prepare_validation_features(test_dataset)
        self.assertEqual(list(test_output.keys()), ["input_ids", "attention_mask", "offset_mapping", "example_id"])

    def test_no_answer(self):
        test_no_answer_dataset = copy.deepcopy(test_dataset)
        test_no_answer_dataset["answers"] = [{"answer_start": [], "text": [""]}]

        preprocessor = PreProcessor(
            datasets=self.datasets, tokenizer=self.tokenizer, data_args=self.data_args, training_args=self.training_args
        )
        cls_id = self.tokenizer("[CLS]", add_special_tokens=False)["input_ids"]
        test_output = preprocessor.prepare_train_features(test_no_answer_dataset)
        self.assertEqual(cls_id, test_output["start_positions"])

    def test_get_train_dataset(self):
        preprocessor = PreProcessor(
            datasets=self.datasets, tokenizer=self.tokenizer, data_args=self.data_args, training_args=self.training_args
        )
        train_dataset = preprocessor.get_train_dataset()
        self.assertEqual(
            list(train_dataset.features), ["input_ids", "attention_mask", "start_positions", "end_positions"]
        )

    def test_get_eval_dataset(self):
        preprocessor = PreProcessor(
            datasets=self.datasets, tokenizer=self.tokenizer, data_args=self.data_args, training_args=self.training_args
        )
        train_dataset = preprocessor.get_eval_dataset()
        self.assertEqual(list(train_dataset.features), ["input_ids", "attention_mask", "offset_mapping", "example_id"])

    def test_get_column_names(self):
        test_training_args = copy.deepcopy(self.training_args)
        test_data_args = copy.deepcopy(self.data_args)
        test_training_args.do_train = True
        preprocessor = PreProcessor(
            datasets=self.datasets, tokenizer=self.tokenizer, data_args=test_data_args, training_args=test_training_args
        )
        train_column_names = preprocessor.get_column_names()
        self.assertEqual(
            train_column_names, ["title", "context", "question", "id", "answers", "document_id", "__index_level_0__"]
        )

        test_training_args.do_train = False
        test_data_args.dataset_name = "data/test_dataset"
        preprocessor = PreProcessor(
            datasets=self.test_datasets,
            tokenizer=self.tokenizer,
            data_args=test_data_args,
            training_args=test_training_args,
        )
        validation_column_names = preprocessor.get_column_names()
        self.assertEqual(validation_column_names, ["question", "id"])
