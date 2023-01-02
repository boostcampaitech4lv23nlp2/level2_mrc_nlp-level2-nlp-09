import unittest

import yaml
from transformers import HfArgumentParser
from yaml.loader import SafeLoader

from src.utils import DataTrainingArguments, ModelArguments, get_training_args


class TestArguments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = HfArgumentParser((ModelArguments, DataTrainingArguments))

    def test_get_training_args(self):
        training_args = get_training_args()
        with open("./src/config/config.yml") as f:
            data = yaml.load(f, Loader=SafeLoader)
        self.assertEqual(training_args.output_dir, data["output_dir"])
