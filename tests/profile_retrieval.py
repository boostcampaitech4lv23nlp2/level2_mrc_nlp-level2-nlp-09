import sys
from cProfile import Profile
from pstats import Stats

sys.path.append(".")

from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

from src.inference import run_sparse_retrieval
from src.utils import DataTrainingArguments, ModelArguments, get_training_args


class ProfileRetrieval:
    def __init__(self):
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

    def test(self):
        datasets = run_sparse_retrieval(self.tokenizer.tokenize, self.datasets, self.training_args, self.data_args)
        return datasets


test_object = ProfileRetrieval()
profiler = Profile()
profiler.runcall(test_object.test)

stats = Stats(profiler)
stats.strip_dirs()
stats.sort_stats("tottime")
stats.print_stats(0.05)
