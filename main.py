import sys

from transformers import HfArgumentParser

from src import dpr_train, inference, train
from src.utils import DataTrainingArguments, ModelArguments, get_training_args

if __name__ == "__main__":
    sys.path.insert(0, "./src")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    training_args = get_training_args()
    if data_args.do_dpr_train:
        dpr_train(model_args, data_args, training_args)
    if training_args.do_train:
        train(model_args, data_args, training_args)
    if training_args.do_predict:
        inference(model_args, data_args, training_args)
