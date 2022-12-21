from transformers import HfArgumentParser, TrainingArguments

from src import inference, train
from src.utils import DataTrainingArguments, ModelArguments

if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.do_train:
        train(model_args, data_args, training_args)
    if training_args.do_predict:
        inference(model_args, data_args, training_args)
