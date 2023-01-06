from typing import Optional

from dataclasses import dataclass, field

import yaml
from transformers import TrainingArguments
from yaml.loader import SafeLoader

with open("./src/config/config.yml") as f:
    data = yaml.load(f, Loader=SafeLoader)


def get_training_args(
    output_dir=data["output_dir"],
    save_total_limit=data["save_total_limit"],
    save_strategy=data["save_strategy"],
    num_train_epochs=data["num_train_epochs"],
    learning_rate=data["learning_rate"],
    per_device_train_batch_size=data["per_device_train_batch_size"],
    per_device_eval_batch_size=data["per_device_eval_batch_size"],
    warmup_steps=data["warmup_steps"],
    weight_decay=data["weight_decay"],
    logging_dir=data["logging_dir"],
    logging_steps=data["logging_steps"],
    evaluation_strategy=data["evaluation_strategy"],
    do_train=data["do_train"],
    do_eval=data["do_eval"],
    do_predict=data["do_predict"],
    report_to=["wandb"],
    fp16=data["fp16"],
):
    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        save_total_limit=save_total_limit,  # number of total save model.
        save_strategy=save_strategy,
        num_train_epochs=num_train_epochs,  # total number of training epochs
        learning_rate=learning_rate,  # learning_rate
        per_device_train_batch_size=per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=per_device_eval_batch_size,  # batch size for evaluation
        warmup_steps=warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=weight_decay,  # strength of weight decay
        logging_dir=logging_dir,  # directory for storing logs
        logging_steps=logging_steps,  # log saving step.
        evaluation_strategy=evaluation_strategy,  # evaluation strategy to adopt during training
        do_train=do_train,
        do_eval=do_eval,
        do_predict=do_predict,
        report_to=report_to,
        fp16=fp16,
    )
    return training_args


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=data["model_name_or_path"],
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=data["config_name"],
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=data["tokenizer_name"],
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    dpr: bool = field(default=data["dpr"])
    bm25: bool = field(default=data["bm25"])


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=data["dataset_name"],
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=data["overwrite_cache"],
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=data["preprocessing_num_workers"],
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=data["max_seq_length"],
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=data["pad_to_max_length"],
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=data["doc_stride"],
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    max_answer_length: int = field(
        default=data["max_answer_length"],
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=data["eval_retrieval"],
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=data["num_clusters"], metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=data["top_k_retrieval"],
        metadata={"help": "Define how many top-k passages to retrieve based on similarity."},
    )
    use_faiss: bool = field(default=data["use_faiss"], metadata={"help": "Whether to build with faiss"})
    num_neg: int = field(default=data["num_neg"])
    evaluate: bool = field(default=False)
    do_dpr_train: bool = field(default=False)
