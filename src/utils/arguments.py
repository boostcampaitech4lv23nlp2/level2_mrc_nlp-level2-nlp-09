from typing import Optional

from dataclasses import dataclass, field

from transformers import TrainingArguments


def get_training_args(
    output_dir="./results",
    save_total_limit=5,
    save_strategy="epoch",
    num_train_epochs=1,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=0,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    do_train=True,
    do_eval=True,
    do_predict=False,
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
    )
    return training_args


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(default=64, metadata={"help": "Define how many clusters to use for faiss."})
    top_k_retrieval: int = field(
        default=10,
        metadata={"help": "Define how many top-k passages to retrieve based on similarity."},
    )
    use_faiss: bool = field(default=False, metadata={"help": "Whether to build with faiss"})
