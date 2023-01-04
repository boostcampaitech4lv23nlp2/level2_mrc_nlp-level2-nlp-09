import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, set_seed

from src.model.model import BertEncoder
from src.retrieval.dense_retrieval import DenseRetrieval
from src.utils.dataset import CustomDataset


def dpr_train(model_args, data_args, training_args):
    set_seed(training_args.seed)
    datasets = load_from_disk(data_args.dataset_name)

    model_checkpoint = "kykim/bert-kor-base"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(device)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(device)

    train_dataset = CustomDataset(datasets["train"], "../data/train_doc_scores.json", tokenizer)
    valid_dataset = CustomDataset(datasets["validation"], "../data/valid_doc_scores.json", tokenizer)

    retriever = DenseRetrieval(
        args=training_args,
        dataset=datasets,
        tokenizer=tokenizer,
        num_neg=data_args.num_neg,
    )
    retriever.train(train_dataset=train_dataset, valid_dataset=valid_dataset, p_encoder=p_encoder, q_encoder=q_encoder)
