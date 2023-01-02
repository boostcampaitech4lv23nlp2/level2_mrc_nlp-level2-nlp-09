import torch
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed

from model.model import BertEncoder
from retrieval.dense_retrieval import DenseRetrieval
from utils import DataTrainingArguments, ModelArguments, get_training_args

parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
model_args, data_args = parser.parse_args_into_dataclasses()
training_args = get_training_args()

set_seed(training_args.seed)

datasets = load_from_disk(data_args.dataset_name)

model_checkpoint = "kykim/bert-kor-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
config = AutoConfig.from_pretrained(model_checkpoint)
p_encoder = BertEncoder.from_pretrained(config).to(device)
q_encoder = BertEncoder.from_pretrained(config).to(device)
retriever = DenseRetrieval(
    args=training_args,
    dataset=datasets,
    tokenizer=tokenizer,
    num_neg=data_args.num_neg,
    p_encoder=p_encoder,
    q_encoder=q_encoder,
)
retriever.train()
