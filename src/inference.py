"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""


from typing import Callable, Dict, List

import logging
import sys

from datasets import Dataset, DatasetDict, load_from_disk, load_metric
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    set_seed,
)

from .model.model import BertEncoder
from .preprocess import PreProcessor
from .retrieval.bm25_dense_retrieval import BM25DenseRetrieval
from .retrieval.bm25_retrieval import BM25Retrieval
from .retrieval.dense_retrieval import DenseRetrieval
from .retrieval.retrieval import SparseRetrieval
from .trainer_qa import QuestionAnsweringTrainer
from .utils import DataTrainingArguments, ModelArguments

logger = logging.getLogger(__name__)

metric = load_metric("squad")


def compute_metrics(p: EvalPrediction) -> Dict:
    return metric.compute(predictions=p.predictions, references=p.label_ids)


def evaluate(model_args, data_args, training_args):

    datasets = load_from_disk("./data/train_dataset")
    tokenizer = AutoTokenizer.from_pretrained("kykim/bert-kor-base")
    q_encoder = "./output/q_encoder"
    q_encoder = BertEncoder.from_pretrained(q_encoder)

    if model_args.bm25:
        retriever = BM25Retrieval(
            tokenize_fn=tokenizer.tokenize, data_path="./data/", context_path="wikipedia_documents.json"
        )
        retriever.evaluate(datasets, [1, 5, 10, 15, 20, 25, 30, 50, 100])
    elif model_args.dpr:

        bm25_dense_retrieval = BM25DenseRetrieval(training_args, datasets, tokenizer, 2, q_encoder)
        bm25_dense_retrieval.evaluate(datasets["validation"]["question"], 100)


def inference(model_args, data_args, training_args):
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    data_args.use_faiss = False
    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval and model_args.bm25 and not model_args.dpr:
        datasets = run_bm25_retrieval(
            tokenizer.tokenize,
            datasets,
            training_args,
            data_args,
        )
    elif data_args.eval_retrieval and model_args.dpr and not model_args.bm25:
        tokenizer = AutoTokenizer.from_pretrained("kykim/bert-kor-base")
        datasets = run_dense_retrieval(tokenizer, datasets, training_args, data_args)
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            use_fast=True,
        )

    elif data_args.eval_retrieval and model_args.dpr and model_args.bm25:
        tokenizer = AutoTokenizer.from_pretrained("kykim/bert-kor-base")
        datasets = run_bm25_dense_retrieval(tokenizer, datasets, training_args, data_args)
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            use_fast=True,
        )

    else:
        datasets = run_sparse_retrieval(
            tokenizer.tokenize,
            datasets,
            training_args,
            data_args,
        )
    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_bm25_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    retriever = BM25Retrieval(tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path)

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(datasets["validation"], topk=data_args.top_k_retrieval)
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    datasets = DatasetDict({"validation": Dataset.from_pandas(df)})
    return datasets


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path)
    retriever.get_sparse_embedding()

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(datasets["validation"], topk=data_args.top_k_retrieval)
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    datasets = DatasetDict({"validation": Dataset.from_pandas(df)})
    return datasets


def run_dense_retrieval(
    tokenizer,
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    p_encoder = BertEncoder.from_pretrained(training_args.output_dir + "/p_encoder")
    q_encoder = BertEncoder.from_pretrained(training_args.output_dir + "/q_encoder")
    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = DenseRetrieval(
        training_args,
        datasets,
        tokenizer,
        data_args.num_neg,
        data_path=data_path,
        context_path=context_path,
    )

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(datasets["validation"], topk=data_args.top_k_retrieval)
    else:
        df = retriever.retrieve(p_encoder, q_encoder, datasets["validation"], topk=data_args.top_k_retrieval)

    datasets = DatasetDict({"validation": Dataset.from_pandas(df)})
    return datasets


def run_bm25_dense_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    q_encoder = BertEncoder.from_pretrained(training_args.output_dir + "/q_encoder")
    retriever = BM25DenseRetrieval(
        training_args,
        datasets,
        tokenize_fn,
        data_args.num_neg,
        q_encoder,
        data_path=data_path,
        context_path=context_path,
    )

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(datasets["validation"], topk=data_args.top_k_retrieval)
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    datasets = DatasetDict({"validation": Dataset.from_pandas(df)})
    return datasets


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> None:

    preprocessor = PreProcessor(
        datasets=datasets, tokenizer=tokenizer, data_args=data_args, training_args=training_args
    )

    eval_dataset = datasets["validation"]

    eval_dataset = preprocessor.get_eval_dataset()

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        data_args=data_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    # eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        trainer.predict(test_dataset=eval_dataset, test_examples=datasets["validation"])

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print("No metric can be presented because there is no correct answer given. Job done!")
