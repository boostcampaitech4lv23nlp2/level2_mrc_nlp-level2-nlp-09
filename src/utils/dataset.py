from typing import Optional

import json
import os

from torch.utils.data import Dataset
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(
        self,
        dataset,
        bm25_dataset,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ):
        self.dataset = dataset
        self.bm25_dataset = bm25_dataset

        preprocess_data = self.preprocess_train()

        self.p_input_ids = preprocess_data[0]
        self.p_attension_mask = preprocess_data[1]
        self.p_token_type_ids = preprocess_data[2]

        self.np_input_ids = preprocess_data[3]
        self.np_attension_mask = preprocess_data[4]
        self.np_token_type_ids = preprocess_data[5]

        self.q_input_ids = preprocess_data[6]
        self.q_attension_mask = preprocess_data[7]
        self.q_token_type_ids = preprocess_data[8]

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

    def __len__(self):
        return self.p_input_ids.size()[0]

    def __getitem__(self, index):
        return (
            self.p_input_ids[index],
            self.p_attension_mask[index],
            self.p_token_type_ids[index],
            self.np_input_ids[index],
            self.np_attension_mask[index],
            self.np_token_type_ids[index],
            self.q_input_ids[index],
            self.q_attension_mask[index],
            self.q_token_type_ids[index],
        )

    def preprocess_train(
        self, data_path: Optional[str] = "../data/", context_path: Optional[str] = "wikipedia_documents.json"
    ):

        dataset = self.dataset.to_pandas()
        num_neg = 2

        pos_ctx = dataset["context"].to_list()
        questions = dataset["question"].to_list()

        neg_ctx = []
        with open(self.bm25_dataset, "r") as f:
            bm25 = json.load(f)

        for i in tqdm(range(len(pos_ctx))):
            q = questions[i]  # i 번째 question
            ground_truth = pos_ctx[i]  # 정답 문장
            cnt = num_neg  # 추가할 negative context 갯수
            answer = dataset["answers"][i]["text"][0]  # 정답
            idx = 0

            while cnt != 0:
                neg_ctx_sample = self.contexts[bm25[q][idx]]
                if (ground_truth != neg_ctx_sample) and (answer not in neg_ctx_sample):
                    # 비슷한 context를 추가하되 정답을 포함하지 않는 문장을 추가한다.
                    neg_ctx.append(self.contexts[int(bm25[q][idx])])
                    cnt -= 1
                idx += 1
                if idx == len(bm25[q]):
                    # 예외처리 ex) 정답이 전부 포함되서 추가할 문장이 없을 경우
                    idx_step = 1
                    while cnt != 0:
                        temp_neg = pos_ctx[i - idx_step]
                        # 이전에 추가된 ground truth context를 negative sample로 생성
                        neg_ctx.append(temp_neg)
                        idx_step += 1
                        cnt -= 1

        q_seqs = self.tokenizer(
            questions,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        p_seqs = self.tokenizer(
            pos_ctx,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        np_seqs = self.tokenizer(
            neg_ctx,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        max_len = np_seqs["input_ids"].size(-1)
        np_seqs["input_ids"] = np_seqs["input_ids"].view(-1, num_neg, max_len)
        np_seqs["attention_mask"] = np_seqs["attention_mask"].view(-1, num_neg, max_len)
        np_seqs["token_type_ids"] = np_seqs["token_type_ids"].view(-1, num_neg, max_len)

        return (
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            np_seqs["input_ids"],
            np_seqs["attention_mask"],
            np_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )
