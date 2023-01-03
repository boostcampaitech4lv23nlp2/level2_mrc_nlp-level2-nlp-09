import json

import pandas as pd
import streamlit as st
from datasets import load_from_disk

datasets = load_from_disk("data/train_dataset/")
validation_data = datasets["validation"]
with open("output/predictions.json", "r") as f:
    predictions_dict = json.load(f)

q_ids = []
answers = []
predictions = []
questions = []
contexts = []


def get_answer_text(row):
    return row["text"][0]


validation_df = validation_data.to_pandas()
validation_df["answer"] = validation_df["answers"].apply(get_answer_text)
prediction_df = pd.DataFrame(predictions_dict.items(), columns=["id", "prediction"])

merge_df = pd.merge(left=validation_df, right=prediction_df, how="inner", on="id")
compare_df = merge_df[merge_df["answer"] != merge_df["prediction"]]
compare_df = compare_df[["id", "question", "answer", "prediction", "context"]]
st.dataframe(compare_df)
st.text(len(validation_df))
st.text(len(compare_df))

sentence = st.text_input("Input the sentence")
st.text("모델 추론 결과")
st.text("리트리버 결과")
