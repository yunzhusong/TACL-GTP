"""
This code is self-contained and for converting the pickle file to json file
"""
import os
import pdb
import json
import pandas as pd

import textacy
from tqdm import tqdm


def main():
   
    out_dir = "../datasets/pens"

    # Keyword Extraction
    news = pd.read_pickle("../datasets/pens/news.pkl")
    news = news.fillna("")

    titles = news["title"].tolist()

    body_keywords = []
    for b in tqdm(news["body"].tolist(), desc="Extracting keywords for bodies"):
        doc = textacy.make_spacy_doc(b, lang="en_core_web_sm")
        body_keywords.append([kps for kps, weights in textacy.extract.keyterms.textrank(doc, normalize="lemma", topn=10)])

    title_keywords = []
    for t in tqdm(news["title"].tolist(), desc="Extracting keywords for titles"):
        doc = textacy.make_spacy_doc(t, lang="en_core_web_sm")
        title_keywords.append([kps for kps, weights in textacy.extract.keyterms.textrank(doc, normalize="lemma", topn=10)])

    news["title_keywords"] = title_keywords
    news["body_keywords"] = body_keywords

    news.to_pickle("../datasets/pens/news_with_keyword.pkl")


    test_df = pd.read_pickle("../datasets/pens/test.pkl")
    with open(f"{out_dir}/test.json", "w") as f:
        print(test_df.to_json(orient="records", lines=True), file=f, flush=False)

    train_df = pd.read_pickle("../datasets/pens/train.pkl")
    with open(f"{out_dir}/train.json", "w") as f:
        print(train_df.to_json(orient="records", lines=True), file=f, flush=False)

    val_df = pd.read_pickle("../datasets/pens/validation.pkl")
    with open(f"{out_dir}/validation.json", "w") as f:
        print(val_df.to_json(orient="records", lines=True), file=f, flush=False)

if __name__=="__main__":
    main()
