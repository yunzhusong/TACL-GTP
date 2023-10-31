""" 
Base function for Calculating ROUGE score
"""
import pdb
import nltk
import pandas as pd
import numpy as np
from datasets import load_metric

# Metric
metric = load_metric("rouge")


def compute_metrics(preds, labels):
    """
    preds: list of string
    labels: list of string
    """

    def _postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

    # Some simple post-processing
    decoded_preds, decoded_labels = _postprocess_text(preds, labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()

    result = {k: round(v, 4) for k, v in result.items()}
    print(result)

    return result


if __name__=="__main__":

    data_df = pd.read_pickle("../datasets/specialize_own/finetune/4/test.pkl")
    _preds = data_df["posnews"].tolist()
    _labels = data_df["title"].tolist()

    compute_metrics(_preds, _labels)

