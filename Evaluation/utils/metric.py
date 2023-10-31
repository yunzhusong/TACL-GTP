""" Evaluation metrics from original PENS. """
import pdb
import numpy as np
from sklearn.metrics import roc_auc_score

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def metric_recommend(preds, labels):

    ndcg_5_list = []
    ndcg_10_list = []
    mrr_list = []
    auc_list = []

    for pred, label in zip(preds, labels):
        mask = label!=-100
        pred = pred[mask]
        label = label[mask]

        ndcg_5_list.append(ndcg_score(label, pred, k=5))
        ndcg_10_list.append(ndcg_score(label, pred, k=10))
        mrr_list.append(mrr_score(label, pred))
        auc_list.append(roc_auc_score(label, pred))
    
    result = {
        "nDCG@5": np.mean(ndcg_5_list),
        "nDCG@10": np.mean(ndcg_10_list),
        "MRR": np.mean(mrr_list),
        "AUC": np.mean(auc_list),
    }

    return result

def zero():
    return 0

def update_model_output(outputs, outputs_new, force_replace=True, ignore_keys_for_log=[]):

    existing_keys = outputs.keys()
    for k, v in outputs_new.items():
        if k in ignore_keys_for_log:
            continue

        if force_replace:
            outputs.__setitem__(k, v)
        elif k not in existing_keys:
            outputs.__setitem__(k, v)
        else:
            raise ValueError("Both outputs and outputs_new are not None in {}".format(k))

    return outputs



