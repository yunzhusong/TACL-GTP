"""
This code is self-contained and is for retrieving 
target related sentences from source.
"""
import os
import re
import sys
import pdb
import nltk
import math
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_metric 

sys.path.insert(1, os.getcwd())

tqdm.pandas()
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


#### Supporting function ####

def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    words = sum(sentences, [])
    return _get_ngrams(n, words)

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def greedy_selection(doc_sent_list, ref_sent_list, max_sent_num, rouge_type):

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    ref = sum(ref_sent_list, [])
    ref = _rouge_clean(' '.join(ref)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]

    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [ref])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [ref])

    rouge_scores = []
    for i in range(len(sents)):
        candidates_1 = [evaluated_1grams[i]]
        rouge_1 = cal_rouge(evaluated_1grams[i], reference_1grams)[rouge_type]
        rouge_2 = cal_rouge(evaluated_2grams[i], reference_2grams)[rouge_type]
        rouge_scores.append(rouge_1 + rouge_2)

    rouge_scores = np.array(rouge_scores)
    selected = np.argpartition(rouge_scores, -max_sent_num)[-max_sent_num:]
    return selected

#### Main function ####

def retrieve(df_news, tgt_column_name, src_column_name):
    """
    bodys: list of news body
    titles: list of news titlte
    """
    rouge_type = 'r'
    max_sent_num = 3
    tgt_name = tgt_column_name
    src_name = src_column_name

    def _retrieve(data):
        title = data[tgt_name]
        body = data[src_name]
        if type(body) != str and math.isnan(body):
            body = title
        sents = nltk.sent_tokenize(body)
        sents_tokens = [nltk.word_tokenize(sent) for sent in sents]
        title_tokens = [nltk.word_tokenize(title)]

        _max_sent_num = min(len(sents), max_sent_num)
        selected_indices = greedy_selection(
            sents_tokens,
            title_tokens,
            _max_sent_num,
            rouge_type,
        )
        selected_sents = []
        for idx in np.sort(selected_indices):
            selected_sents.append(sents[idx])
        data[f"retrieved"] = ' '.join(selected_sents)
        return data

    df_news = df_news.progress_apply(lambda d: _retrieve(d), axis=1)
    return df_news

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../datasets/pens/news_w_parsed.pkl",
                       help="The file sould be a pickle file")
    parser.add_argument("--output_file", type=str, default="../datasets/pens/news_w_parsed_retri.pkl",
                       help="The output file name")
    parser.add_argument("--target_column_name", type=str, default="title",
                       help="column name of extraction target",
                       )
    parser.add_argument("--source_column_name", type=str, default="body",
                       help="column name of extraction source")

    args = parser.parse_args()

    news = pd.read_pickle(args.input_file)
    news_retrieved = retrieve(news, args.target_column_name, args.source_column_name)
    news_retrieved.to_pickle(args.output_file)

if __name__=="__main__":
    main()
