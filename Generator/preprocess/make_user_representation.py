import os
import numpy as np
import pandas as pd
import pdb
from simcse import SimCSE
from sentence_transformers import SentenceTransformer

#model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased") # NOTE: for user data sampling
model = SentenceTransformer('sentence-transformers/msmarco-roberta-base-v2')

def main():
    news = pd.read_pickle("../datasets/pens/news.pkl")
    titles = news['title'].tolist()[:10]
    feats = model.encode(titles)
    pdb.set_trace()
    np.savez("../datasets/user_feat/roberta/news_features.npz", news_features=feats)


if __name__=="__main__":
    main()
