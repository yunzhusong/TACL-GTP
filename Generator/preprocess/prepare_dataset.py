import os
import pdb
import pandas as pd
import numpy as np

def main():
    """
    Retrieve the topics for each test data
    """
    ori_news_path = "../datasets/original_PENS/news.tsv"
    test_path = "../datasets/specialize_own/test.pkl"
    
    news_df = pd.read_csv(ori_news_path, delimiter='\t')
    test_df = pd.read_pickle(test_path)


    topics = news_df["Topic"]
    ids = np.array(test_df["posnewsID"].tolist()).reshape(-1)
    test_topics = topics.iloc[ids]
    test_df["topic"] = test_topics.tolist()
    test_df.to_pickle(test_path)

if __name__=="__main__":
    main()
