"""
This code is self-contained and is for preparing dataset to train the headline generators. Splitting all news into train/eval/test.
"""
import os
import ast
import pandas as pd
import numpy as np
import pdb

def main():
    # Input files
    news_file = "../datasets/pens/news.pkl"
    test_file = "../datasets/pens/test.pkl"

    news = pd.read_pickle(news_file)
    test = pd.read_pickle(test_file)
   
    with open(f"../datasets/pens/test.json", "w") as f:
        print(test.to_json(orient="records", lines=True), file=f, flush=False)

    # News that used in test dataset
    expanded_test_ids = []
    for i in test["posnewsID"]:
        expanded_test_ids += i

    test_ids = set(expanded_test_ids)
    all_news_ids = set(np.arange(len(news)))

    # Problematic data
    same_body_title_ids = news.index[news["title"]==news["body"]]
    nan_body_ids = news.index[news["body"].isnull()]
    same_body_title_ids = same_body_title_ids.tolist()
    nan_body_ids = nan_body_ids.tolist()
    problem_ids = set(same_body_title_ids + nan_body_ids)

    # Remove test data and problematic data
    news_ids = all_news_ids - problem_ids
    news_ids = news_ids - test_ids

    news_ids = [*news_ids,]
    valid_ids = news_ids[:5000]
    train_ids = news_ids[5000:]
    all_news_ids = [*all_news_ids,]

    train_ids = [[i] for i in train_ids]
    valid_ids = [[i] for i in valid_ids]
    test_ids = [[i] for i in test_ids]
    all_ids = [[i] for i in all_news_ids]
    expanded_test_ids = [[i] for i in expanded_test_ids]

    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    test_df = pd.DataFrame()
    expanded_test_df = pd.DataFrame()
    all_df = pd.DataFrame()
    train_df["newsID"] = train_ids
    valid_df["newsID"] = valid_ids
    test_df["newsID"] = test_ids
    all_df["newsID"] = all_ids
    expanded_test_df["newsID"] = expanded_test_ids

    # Output files
    out_dir = "../datasets/GHG"
    os.makedirs(out_dir, exist_ok=True)
    #train_df.to_pickle(out_dir+"/train.pkl")
    #valid_df.to_pickle(out_dir+"/validation.pkl")
    #test_df.to_pickle(out_dir+"/test.pkl")
    
    with open(f"{out_dir}/all_news_ids.json", "w") as f:
        print(all_df.to_json(orient="records", lines=True), file=f, flush=False)
    with open(f"{out_dir}/train.json", "w") as f:
        print(train_df.to_json(orient="records", lines=True), file=f, flush=False)
    with open(f"{out_dir}/validation.json", "w") as f:
        print(valid_df.to_json(orient="records", lines=True), file=f, flush=False)
    with open(f"{out_dir}/test.json", "w") as f:
        print(test_df.to_json(orient="records", lines=True), file=f, flush=False)
    with open(f"{out_dir}/expanded_test.json", "w") as f:
        print(expanded_test_df.to_json(orient="records", lines=True), file=f, flush=False)


    #train_df.to_csv(out_dir+"/train.csv", index=False)
    #valid_df.to_csv(out_dir+"/validation.csv", index=False)
    #test_df.to_csv(out_dir+"/test.csv", index=False)

    # Write out dataset information
    with open("../datasets/GHG/info.txt", "w") as f:
        f.write("This dataset id built for train a general headline generator.\n")
        f.write("The train and valid data are all news except the news in test posnewID and problem data.\n")
        f.write("Train: {}\nValid: {}\nTest: {}\n".format(len(train_df),len(valid_df),len(test_df)))
        f.write("There are {} problematic data (body and title is the same or NaN body)".format(len(problem_ids)))

if __name__=="__main__":
    main()
