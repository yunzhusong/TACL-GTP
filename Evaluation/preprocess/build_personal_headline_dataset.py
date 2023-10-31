"""
The code is self-contained and is for pretraining dataset to train the personalized headline generators. 

Type 1 - Expansion: Flatten the positive samples of users to create the (news, click history) pairs.
Type 2 - Collection: Collect clicked history of users with the same positive news
"""
import os
import pdb
import json

import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
import numpy as np
import pprint

def main():
    
    # Input files
    test_df = pd.read_pickle("../datasets/pens/test.pkl")
    train_df = pd.read_pickle("../datasets/pens/train.pkl")
    valid_df = pd.read_pickle("../datasets/pens/validation.pkl")

    # Expand the data according to the positive news
    valid_df, valid_info = dataframe_expansion(valid_df)
    train_df, train_info = dataframe_expansion(train_df)
    test_df, test_info = dataframe_expansion(test_df, test_data=True)

    info = {}
    info["Original dataset with positive news expansion"] = {
        "train": train_info,
        "validation": valid_info,
        "test":test_info,
    }

    # Merge the training and validation dataframe to better collect 
    # the data with the same positive news
    all_df = pd.concat([train_df, valid_df])

    # Collect the data with the same postive news
    collected_df, collected_info = dataframe_collection(all_df)

    # Divide the all_df into training and validation
    num_validation = 2000
    collected_train_df = collected_df.iloc[:-num_validation]
    collected_valid_df = collected_df.iloc[-num_validation:]

    info["After collecting the train/validation data with the same positive news"] = {
        "train_and_validation": collected_info,
        "num_train": len(collected_train_df),
        "num_validation": len(collected_valid_df),
    }

    # Output files
    out_dir = "../datasets/PHG"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/expanded_train.json", "w") as f:
        print(train_df.to_json(orient="records", lines=True), file=f, flush=False)
    with open(f"{out_dir}/expanded_validation.json", "w") as f:
        print(valid_df.to_json(orient="records", lines=True), file=f, flush=False)
    with open(f"{out_dir}/expanded_test.json", "w") as f:
        print(test_df.to_json(orient="records", lines=True), file=f, flush=False)

    with open(f"{out_dir}/collected_train.json", "w") as f:
        print(collected_train_df.to_json(orient="records", lines=True), file=f, flush=False)
    with open(f"{out_dir}/collected_validation.json", "w") as f:
        print(collected_valid_df.to_json(orient="records", lines=True), file=f, flush=False)

    with open(f"{out_dir}/info.txt", "w") as f:
        pprint.pprint(info, indent=4, sort_dicts=False, stream=f)


# Type 1. A postive news will be expanded to multiple users
def dataframe_expansion(df, test_data=False):
    """
    Expand dataframe according to the number of positive news.
    If the dataframe is test_file, it will include user-written titles.
    """

    info = {"# of data before expanding": f"{len(df)}"}

    userID, clicknewsID, posnewsID, title = [], [] ,[], []
    for index, row in tqdm(df.iterrows(), desc="Expanding news"):

        if test_data:
            user_id, click_ids, pos_ids, user_titles = row
        else:
            user_id, click_ids, pos_ids, neg_ids = row

        userID += [user_id for _ in range(len(pos_ids))]
        clicknewsID += [click_ids for _ in range(len(pos_ids))]
        posnewsID += [[pos_id] for pos_id in pos_ids]
 
        if test_data:
            title += [user_title for user_title in user_titles.split(";;")]

    expanded_df = pd.DataFrame({
        "userID": userID,
        "clicknewsID": clicknewsID,
        "posnewsID": posnewsID,
    })
    if test_data:
        expanded_df["title"] = title

    info["# of data after expanding" ] = len(expanded_df)
    return expanded_df, info


# Type 2. A postive news will mapped to the collection of multiple users
def dataframe_collection(df, test_data=False):
    """ Collect the users' click history with the same positive news """
    clk_dict, user_dict = defaultdict(list), defaultdict(list)

    for i, row in df.iterrows():
        clk_dict[row["posnewsID"][0]] += row["clicknewsID"]
        user_dict[row["posnewsID"][0]] += [row["userID"]]

    num_news_per_posnews, num_user_per_posnews, num_click_per_posnews = [], [], []
    frq_dict = dict()
    posnewsID = []
    for posnews, v in tqdm(clk_dict.items(), desc="Collecting"):
        users = user_dict[posnews]
        clicks = clk_dict[posnews]

        sorted_clicks = dict(sorted(Counter(clicks).items(), key=lambda item: item[1]))
        clk_dict[posnews] = list(sorted_clicks.keys())
        frq_dict[posnews] = list(sorted_clicks.values())

        num_news_per_posnews.append(len(sorted_clicks))
        num_user_per_posnews.append(len(users))
        num_click_per_posnews.append(sum(sorted_clicks.values()))

    info = {
        "statistic_news_per_posnews": _statistic(num_news_per_posnews),
        "statistic_user_per_posnews": _statistic(num_user_per_posnews),
        "statistic_click_per_posnews": _statistic(num_click_per_posnews)
    }

    new_df = pd.DataFrame({
        "userID": user_dict.values(), # [[user_id1, user_id2, ...], ...]
        "clicknewsID": clk_dict.values(), # [[clicked_id1, clicked_id2, ...], ...]
        "clicknewsFreq": frq_dict.values(), # [[freq_id1, freq_id2, ...], ...]
        "posnewsID": [[i] for i in clk_dict.keys()], # [[posnews_id1], ...]
    })

    print(info)
    return new_df, info

def _statistic(data_list):
    statistic = {
        "mean": np.mean(data_list),
        "std": np.std(data_list),
        "max": np.max(data_list),
        "min": np.min(data_list),
    }
    return statistic





if __name__=="__main__":
    main()
