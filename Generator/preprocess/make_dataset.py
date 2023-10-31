"""
Expand the positive samples and create the pkl file.
This function is self-contained.
"""
import os
import pdb
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter

def build_samples_by_expansion(df):

    new_df = {}
    num_neg = 1
    userID, clicknewsID, posnewsID, negnewsID = [], [], [], []
    print("Before expansion: {}".format(len(df)))

    for i in tqdm(range(len(df)), desc="Building samples by expansion"):

        user_id, click_ids, pos_ids, neg_ids = df.iloc[i]
        #indices = np.arange(len(neg_ids)).tolist()
        
        for pos_id in pos_ids:
            #shuffle(indices)
            #neg_id = np.array(neg_ids)[indices[:num_neg]].tolist()

            userID.append(user_id)
            clicknewsID.append(click_ids)
            posnewsID.append([pos_id])
            #negnewsID.append(neg_id)

    new_df["userID"] = userID
    new_df["clicknewsID"] = clicknewsID
    new_df["posnewsID"] = posnewsID
    #new_df["negnewsID"] = negnewsID
    new_df = pd.DataFrame(new_df)

    print("After expansion: {}".format(len(new_df)))

    return new_df 
 

def build_test_samples_by_expansion(df):

    new_df = {}
    userID, clicknewsID, posnewsID, titles = [], [], [], []

    for i in range(len(df)):

        user_id, click_ids, pos_ids, raw_titles = df.iloc[i]
        raw_titles = raw_titles.split(";;")
        
        for pos_id, title in zip(pos_ids, raw_titles):

            userID.append(user_id)
            clicknewsID.append(click_ids)
            posnewsID.append([pos_id])
            titles.append(title)

    new_df["userID"] = userID
    new_df["clicknewsID"] = clicknewsID
    new_df["posnewsID"] = posnewsID
    new_df["title"] = titles
    new_df = pd.DataFrame(new_df)
    return new_df


def build_specialize_own():
    ext = "pkl"
    train_file = f"train.{ext}"
    valid_file = f"validation.{ext}"
    test_file = f"test.{ext}"

    raw_path = "../datasets/pens"
    out_path = "../datasets/specialize_own"

    test_df = pd.read_pickle(os.path.join(raw_path,test_file))
    valid_df = pd.read_pickle(os.path.join(raw_path,valid_file))
    train_df = pd.read_pickle(os.path.join(raw_path, train_file))

    test_df = build_test_samples_by_expansion(test_df)
    valid_df = build_samples_by_expansion(valid_df)
    train_df = build_samples_by_expansion(train_df)

    os.makedirs(out_path, exist_ok=True)

    train_df.to_pickle(os.path.join(out_path, train_file))
    valid_df.to_pickle(os.path.join(out_path, valid_file))
    test_df.to_pickle(os.path.join(out_path, test_file))


def build_s2(raw_path, out_path):
    num_validation = 2000
    ext = "pkl"
    train_file = f"train.{ext}"
    valid_file = f"validation.{ext}"

    valid_df = pd.read_pickle(os.path.join(raw_path,valid_file))
    train_df = pd.read_pickle(os.path.join(raw_path, train_file))

    if len(valid_df["posnewsID"][0]) > 1:
        valid_df = build_samples_by_expansion(valid_df)
        train_df = build_samples_by_expansion(train_df)

    def _build_s2_pretrain(raw_df):

        clk_dict = defaultdict(list)
        user_dict = defaultdict(list)
        
        for i, row in raw_df.iterrows():
            clk_dict[row["posnewsID"][0]] += row["clicknewsID"]
            user_dict[row["posnewsID"][0]].append(row["userID"])

        user_ids = []
        clicknews_ids = []
        posnews_ids = []

        num_click_list = []
        num_users_list = []
        for posnews, v in tqdm(clk_dict.items(), desc="building dataset"):
            # 
            users = user_dict[posnews]
            ## the news shared be shared between 5% users
            id_counter = Counter(v)
            clicknews_id = [_id for _id, freq in id_counter.items() if freq>len(users)*0.05]
            if len(clicknews_id) < len(users):
                clicknews_id = [_id for _id, freq in id_counter.items() if freq>len(users)*0.05*0.5]

            assert len(clicknews_id)>=1
            ## all clicked news for all users
            #clicknews_id = list(set(v))

            num_click_list.append(len(clicknews_id))
            num_users_list.append(len(users))

            posnews_ids.append([posnews])
            clicknews_ids.append(clicknews_id)
            user_ids.append(users)

        info = "\tAvg click per posnews: {}\n\tAvg user per posnews: {}\n".format(
                np.average(num_click_list), np.average(num_users_list))

        print(info)
        df = pd.DataFrame({
            "userID": user_ids,
            "clicknewsID": clicknews_ids,
            "posnewsID": posnews_ids
        })
        return df, info

    df = pd.concat([train_df, valid_df])
    df, df_info = _build_s2_pretrain(df)

    train_df = df.iloc[:-num_validation]
    valid_df = df.iloc[-num_validation:]

    os.makedirs(out_path, exist_ok=True)

    valid_df.to_pickle(os.path.join(out_path, "validation.pkl"))
    train_df.to_pickle(os.path.join(out_path, "train.pkl"))

    with open(os.path.join(out_path, "info.txt"), "w") as f:
        f.write("To train the stage 2 generation.\n")
        f.write("Combine the clicknews id of users with the same positive news.\n")
        f.write(df_info)


if __name__=="__main__":
    #build_specialize_own()

    # build dataset for 2nd stage generation
    raw_path = "../datasets/specialize_own"
    #out_path = "../datasets/pens_s2"
    out_path = "../datasets/pens_s2_shared" # interection

    build_s2(raw_path, out_path)
