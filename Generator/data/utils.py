import os
import pdb
import nltk
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_metric

import sklearn
from sklearn.cluster import MiniBatchKMeans, KMeans

from simcse import SimCSE
from sklearn.metrics import pairwise_distances_argmin_min

metric = load_metric("rouge")
model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased") # NOTE: for user data sampling

    
def get_split_index(df, split, sample_type, seed=0):
    """
    Arguments
        - split: string, "{train}_{eval}_{test}" split porition, e.g., "80_20_100" 
        - sample_type: string, how to choose the train split, default=random
        - developing: bool, if True, take the test split for eval 
    Return
        - train_index, eval_index, test_index for df
    """

    num_train, num_eval, num_test = split.split()
    num_train, num_eval, num_test = int(num_train), int(num_eval), int(num_test)
   

    if sample_type == "inter_user":
        num_train_user, num_eval_user, num_test_user = num_train//200, num_eval//200, num_test//200
        user_index = np.arange(1, 104)
        np.random.seed(seed)
        np.random.shuffle(user_index)
        train_user_index = user_index[:num_train_user]
        eval_user_index = user_index[num_train_user:num_train_user+num_eval_user]
        test_user_index = user_index[num_train_user+num_eval_user:]

        train_index = [df.loc[df["userID"]==f"NT{i}"].index.to_numpy() for i in train_user_index]
        eval_index = [df.loc[df["userID"]==f"NT{i}"].index.to_numpy() for i in eval_user_index]
        test_index = [df.loc[df["userID"]==f"NT{i}"].index.to_numpy() for i in test_user_index]

        train_index = np.concatenate(train_index)
        eval_index = np.concatenate(eval_index)
        test_index = np.concatenate(test_index)

        assert (num_train_user + num_eval_user + num_test_user == 103)
        return np.sort(train_index), np.sort(eval_index), np.sort(test_index)

    index = df.index.to_numpy()
    np.random.seed(seed)
    np.random.shuffle(index)

    test_index = index[-num_test:]
    other_index = index[:-num_test]

    if sample_type == "random":
        print("Userwise finetuing, sample train data by random")
        train_index = index[:num_train]
        eval_index = index[num_train:num_train+num_eval]

    elif sample_type == "diversity":
        print("Userwise finetuing, sample train data by diversity")
        test_index = index[-num_test:]
        other_index = index[:-num_test]

        embeddings = model.encode(df.loc[other_index]['title'].tolist())
        kmeans = KMeans(init='k-means++', n_clusters=num_train, random_state=0).fit(embeddings)
        centers = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)[0]

        train_index = other_index[centers]
        eval_index = set(index) - set(train_index) - set(test_index)
        eval_index = [*eval_index,]
    elif sample_type == "informativeness":
        print("Userwise finetunining, sample train data by informativeness")

        title_embeddings  = model.encode(df['title'].loc[other_index].tolist())
        posnews_embeddings = model.encode(df['posnews'].loc[other_index].tolist())

        scores = torch.sum(title_embeddings*posnews_embeddings, dim=1)
        highest = torch.topk(scores, num_train)[1]
        train_index = other_index[highest]
        eval_index = set(index) - set(train_index) - set(test_index)
        eval_index = [*eval_index,]

    elif sample_type == "informativeness_low":
        print("Userwise finetunining, sample train data by informativeness low")

        title_embeddings  = model.encode(df['title'].loc[other_index].tolist())
        posnews_embeddings = model.encode(df['posnews'].loc[other_index].tolist())

        scores = torch.sum(title_embeddings*posnews_embeddings, dim=1)
        lowest = torch.topk(scores, num_train, largest=False)[1]
        train_index = other_index[lowest]
        eval_index = set(index) - set(train_index) - set(test_index)
        eval_index = [*eval_index,]

    else:
        ValueError(f"Invalid sample_type: {sample_type}")

    return train_index, np.sort(eval_index), np.sort(test_index)


def get_inputs(news_inputs, index, columns, num_index=1):

    outputs = {}
    bs = len(index)
    index = np.array(index).reshape(-1)
    for column in columns:
        output = np.take(news_inputs[column], index, axis=0)
        if num_index==1:
            outputs[column] = np.array(output).reshape(bs, -1)
        else:
            outputs[column] = np.array(output).reshape(bs, num_index, -1)
    return outputs


def get_inputs_remove(news_inputs, index, columns, num_index=1):

    outputs = {}
    bs = len(index)
    index = torch.tensor(index).reshape(-1)
    for column in columns:
        output = torch.index_select(news_inputs[column], 0, index)
        if num_index==1:
            outputs[column] = np.array(output).reshape(bs, -1)
        else:
            outputs[column] = np.array(output).reshape(bs, num_index, -1)

    return outputs


def add_padding(inputs, return_tensors=False, return_array=False):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    length = len(input_ids[0])
    if return_tensors:
        padding = torch.zeros((1,length), dtype=torch.int)
        input_ids = torch.cat((input_ids, padding), 0)
        attention_mask = torch.cat((attention_mask, padding), 0)
    elif return_array:
        padding = np.zeros((1,length), dtype=np.int)
        input_ids = np.concatenate((np.array(input_ids), padding), 0)
        attention_mask = np.concatenate((np.array(attention_mask), padding), 0)
    else:
        padding = [0] * length
        input_ids.append(padding)
        attention_mask.append(padding)

    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask
    return inputs


def get_user_features(user_dict, user_names):
    """
    user the user_name to retrieve the user_feature from user_dict
    """
    if type(user_names[0])==list:
        user_names = [int(i[0].replace("U", "").replace("NT", "")) for i in user_names]
        user_features = [user_dict[name] for name in user_names]

    else:
        user_names = [int(i.replace("U", "").replace("NT", "")) for i in user_names]
        user_features = [user_dict[name] for name in user_names]
    return user_features



def get_tail_features(editor_news_feat, click_ids, ctr, quant):

    user_features = []
    for click_id in tqdm(click_ids, desc="get history from tail"):
        click_ctr = ctr.iloc[click_id]
        tail_id = click_ctr.loc[click_ctr<=click_ctr.quantile(quant)]
        clk_feats = np.take(editor_news_feat, tail_id, axis=0)
        user_features.append(np.average(clk_feats, axis=0))

    return np.stack(user_features, axis=0)


def get_news_features(editor_news_feat, first_news_feat, click_ids, ids, feat_type, n_clusters=5):

    if feat_type == "average_news":
        user_features = []
        for click_id in tqdm(click_ids, desc="Get history by averaging"):
            clk_feats = np.take(editor_news_feat, click_id, axis=0)
            user_features.append(np.average(clk_feats, axis=0))
        return np.stack(user_features, axis=0)

    else:
        ValueError("Please specify userize_ufeat_type to the valid values.")


def get_text_features(editor_news_feat, first_news_feat, click_ids, ids, title_inputs):
    """ Take the raw title of the cloest news as user feature, further encode in get_user_embeds() """

    ids = np.array(ids).squeeze()
    # the query news should the output of the 1st mdoel but not the editor titles to avoid cheating
    all_feats = np.take(first_news_feat, ids, axis=0)
    user_features = []
    all_closest_ids = []
    for i, click_id in enumerate(click_ids):
        # calcuate the distance between the input news and clicked history
        clk_feats = np.take(editor_news_feat, click_id, axis=0) 
        closest_idx = np.argmin(np.linalg.norm(clk_feats-all_feats[i], axis=1))
        all_closest_ids.append(click_id[closest_idx])
    return np.take(title_inputs["input_ids"], all_closest_ids, axis=0)


def load_user_features(file_path):
    """
    file_path: a npz file contain user_names and user_features
    """
    if os.path.exists(file_path):
        data = np.load(file_path)
        data_dict = dict(zip(data["user_names"], data["user_features"]))
        return data_dict
    else:
        return None


def _analyze_dataset(df):

    reference = df["title"].tolist()
    prediction = df["posnews"].tolist()

    gen = _postprocess_text(prediction)
    ref = _postprocess_text(reference)

    result = metric.compute(predictions=gen, references=ref, use_stemmer=True)
    result = {key: round(value.mid.fmeasure * 100, 2) for key, value in result.items()}
    result["combined_score"] = round(np.mean(list(result.values())).item(), 4)
    result["size"] = len(df)
    return result


def _postprocess_text(text):

    text = [t.strip().lower() for t in text]
    # rougeLSum expects newline after each sentence
    text = ["\n".join(nltk.sent_tokenize(t)) for t in text]
    return text

