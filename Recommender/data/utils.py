import pdb
import pandas as pd
import torch
import numpy as np

def get_inputs(news_inputs, index):

    input_ids = np.take(news_inputs["input_ids"], index, axis=0)
    attention_mask = np.take(news_inputs["attention_mask"], index, axis=0)

    return input_ids, attention_mask

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

def add_padding_tensor(inputs, return_tensors=True):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    length = len(input_ids[0])
    if return_tensors:
        padding = torch.zeros((1,length), dtype=torch.int)
        input_ids = torch.cat((input_ids, padding), 0)
        attention_mask = torch.cat((attention_mask, padding), 0)
    else:
        padding = [0] * length
        input_ids.append(padding)
        attention_mask.append(padding)

    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask
    return inputs

def filter_data(df, userID, pred_title, test_index):
    df = df[df["userID"]==userID]
    df = df.reset_index(drop=True)
    df = df[test_index]
    df = df.reset_index(drop=True)
    df["pred_title"] = pred_title 

    return df
