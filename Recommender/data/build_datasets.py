""" Builds datasets. """
import os
import pdb
import copy
import logging
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from data.utils import *

logger = logging.getLogger(__name__)

dataset_own_file_mapping = {
    "specialize_own": "../datasets/specialize_own",
    "pens_ghg_own": "../datasets/GHG",
    "pens_rec_own": "../datasets/pens"
}

dataset_column_name_mapping = {
    "specialize_own": ("clicknewsID", "posnewsID", "negnewsID", "title", "userID", "posnews", "body"),
    "pens_ghg_own": ("", "newsID", "", "title", "", "", ""),
    "pens_rec_own": ("clicknewsID", "posnewsID", "negnewsID", "title", "userID", "", "body"),
}
news_file = "../datasets/pens/news.pkl"

def build_datasets(data_args, training_args, model_args, tokenizer):

    # Get the column names for input/target.
    dataset_columns = dataset_column_name_mapping.get(data_args.dataset_name, None)
    clk_column = dataset_columns[0]
    pos_column = dataset_columns[1]
    neg_column = dataset_columns[2]
    tit_column = dataset_columns[3]
    usr_column = dataset_columns[4]
    txt_column = dataset_columns[5] # input text (not used in recommender)
    bdy_column = dataset_columns[6]

    text_column = data_args.text_column if data_args.text_column is not None else tit_column

    max_source_length = data_args.max_source_length
    padding = "max_length" if data_args.pad_to_max_length else False

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    if data_args.dataset_name is not None and data_args.dataset_name[-3:]!='own':
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_dir = dataset_own_file_mapping.get(data_args.dataset_name, None)
        ext = "pkl"
        train_file = os.path.join(data_dir, "train.pkl")
        eval_file = os.path.join(data_dir, "validation.pkl")
        test_file = os.path.join(data_dir, "validation.pkl")

        if data_args.train_file is not None:
            train_file = data_args.train_file
            ext = data_args.train_file.split(".")[-1]

        if data_args.validation_file is not None:
            eval_file = data_args.validation_file
            ext = data_args.validation_file.split(".")[-1]

        if data_args.test_file is not None:
            test_file = data_args.test_file
            ext = data_args.test_file.split(".")[-1]


    if training_args.do_train:
        logger.info(f"Training file: {train_file}")
        raw_train_dataset = Dataset.from_pandas(pd.read_pickle(train_file))
        if training_args.shuffle_before_select and data_args.max_train_samples is not None:
            raw_train_dataset = raw_train_dataset.select(range(data_args.max_train_samples))
            #raw_train_dataset = raw_train_dataset.iloc[:training_args.max_train_samples]
        raw_train_dataset = raw_train_dataset.shuffle()
        train_column_names = raw_train_dataset.column_names

    if training_args.do_eval:
        logger.info(f"Validation file: {eval_file}")
        raw_eval_dataset = Dataset.from_pandas(pd.read_pickle(eval_file))
        eval_column_names = raw_eval_dataset.column_names

    if training_args.do_predict:
        logger.info(f"Test file: {test_file}")
        if ext == "pkl":
            test_df = pd.read_pickle(test_file)
        elif ext == "txt":
            test_df = pd.read_csv(data_args.test_file, sep="\n", header=None)
            test_df[text_column] = test_df[0].tolist()
            test_df = test_df.drop(columns=[0])
        # filter the data in pred_file
        if training_args.get_recommended_score:
            pred_title = pd.read_csv(training_args.pred_file+"/generated_predictions.txt", sep="\n", header=None)[0]
            if training_args.pred_file_is_full:
                test_df["pred_title"] = pred_title
            else:
                test_index = pd.read_csv(training_args.pred_file+"/split.csv")["split"]=="test"
                test_df = filter_data(test_df, training_args.user_name, pred_title, test_index)
        raw_test_dataset = Dataset.from_pandas(test_df)
        test_column_names = raw_test_dataset.column_names

    if not training_args.get_news_features:
        news = pd.read_pickle("../datasets/pens/news.pkl")
        news = news.fillna("")
        news_title_inputs = tokenizer(news["title"].tolist(), padding="max_length",
            max_length=data_args.max_source_length, truncation=True, return_tensors="pt")
        news_title_inputs = add_padding(news_title_inputs,return_array=True)
        news_ctr = news["ctr"].tolist()
        news_ctr.append(1)
        news_ctr = np.array(news_ctr)
        news_ctr = np.log10(news_ctr+1e-7)
        news_ctr = news_ctr/np.max(news_ctr)
        num_news = len(news)

    def preprocess_for_recommendation(examples):
        his_ids = examples[clk_column]
        pos_ids = examples[pos_column]
        if neg_column not in examples.keys():
            neg_ids = [[] for i in examples[pos_column]]
            max_num_data = 1
        else:
            neg_ids = examples[neg_column]
            if process_training:
                max_num_data = 2
            else:
                max_num_data = 50
        max_num_hist = 50

        #// Construct the news samples and dp the truncation and padding
        data_ids, user_ids, labels = [], [], []

        for _pos_ids, _neg_ids in zip(pos_ids, neg_ids):
            if process_training:
                data_ids.append(_pos_ids + _neg_ids)
                labels.append([1]*len(_pos_ids)+[0]*len(_neg_ids))
            else:
                data = _pos_ids + _neg_ids
                data = data + [num_news]*max(max_num_data - len(data), 0)
                data_ids.append(data[:max_num_data])
                label = [1]*len(_pos_ids) + [0]*len(_neg_ids) + [-100]*max(max_num_data-len(_pos_ids)-len(_neg_ids), 0)
                labels.append(label[:max_num_data])

        #// Construct the user's click history and do the truncation and padding
        for _his_ids in his_ids:
            user_ids.append(_his_ids[:max_num_hist] + [num_news]*max(max_num_hist-len(_his_ids), 0))

        #// Obtain news title for sample and user click history
        data_input_ids, data_attention_mask = get_inputs(news_title_inputs, data_ids)
        user_input_ids, user_attention_mask = get_inputs(news_title_inputs, user_ids)

        results = {
            "input_ids": data_input_ids, "attention_mask": data_attention_mask,
            "user_input_ids": user_input_ids, "user_attention_mask": user_attention_mask,
            "labels_rec": labels, "labels": data_input_ids
        }

        #// Obtain CTR
        if training_args.ctr_loss:
            results["labels_ctr"] = np.take(news_ctr, data_ids)

        return results

    def preprocess_to_get_news_features(examples):
        """
        input text is in input_file
        """
        text = examples[text_column]
        inputs = tokenizer(text, padding=padding, max_length=max_source_length,
            truncation=True)
        #inputs["labels"] = np.zeros(len(text))
        inputs["labels"] = inputs["input_ids"]
        #ids = [[i] for i in range(len(text))]
        #inputs["news_id"] = ids
        return inputs

    def preprocess_to_get_recommended_score_from_text(examples):
        # num_news
        clk_ids = examples[clk_column]
        max_num_clk = 50
        # build user click ids
        user_ids = []
        for _clk_ids in clk_ids:
            user_ids.append(_clk_ids[:max_num_clk] + [num_news]*max(max_num_clk - len(_clk_ids), 0))
        
        user_input_ids, user_attention_mask = get_inputs(news_title_inputs, user_ids)

        pred_title = examples["pred_title"]
        pred_inputs = tokenizer(pred_title, padding="max_length",
                                 max_length=data_args.max_source_length,
                                 truncation=True)

        return {
            "input_ids": pred_inputs["input_ids"], "attention_mask": pred_inputs["attention_mask"],
            "user_input_ids": user_input_ids, "user_attention_mask": user_attention_mask,
            "labels": pred_inputs["input_ids"],
        }

    if training_args.get_news_features:
        preprocess_function = preprocess_to_get_news_features
    elif training_args.get_recommended_score:
        if training_args.take_pred_text:
            # get the prediction target by text file
            preprocess_function = preprocess_to_get_recommended_score_from_text
        else:
            # get the prediction target by news index
            preprocess_function = preprocess_to_get_recommended_score_from_index
    else:
        preprocess_function = preprocess_for_recommendation
        
    process_training, process_testing = False, False

    train_dataset, eval_dataset, test_dataset = None, None, None

    if training_args.do_train:
        process_training = True
        if data_args.max_train_samples is not None:
            raw_train_dataset = raw_train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = raw_train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        process_training = False
        if data_args.max_eval_samples is not None:
            raw_eval_dataset = raw_eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = raw_eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        process_testing = True
        if data_args.max_predict_samples is not None:
            raw_test_dataset = raw_test_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            test_dataset = raw_test_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=test_column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
    return train_dataset, eval_dataset, test_dataset

