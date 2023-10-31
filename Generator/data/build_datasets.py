""" Builds datasets. """
import os
import pdb
import copy
import json
import pprint
import logging
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from data.utils import add_padding
from data.utils import load_user_features
from data.utils import get_inputs
from data.utils import get_user_features, get_news_features, get_tail_features, get_text_features
from data.utils import _analyze_dataset
from data.utils import get_split_index
from preprocess.retrieve import retrieve_from_predicted, retrieve

logger = logging.getLogger(__name__)

dataset_own_file_mapping = {
    "specialize_own": "../datasets/specialize_own",
    "pens_ghg_own": "../datasets/GHG",
    "s2_own": "../datasets/pens_s2",
    "s2_shared_own": "../datasets/pens_s2_shared",
    "userwise_own": "",
}

dataset_column_name_mapping = {
    "specialize_own": ("clicknewsID", "posnewsID", "negnewsID", "title", "userID", "posnews", "body"),
    "pens_ghg_own": ("", "posnewsID", "", "title", "", "", ""),
    "s2_own": ("clicknewsID", "posnewsID", "", "title", "userID", "posnews", "body"),
    "s2_shared_own": ("clicknewsID", "posnewsID", "", "title", "userID", "posnews", "body"),
    "userwise_own": ("clicknewsID", "posnewsID", "negnewsID", "title", "userID", "posnews", "body"),
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
    txt_column = dataset_columns[5]
    bdy_column = dataset_columns[6]

    text_column = data_args.text_column if data_args.text_column is not None else tit_column
    summ_column = data_args.summary_column if data_args.summary_column is not None else tit_column

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length-1
    max_source_length = data_args.max_source_length
    padding = "max_length" if data_args.pad_to_max_length else False

    news = pd.read_pickle(news_file)
    news = news.fillna("")

    # Prepare inputs (if text_file is None, take ground truth headline as input to training)
    if training_args.text_file is not None:
        # title'
        raw_text = pd.read_csv(training_args.text_file, sep="\n", header=None)[0].tolist()
    else:
        # title
        raw_text = news[text_column].tolist()

    if training_args.extra_info:
        logger.warning("Appending retrieved news body")
        if training_args.retrieve_online:
            retrieved_file = training_args.text_file[:-4]+"_retrieved.csv"
            if os.path.exists(retrieved_file):
                logger.warning(f"The predictions have been processed. Read from the local file: {retrieved_file}")
                bodies = pd.read_csv(retrieved_file)["retrieved"]
                bodies = bodies.fillna("").tolist()
            else:
                logger.warning(f"Online retrieving and will be saved to {retrieved_file}")
                # online -> retrieve based on raw text (extra' or extra)
                news["target"] = raw_text
                news = retrieve(news, "target", "body")
                bodies = news["retrieved"].tolist()
                news["retrieved"].to_csv(retrieved_file, index=False)
        else:
            # extra
            if training_args.do_finetune:
                logger.warning("[WARNING] When finetuning or testing, we should use 1st prediction to retrieve extra information! Please set retrieve_online=True to conduct fair experiments!")
            logger.warning(f"Take the retrieved results from {news_file}, which is retrieved based on ground-truth titles.")
            bodies = news[f"retrieved"].tolist()
        raw_text = [t+" "+b for t, b in zip(raw_text, bodies)]

    text_inputs = tokenizer(raw_text, padding=padding, max_length=max_source_length,
        truncation=True, return_tensors="pt", return_special_tokens_mask=True)
    text_inputs = add_padding(text_inputs, return_array=True)

    # Prepare Outputs
    #title_inputs = None
    titles = news[summ_column].tolist()
    if not training_args.do_finetune and not training_args.eval_with_test_data:
        title_inputs = tokenizer(titles, padding=padding, max_length=max_target_length,
            truncation=True, return_tensors="pt", return_special_tokens_mask=True)
        title_inputs = add_padding(title_inputs, return_array=True)

    user_dir = training_args.userize_ufeat_path if training_args.userize_ufeat_path else ""

    #// user-wise training
    if training_args.userwise:
        ui = training_args.userwise_index
        logger.warning(f"Userwise training on NT{ui}")
        posnews = pd.read_csv("../results_important/pens_ghg_own/bart/checkpoint-98000/phg/generated_predictions.txt",
                              sep="\n", header=None)[0].tolist()
        user_df = pd.read_pickle("../datasets_old/specialize_own/test.pkl")
        user_df["posnews"] = posnews
        
        if ui != -1:
            user_df = user_df.iloc[ui*200:(ui+1)*200]

        train_index, eval_index, test_index = get_split_index(user_df,
            training_args.userwise_split, training_args.userwise_sample_type,
            training_args.userwise_seed)
        

        train_df = user_df.loc[train_index]
        eval_df = user_df.loc[eval_index]
        test_df = user_df.loc[test_index]

        user_df["split"] = [""]*len(user_df)
        user_df["split"].loc[train_index] = "train"
        user_df["split"].loc[eval_index] = "eval"
        user_df["split"].loc[test_index] = "test"

        if training_args.developing:
            eval_df = test_df

        #if training_args.userwise_sample_type=="inter_user":
        #    eval_df = eval_df.iloc[:1000]

        logger.warning(eval_df.index)

        if not os.path.exists(training_args.output_dir+"/info.txt"):
            with open(training_args.output_dir+"/info.json", 'w') as fp:
                info = {
                    "train": _analyze_dataset(train_df),
                    "eval": _analyze_dataset(eval_df),
                    "test": _analyze_dataset(test_df),
                       }
                json.dump(info, fp, indent=4)
            user_df["split"].to_csv(training_args.output_dir+"/split.csv")

    #// Pretraining (editor titles) or Finetuning (user titles)
    else:
        import pdb; pdb.set_trace()

        data_dir = dataset_own_file_mapping.get(data_args.dataset_name, None)

        if data_args.train_file is not None:
            train_df = pd.read_pickle(data_args.train_file)
        else:
            train_df = pd.read_pickle(os.path.join(data_dir, "train.pkl"))

        if data_args.validation_file is not None:
            eval_df = pd.read_pickle(data_args.validation_file)
        else:
            eval_df = pd.read_pickle(os.path.join(data_dir, "validation.pkl"))

        if data_args.test_file is not None:
            test_df = pd.read_pickle(data_args.test_file)
        else:
            test_df = pd.read_pickle(os.path.join(data_dir, "test.pkl"))


    if training_args.do_finetune:
        train_ufeat_file = os.path.join(user_dir, "user_features_test.npz")
        eval_ufeat_file = os.path.join(user_dir, "user_features_test.npz")
        test_ufeat_file = os.path.join(user_dir, "user_features_test.npz")
    else:
        train_ufeat_file = os.path.join(user_dir, "user_features_train.npz")
        eval_ufeat_file = os.path.join(user_dir, "user_features_eval.npz")
        test_ufeat_file = os.path.join(user_dir, "user_features_test.npz")

    if training_args.eval_with_test_data:
        logger.warning("Evaluate on personal headlines")
        eval_df = test_df
        eval_ufeat_file = test_ufeat_file

    if training_args.do_train:
        raw_train_dataset = Dataset.from_pandas(train_df)
        if training_args.shuffle_before_select and data_args.max_train_samples is not None:
            #raw_train_dataset = raw_train_dataset.iloc[:training_args.max_train_samples]
            raw_train_dataset = raw_train_dataset.select(range(data_args.max_train_samples))
        raw_train_dataset = raw_train_dataset.shuffle()
        train_column_names = raw_train_dataset.column_names

    if training_args.do_eval:
        raw_eval_dataset = Dataset.from_pandas(eval_df)
        eval_column_names = raw_eval_dataset.column_names
        if pos_column not in eval_column_names:
            pos_column = eval_column_names[2]

    if training_args.do_predict:
        #if training_args.predict_txt_file is not None:
        #    logger.warning(f"Inputs are generated headlines from {training_args.predict_txt_file}")
        #    if text_column not in test_df.keys():
        #        text = pd.read_csv(training_args.predict_txt_file, delimiter="\n", header=None)
        #        test_df[text_column] = text[0].tolist()
        #    test_df = .retrieve_from_predicted(test_df, news, pos_column, text_column, bdy_column)
        raw_test_dataset = Dataset.from_pandas(test_df)
        test_column_names = raw_test_dataset.column_names
        if pos_column not in test_column_names:
            pos_column = test_column_names[2]

    #// Prepare the source for different types of user features
    if training_args.userize:
        logger.warning(f"Loading user features from {user_dir}")
        editor_news_feat = np.load(os.path.join(user_dir, "editor_headline/news_features.npz"))["news_features"]
        first_news_feat = np.load(os.path.join(user_dir, "first_stage/news_features.npz"))["news_features"]

        #if training_args.userize_ufeat_type=="tail_feat":
        #    logger.warning(f"Collecting the news with CTR higher than {training_args.userize_ctr_threshold}")
        #    head_ids = news[news["ctr"]>training_args.userize_ctr_threshold].index.tolist()
        if training_args.userize_ufeat_type=="text_closest":
            title_inputs_short = tokenizer(titles, padding=padding, max_length=training_args.userize_user_token_length, truncation=True)

        #news_feat = np.load(news_ufeat_file)["news_features"]
        #train_ufeat_dict = None
        #eval_ufeat_dict = None
        #test_ufeat_dict = None
        train_ufeat_dict = load_user_features(train_ufeat_file) if training_args.do_train else None
        eval_ufeat_dict = load_user_features(eval_ufeat_file) if training_args.do_eval else None
        test_ufeat_dict = load_user_features(test_ufeat_file) if training_args.do_predict else None
        # For stage-2 pretraining, we redivide the train and validation set.
        # Therefore, the user mapping would not be same as that in recommender
        if training_args.do_train and training_args.do_eval:
            if train_ufeat_dict is not None and eval_ufeat_dict is not None:
                train_ufeat_dict = {**train_ufeat_dict, **eval_ufeat_dict}
                eval_ufeat_dict = train_ufeat_dict

    #// Arguments setup
    userize = training_args.userize

    def preprocess_default(examples):
        """
        Default setup for prepaing text and label
        """
        
        ids = examples[pos_column]
        inputs = get_inputs(text_inputs, ids, columns=["input_ids", "attention_mask"], num_index=1)

        if training_args.eval_with_test_data or training_args.do_finetune:
            labels = examples[tit_column]
            labels = pd.DataFrame({tit_column: labels}).fillna(" ")[tit_column].tolist()
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(labels, padding=padding, max_length=max_target_length,
                                   truncation=True)["input_ids"]
        else:
            labels = get_inputs(title_inputs, ids, columns=["input_ids"], num_index=1
                               )["input_ids"]

        labels = np.array(labels)
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels[labels==tokenizer.pad_token_id] = -100

        # Additional setup, should be formalized later
        if userize:
            if training_args.userize_ufeat_type=="text_closest":
                click = examples[clk_column]
                inputs["user_features"] = get_text_features(editor_news_feat, first_news_feat, click, ids, title_inputs_short)
            elif training_args.userize_ufeat_type=="user_feat":
                user_names = examples[usr_column]
                inputs["user_features"] = get_user_features(user_dict, user_names)
            elif training_args.userize_ufeat_type=="tail_feat":
                user_names = examples[usr_column]
                click = examples[clk_column]
                inputs["user_features"] = get_tail_features(editor_news_feat, click, news["ctr"], training_args.userize_ctr_quant)
            else:
                click = examples[clk_column]
                inputs["user_features"] = get_news_features(
                    editor_news_feat, first_news_feat, click, ids,
                    feat_type=training_args.userize_ufeat_type)

        inputs["labels"] = labels
        return inputs

    preprocess_function = preprocess_default
    train_dataset, eval_dataset, test_dataset = None, None, None

    if training_args.do_train:
        user_dict = train_ufeat_dict if userize else None
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
        user_dict = eval_ufeat_dict if userize else None
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
        user_dict = test_ufeat_dict if userize else None
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

