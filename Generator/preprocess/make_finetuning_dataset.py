import pandas as pd
import os
import pdb
import numpy as np
import pprint
from make_dataset import build_test_samples_by_expansion

# For _analyze function
import nltk
from datasets import load_metric

metric = load_metric("rouge")


def expand_sample(samples):
    """ each user has 200 news"""
    expanded_sample_list = []
    for i in samples:
        expanded_sample_list.append(np.arange(i*200,(i+1)*200))
    expanded_samples = np.concatenate(expanded_sample_list)

    return expanded_samples


def build_intra_dataset():

    # raw test data (user-based)
    data = "../datasets/pens/test.pkl"
    data_df = pd.read_pickle(data)

    # test 1st stage generation results
    predict_txt_file = "../results/pens_ghg_own/bart/checkpoint-98000/phg/generated_predictions.txt"
    posnews_df = pd.read_csv(predict_txt_file, delimiter="\n", header=None)

    fold_num = 5 # folds
    shot_num = 5 # users

    print(f"Build dataset for intra {shot_num}-shot finetuning, there are {fold_num} folds")
    print(f"First stage predictions ({predict_txt_file}) are combined to output datasets.")

    for i in range(fold_num):

        # sample a few users for training and evaluating 
        all_samples = set(np.arange(len(data_df)))

        train_samples = set(np.arange(i*shot_num, (i+1)*shot_num))
        eval_samples = set(np.arange((i+1)*shot_num, (i+2)*shot_num))
 
        # the rest data is for testing
        test_samples = all_samples - train_samples
        test_samples = test_samples - eval_samples

        train_samples = list(train_samples)
        eval_samples = list(eval_samples)
        test_samples = list(test_samples)

        train_df = data_df.iloc[train_samples]
        eval_df = data_df.iloc[eval_samples]
        test_df = data_df.iloc[test_samples]

        print("Training users: ", train_samples)
        print("Eval users: ", eval_samples)
        print("Test users: ", test_samples)

        train_df = build_test_samples_by_expansion(train_df)
        eval_df = build_test_samples_by_expansion(eval_df)
        test_df = build_test_samples_by_expansion(test_df)
 
        # save the 1st stage generated results to dataset
        train_df["posnews"] = posnews_df.iloc[expand_sample(train_samples)][0].tolist()
        eval_df["posnews"] = posnews_df.iloc[expand_sample(eval_samples)][0].tolist()
        test_df["posnews"] = posnews_df.iloc[expand_sample(test_samples)][0].tolist()

        # output the results
        output_dir = f"../datasets/specialize_own/intra_{fold_num}users/{i}"
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_pickle(os.path.join(output_dir, "train.pkl"))
        eval_df.to_pickle(os.path.join(output_dir, "eval.pkl"))
        test_df.to_pickle(os.path.join(output_dir, "test.pkl"))

        # analyze dataset
        eval_info = _analyze_dataset(eval_df)
        test_info = _analyze_dataset(test_df)
        train_info = _analyze_dataset(train_df)

        info = {"test.pkl": test_info, "eval.pkl": eval_info, "train.pkl": train_info}
        with open(os.path.join(output_dir, "info.txt"), "w") as f:
            f.write("ROUGE scores between title and posnews (1st predictions)\n")
            pprint.pprint(info, f)


def build_inter_dataset():

    # raw test data (user-based)
    data = "../datasets/pens/test.pkl"
    data_df = pd.read_pickle(data)
    num_user = len(data_df)

    # test 1st stage generation results
    predict_txt_file = "../results/pens_ghg_own/bart/checkpoint-98000/phg/generated_predictions.txt"
    posnews_df = pd.read_csv(predict_txt_file, delimiter="\n", header=None)

    df = build_test_samples_by_expansion(data_df)
    df["posnews"] = posnews_df[0].tolist()

    fold_num = 1 # folds
    shot_num_train = 10 # users
    shot_num_eval = 10 # users

    print(f"Build dataset for inter {shot_num_train}-shot finetuning, there are {fold_num} folds")
    print(f"First stage predictions ({predict_txt_file}) are combined to output datasets.")

    for i in range(fold_num):

        all_index = np.arange(200)
        np.random.shuffle(all_index)
        train_index = all_index[:shot_num_train]
        eval_index = all_index[shot_num_train:shot_num_train+shot_num_eval]
        test_index = all_index[shot_num_train+shot_num_eval:]

        #train_index = np.arange(i*shot_num_train, (i+1)*shot_num_train)
        #eval_index = np.arange((i+1)*shot_num_train, (i+1)*shot_num_train+shot_num_eval)
        all_samples = set(np.arange(len(df)))

        train_samples = set(np.concatenate([train_index + 200 * j for j in range(num_user)]))
        eval_samples = set(np.concatenate([eval_index + 200 * j for j in range(num_user)]))
        test_samples = all_samples - train_samples
        test_samples = test_samples - eval_samples

        train_samples = list(train_samples)
        eval_samples = list(eval_samples)
        test_samples = list(test_samples)

        train_df = df.iloc[train_samples]
        eval_df = df.iloc[eval_samples]
        test_df = df.iloc[test_samples]

        # output the results
        output_dir = f"../datasets/specialize_own/inter_{shot_num_train}samples/{i}"
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_pickle(os.path.join(output_dir, "train.pkl"))
        eval_df.to_pickle(os.path.join(output_dir, "eval.pkl"))
        test_df.to_pickle(os.path.join(output_dir, "test.pkl"))

        # analyze dataset
        eval_info = _analyze_dataset(eval_df)
        test_info = _analyze_dataset(test_df)
        train_info = _analyze_dataset(train_df)

        info = {"test.pkl": test_info, "eval.pkl": eval_info, "train.pkl": train_info}

        with open(os.path.join(output_dir, "info.txt"), "w") as f:
            f.write("ROUGE scores between title and posnews (1st predictions)\n")
            pprint.pprint(info, f)


def _postprocess_text(text):

    text = [t.strip().lower() for t in text]
    # rougeLSum expects newline after each sentence
    text = ["\n".join(nltk.sent_tokenize(t)) for t in text]
    return text


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

if __name__=="__main__":
    #build_intra_dataset()
    build_inter_dataset()
