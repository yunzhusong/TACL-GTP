import pandas as pd
import os
import pdb
import numpy as np

ghg = "../datasets/GHG"
spe = "../datasets/specialize_own"


def main():
    news = pd.read_pickle("../datasets/pens/news.pkl")

    train_data = pd.read_pickle(os.path.join(spe, "train.pkl"))
    eval_data = pd.read_pickle(os.path.join(spe, "validation.pkl"))
    test_data = pd.read_pickle(os.path.join(spe, "test.pkl"))

    posnews_train = set(np.concatenate(train_data["posnewsID"].tolist()).tolist())
    print("# of train_data for all posnews in specialize_own: ", len(train_data))
    print("# of unique posnews: ", len(posnews_train))

    posnews_eval = set(np.concatenate(eval_data["posnewsID"].tolist()).tolist())
    print("# of evl_data for all posnews in specialize_own: ", len(eval_data))
    print("# of unique posnews: ", len(posnews_eval))

    posnews_test = set(np.concatenate(test_data["posnewsID"].tolist()).tolist())
    print("# of test_data for all posnews in specialize_own: ", len(test_data))
    print("# of unique posnews: ", len(posnews_test))


    interset_train_eval = posnews_train.intersection(posnews_eval)
    print("# of intersection between train and valid: ", len(interset_train_eval))

    interset_train_test = posnews_train.intersection(posnews_test)
    print("# of intersection between train and test ", len(interset_train_test))

    interset_eval_test = posnews_eval.intersection(posnews_test)
    print("# of intersection between valid and test: ", len(interset_eval_test))

    interset_train_eval_test = interset_train_eval.intersection(posnews_test)
    print("# of intersection between train, valid and test: ", len(interset_train_eval_test))

    union_train_eval = posnews_train.union(posnews_eval)
    union_train_eval_test = union_train_eval.union(posnews_test)
    print("# of union of all posnews: ", len(union_train_eval_test))


    all_news = set(np.arange(len(news)))

    ## pens_a (headline generation)
    news_a = all_news - union_train_eval_test # news_without history
    data_a = pd.DataFrame({"posnews": list(news_a)})

    ## pens_b (headline generation with click history) 

    ## pens_c (headline generation with click history and personal headline)


    """
    union_train_eval = posnews_train.union(posnews_eval)
    print("# of overlap between train and valid: ", len(posnews_train)+len(posnews_eval)-len(union_train_eval))

    union_train_test = posnews_train.union(posnews_test)
    print("# of overlap between train and test: ", len(posnews_train)+len(posnews_test)-len(union_train_test))

    union_eval_test = posnews_eval.union(posnees_test)
    print("# of overlap between valid and test: ", len(posnews_eval)+len(posnews_test)-len(union_eval_test))

    union_train_eval_test = union_train_eval.union(posnews_test)
    print("# of overlap between train + valid and test: ", len(union_train_eval)+len(posnews_test)-len(union_train_eval_test))
    """


    # -----
    train_ghg = pd.read_pickle(os.path.join(ghg, "train.pkl"))
    eval_ghg = pd.read_pickle(os.path.join(ghg, "validation.pkl"))
    test_ghg = pd.read_pickle(os.path.join(ghg, "test.pkl"))

    pdb.set_trace()


main()
