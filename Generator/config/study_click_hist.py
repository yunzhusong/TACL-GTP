import os
import pdb
import pandas as pd
import argparse
from collections import Counter, defaultdict, OrderedDict
import matplotlib.pyplot as plt
import numpy as np


def _counting(df, counter=None, return_clk_dict=False):
    #// Get the data without duplicated

    print("Counting the CTR ...")
    if counter is None:
        counter = Counter()

    clk_dict = dict()
    pos_dict = defaultdict(list)
    for user_id, click_ids, pos_ids in zip(
        df["userID"].tolist(), df["clicknewsID"].tolist(), df["posnewsID"].tolist()):
        clk_dict[user_id] = click_ids
        for pos_id in pos_ids:
            pos_dict[user_id].append(pos_id)

    clk = [i for click_ids in clk_dict.values() for i in click_ids]
    pos = [i for pos_ids in pos_dict.values() for i in pos_ids]

    #// Count the CTR of all news by accumulating each user's clicked news and positive news
    counter.update(clk)
    counter.update(pos)

    if return_clk_dict:
        return counter, clk_dict
    return counter

def plot_click_ctr_distribution(sorted_counters, out_dir="anal/images/CTR"):
    #// Plot Click CTR distribution

    print("Plotting CTR's distribution ...")
    fig, ax = plt.subplots(len(sorted_counters), 1, constrained_layout=True, figsize=(6,3))

    ax.set_facecolor('#d8dcd6')
    ax.grid(color='w')
    ax.plot(np.arange(len(sorted_counters[0])), sorted_counters[0].values())
    ax.set_yscale('log')
    ax.set_title("CTR Distribution")
    plt.xlabel("News Rank")
    plt.ylabel("Click Through Rate")
    fig.savefig(f"{out_dir}/ctr_distri.png")

    '''
    for i in range(len(sorted_counters)):

        ax[i].set_facecolor('#d8dcd6')
        ax[i].grid(color='w')
        ax[i].plot(np.arange(len(sorted_counters[i])), sorted_counters[i].values())
        #ax[i].plot(np.arange(len(sorted_counters[i])), sorted_counters[i].values(), linewidth=2.0, fillstyle='bottom')
        #ax[i].scatter(np.arange(len(sorted_counters[i])), sorted_counters[i].values())
        ax[i].set_yscale('log')
        #ax[i].stem(np.arange(len(sorted_counters[i])), sorted_counters[i].values())

    ax[0].set_title("CTR Distribution")

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/ctr_distri.png")

    print("Plotting CTR's distribution ...")
    fig, ax = plt.subplots(len(sorted_counters), 1, constrained_layout=True)
    for i in range(len(sorted_counters)):

        ax[i].plot(np.log10(np.arange(len(sorted_counters[i]))), sorted_counters[i].values(), linewidth=2.0, fillstyle='bottom')
        #ax[i].stem(np.arange(len(sorted_counters[i])), sorted_counters[i].values())
        ax[i].grid()

    ax[0].set_title("CTR Distribution")

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/ctr_distri_loglog10.png")
    '''

def plot_click_ctr_distribution_leg(sorted_counters, out_dir="anal/images/CTR"):
    #// Plot Click CTR distribution

    print("Plotting CTR's distribution ...")
    fig, ax = plt.subplots(len(sorted_counters), 1, constrained_layout=True)
    for i in range(len(sorted_counters)):

        ax[i].set_facecolor('#d8dcd6')
        ax[i].grid(color='w')
        ax[i].plot(np.arange(len(sorted_counters[i])), sorted_counters[i].values())
        #ax[i].plot(np.arange(len(sorted_counters[i])), sorted_counters[i].values(), linewidth=2.0, fillstyle='bottom')
        #ax[i].scatter(np.arange(len(sorted_counters[i])), sorted_counters[i].values())
        ax[i].set_yscale('log')
        #ax[i].stem(np.arange(len(sorted_counters[i])), sorted_counters[i].values())

    ax[0].set_title("CTR Distribution")

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/ctr_distri.png")

    print("Plotting CTR's distribution ...")
    fig, ax = plt.subplots(len(sorted_counters), 1, constrained_layout=True)
    for i in range(len(sorted_counters)):

        ax[i].plot(np.log10(np.arange(len(sorted_counters[i]))), sorted_counters[i].values(), linewidth=2.0, fillstyle='bottom')
        #ax[i].stem(np.arange(len(sorted_counters[i])), sorted_counters[i].values())
        ax[i].grid()

    ax[0].set_title("CTR Distribution")

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/ctr_distri_loglog10.png")

def plot_ctr_histogram(counter, out_dir="anal/images/CTR"):
    #// Plot the distribution of all news

    print("Plotting CTR histrogram ...")
    fig, ax = plt.subplots()
    df = pd.DataFrame({"Total CTR": list(counter.values())})
    df.plot.kde(ax=ax, legend=False, title=f'CTR histogram of all news')
    df.plot.hist(density=False, ax=ax, legend=False)

    ax.grid(axis='y')
    ax.set_facecolor('#d8dcd6')

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/ctr_hist.png")

def plot_ctr_histogram_for_users(counter, clk_dict, out_dir="anal/images/CTR_users"):
    #// Plot the distribution for users

    print("Plotting CTR histrogram for users ...")
    os.makedirs(out_dir, exist_ok=True)
    users = list(clk_dict.keys())
    index = 0

    for k in range(10):

        fig, ax = plt.subplots(2,3, constrained_layout=True)
        for i in range(2):

            for j in range(3):

                user_id = users[index]
                ctr = [counter[i] for i in clk_dict[user_id]]

                df = pd.DataFrame({"CTR": ctr})
                ax[i][j].grid(axis='y')
                df.plot.kde(ax=ax[i][j], legend=False, title=f'{user_id}')
                df.plot.hist(density=False, ax=ax[i][j], legend=False)
                #ax[i][j].set_ylabel('Probability')
                ax[i][j].set_facecolor('#d8dcd6')

                index += 1

        fig.savefig(f"{out_dir}/ctr_hist_{k}.png")

    return

def anal_ctr(counter):
    #// Analyze the CTR

    print("Analyzing CTR ...")
    ctr_df = pd.DataFrame({"ctr": counter.values()})
    statis = ctr_df.agg(["min", "max", "mean", "std"]).round(decimals=2)
    statis_per = {}
    for per in [0.1,0.2,0.25,0.5,0.75,0.8,0.9,0.95,0.98]:
        statis_per[f"quantile_{per}"] = ctr_df.quantile(per)
    statis = statis.append(
        pd.DataFrame.from_dict(statis_per, orient='index', columns=["ctr"]))
    print(statis)
    return statis

def sort_dict(counter):
    print("Sorting the Counter by values ...")
    #// Sort by the counter' values
    sorted_keys = sorted(counter, key=counter.get, reverse=True)
    sorted_counter = {}
    for k in sorted_keys:
        sorted_counter[k] = counter[k]

    return sorted_counter

def get_ctr_by_newsID(newsID, counter):
    print("Returning the CTR for each news")
    #// Write the CTR to news_file
    #// If the ctr is zeros, means the news is not mentioned in train and eval files 
    ctr = [counter[int(i.replace('N', ''))-10000] for i in newsID]
    return ctr


if __name__=="__main__":

    parser = argparse.ArgumentParser()
 
    parser.add_argument("--data_dir", type=str, default="../datasets/pens/")
    parser.add_argument("--train_file", type=str, default="../datasets/pens/train.pkl")
    parser.add_argument("--eval_file",  type=str, default="../datasets/pens/validation.pkl")
    parser.add_argument("--test_file",  type=str, default="../datasets/pens/test.pkl")
    parser.add_argument("--news_file",  type=str, default="../datasets/pens/news.pkl")
    parser.add_argument("--plot_ctr_histogram", type=bool, default=False)
    parser.add_argument("--write_ctr_to_news_file", type=bool, default=False)

    args = parser.parse_args()

    # NOTE: the click history in datasets/pens/ is duplicated

    #// Read train and validation data to compute CTR
    train_df = pd.read_pickle(args.train_file)
    eval_df = pd.read_pickle(args.eval_file)
    test_df = pd.read_pickle(args.test_file)
    news_df = pd.read_pickle(args.news_file)

    all_news_ids = [int(i.replace("N",""))-10000 for i in news_df["newsID"]]
    counter = Counter(all_news_ids)
    counter, train_clk_dict = _counting(train_df, counter, return_clk_dict=True)
    counter = _counting(eval_df, counter)
    counter = _counting(test_df, counter)

    print(counter.most_common(10))

    #// Take log to ctr to better view the distribution
    print("Taking log to CTR ...")
    log_counter={}
    for k, v in counter.items():
        log_counter[k] = np.log10(v)

    #// Sort the CTR for plotting the CTR distribution
    sorted_counter = sort_dict(counter)
    #sorted_log_counter = sort_dict(log_counter)
    #plot_click_ctr_distribution([sorted_counter, sorted_log_counter], out_dir="anal/images/CTR")
    plot_click_ctr_distribution([sorted_counter,], out_dir="anal/images/CTR")

    ##// Analyze the CTR distribution
    #statis = anal_ctr(counter)
    #statis.to_json(os.path.join(args.data_dir, "anal/ctr/news_ctr.json"), indent=4)

    ##// Write the CTR to the news file
    #news = pd.read_pickle(args.news_file)
    #news["ctr"] = get_ctr_by_newsID(news["newsID"].tolist(), counter)
    #news.to_pickle(args.news_file)
    #print("Number of all news: ", len(news))

    ##// Plot histogram for all news and some users
    #plot_ctr_histogram(counter, out_dir="anal/images/CTR")
    #plot_ctr_histogram_for_users(counter, train_clk_dict, out_dir="anal/images/CTR_users")
    #plot_ctr_histogram(log_counter, out_dir="anal/images/CTR_log10")
    #plot_ctr_histogram_for_users(log_counter, train_clk_dict, out_dir="anal/images/CTR_log10_users")

    #// Plot histogram for test users
    test_clk_dict = dict(zip(test_df["userID"], test_df["clicknewsID"]))
    plot_ctr_histogram_for_users(log_counter, test_clk_dict, out_dir="anal/images/CTR_log10_test_users")


