import os
import pandas as pd
import pdb

def main():
    """ To check the overlap between the clicked history and posnews
    """

    raw_test_data_dir = "../datasets/pens/test.pkl"
    processed_test_data_dir = "../datasets/specialize_own/test.pkl"

    raw_test = pd.read_pickle(raw_test_data_dir)
    processed_test = pd.read_pickle(processed_test_data_dir)

    raw_clicked = raw_test['clicknewsID']
    raw_posnews = raw_test['posnewsID']

    
    index = 0
    all_difference = []
    for click, pos in zip(raw_clicked, raw_posnews):
        click_num = len(set(click))
        pos_num = len(set(pos))

        tot_num = len(set(list(set(click)) + list(set(pos))))
        difference = (click_num + pos_num) - tot_num
        if difference != 0:
            if index == 15:
                pdb.set_trace()
            print(index, difference)
        index += 1
        all_difference.append(difference)

    clicked = processed_test["clicknewsID"]
    posnews = processed_test["posnewsID"]
    index = 0
    cnt_difference = 0
    for click, pos in zip(clicked, posnews):
        if pos[0] in click:
            print(index, processed_test["userID"].iloc[index], pos)
            cnt_difference += 1
        index += 1

    print(sum(all_difference), cnt_difference)


if __name__=="__main__":
    main()

"""NOTE:
247 NT2 [65343]                                                                                                                                                                                                                                                 2505 NT13 [33328]
2568 NT13 [101697]
3107 NT16 [51670]
3164 NT16 [51670]
7042 NT36 [107342]
7664 NT39 [50879]
7674 NT39 [99693]
9974 NT50 [48853]
11736 NT59 [13442]
12636 NT64 [55462]
13992 NT70 [45197]
14099 NT71 [32888]
14588 NT73 [58948] 
14632 NT74 [68235]
16374 NT82 [23110]
16391 NT82 [106224]
17732 NT89 [77773]
17974 NT90 [64063]
18497 NT93 [47830]
18644 NT94 [112740]
19618 NT99 [4984]
20015 NT101 [83272]
20500 NT103 [41005]
"""
