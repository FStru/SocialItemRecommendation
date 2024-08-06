#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import pandas as pd
import gc
import math
import random

class PrepDouban():
    def dataPreprocess(self):
        prime = []
        f = open(f'./dataset/raw_data/douban/ratings.txt', encoding='utf-8')
        for line in f:
            s = line.strip().split(' ')
            user, item, rating = int(s[0]), int(s[1]), int(s[2])
            if rating > 3:
                prime.append([user, item])
        f.close()
        df = pd.DataFrame(prime, columns=['userid', 'itemid'])
        del prime, user, item, rating
        gc.collect()
        userId = pd.Categorical(df['userid'])
        itemId = pd.Categorical(df['itemid'])
        df['userid'] = userId.codes
        df['itemid'] = itemId.codes
        userCodeDict = {int(value): code for code, value in enumerate(userId.categories.values)}
        itemCodeDict = {int(value): code for code, value in enumerate(itemId.categories.values)}
        allinteraction = df[['userid', 'itemid']]
        allinteraction = allinteraction.drop_duplicates()

        prime = []
        f = open(f'./dataset/raw_data/douban/trusts.txt', encoding='utf-8')
        for line in f:
            s = line.strip().split(' ')
            user, friend= int(s[0]), int(s[1])
            prime.append([user, friend])
        f.close()
        friendNet = pd.DataFrame(prime, columns=['userid', 'friendid'])
        del prime, user, friend
        gc.collect()
        friendNet = friendNet[friendNet['userid'] != friendNet['friendid']].reset_index(drop=True)
        friendNet = friendNet.drop_duplicates().reset_index(drop=True)
        friendNet['userid'] = friendNet.apply(lambda x: reIndex(x.userid, userCodeDict), axis=1)
        friendNet['friendid'] = friendNet.apply(lambda x: reIndex(x.friendid, userCodeDict), axis=1)
        friendNet = friendNet.drop(friendNet[(friendNet['userid'] == -1) | (friendNet['friendid'] == -1)].index)
        friendNet = friendNet.reset_index(drop=True)

        data_dict = defaultdict(set)
        for index, row in allinteraction.iterrows():
            data_dict[row["userid"]].add(row["itemid"])
        train_dict = defaultdict(list)
        valid_dict = defaultdict(list)
        test_dict = defaultdict(list)
        for key, value in data_dict.items():
            if len(data_dict[key]) >= 3:
                test_temp, valid_temp, train_temp = split_list(list(data_dict[key]), [0.1, 0.1, 0.8])
                assert len(test_temp) != 0
                assert len(valid_temp) != 0
                assert len(train_temp) != 0
                train_dict[key] = train_temp
                valid_dict[key] = valid_temp
                test_dict[key] = test_temp
            else:
                train_dict[key] = list(data_dict[key])
        df_train = defaultdict(list)
        df_valid = defaultdict(list)
        df_test = defaultdict(list)
        for key, value in train_dict.items():
            for item in value:
                df_train['userid'].append(key)
                df_train['itemid'].append(item)
        for key, value in valid_dict.items():
            for item in value:
                df_valid['userid'].append(key)
                df_valid['itemid'].append(item)
        for key, value in test_dict.items():
            for item in value:
                df_test['userid'].append(key)
                df_test['itemid'].append(item)
        train_df = pd.DataFrame(df_train)
        valid_df = pd.DataFrame(df_valid)
        test_df = pd.DataFrame(df_test)

        # train_df.to_csv("./dataset/douban/train.csv")
        # valid_df.to_csv("./dataset/douban/valid.csv")
        # test_df.to_csv("./dataset/douban/test.csv")
        # friendNet.to_csv("./dataset/douban/social.csv")

def split_list(lst, ratios):
    random.shuffle(lst)
    total_ratio = sum(ratios)
    if total_ratio != 1:
        raise ValueError("The sum of ratios must be equal to 1.")
    n = len(lst)
    result = []
    start = 0
    for i in range(len(ratios)):
        end = start + math.ceil(n * ratios[i])
        result.append(lst[start:end])
        start = end
    return result

def reIndex(x, userReindex):
    if x in userReindex.keys():
        return userReindex[x]
    else:
        return -1

if __name__ == '__main__':
    douban = PrepDouban()
    douban.dataPreprocess()



