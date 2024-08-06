# !/usr/bin/env python
# -*-coding:utf-8 -*-

import pandas as pd
import numpy as np
import scipy
from collections import defaultdict
import myutil

class DataProcess():
    def __init__(self, dataset=None, trainfile=None, validfile=None, testfile=None, socialfile=None):
        self.dataset = dataset
        self.trainfile = trainfile
        self.validfile = validfile
        self.testfile = testfile
        self.socialfile = socialfile

    def interactionProcess(self):
        self.train_set = pd.read_csv(self.trainfile)
        self.valid_set = pd.read_csv(self.validfile)
        self.test_set = pd.read_csv(self.testfile)

        self.n_user = pd.concat([self.train_set, self.valid_set, self.test_set])['userid'].unique().max() + 1
        self.n_item = pd.concat([self.train_set, self.valid_set, self.test_set])['itemid'].unique().max() + 1
        self.trainDataSize = len(self.train_set)
        self.validDataSize = len(self.valid_set)
        self.testDataSize = len(self.test_set)

        trainUser = np.array(self.train_set['userid'])
        trainItem = np.array(self.train_set['itemid'])
        self.UserItemNet = scipy.sparse.csr_matrix((np.ones(len(self.train_set)), (trainUser, trainItem)), shape=(self.n_user, self.n_item))

        self.usertrainPos = []
        for user in range(self.n_user):
            self.usertrainPos.append(self.UserItemNet[user].nonzero()[1])

    def getValidAndTest(self, testmode=1):
        self.uservalidDic = defaultdict(list)
        self.usertestDic = defaultdict(list)
        if testmode == 1:
            for i in range(len(self.valid_set)):
                user = self.valid_set['userid'][i]
                item = self.valid_set['itemid'][i]
                self.uservalidDic[user].append(item)
            for i in range(len(self.test_set)):
                user = self.test_set['userid'][i]
                item = self.test_set['itemid'][i]
                self.usertestDic[user].append(item)
        else:
            train_item = self.train_set['itemid'].unique()
            valid_item = self.valid_set['itemid'].unique()
            test_item = self.test_set['itemid'].unique()
            valid_item_nintrain = np.setdiff1d(valid_item, train_item)
            test_item_nintrain = np.setdiff1d(test_item, train_item)
            for i in range(len(self.valid_set)):
                user = self.valid_set['userid'][i]
                item = self.valid_set['itemid'][i]
                if item in valid_item_nintrain:
                    continue
                self.uservalidDic[user].append(item)
            for i in range(len(self.test_set)):
                user = self.test_set['userid'][i]
                item = self.test_set['itemid'][i]
                if item in test_item_nintrain:
                    continue
                self.usertestDic[user].append(item)
        return self.uservalidDic, self.usertestDic

    def create_adj_mat_AGE(self, adj_mat):
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = scipy.sparse.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        norm_i = scipy.sparse.eye(adj_mat.shape[0])
        norm_adj = (2.0 / 6.0) * norm_i + (4.0 / 6.0) * norm_adj
        return norm_adj

    def getInteractionFilter(self, filtermode=1):
        adj_mat = scipy.sparse.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.UserItemNet.tolil()
        adj_mat[:self.n_user, self.n_user:] = R
        adj_mat[self.n_user:, :self.n_user] = R.T
        adj_mat = adj_mat.todok()
        if filtermode == 1:
            norm_adj = self.create_adj_mat_AGE(adj_mat=adj_mat)
        elif filtermode == 2:
            norm_adj = self.create_adj_mat_AGE(adj_mat=(adj_mat + scipy.sparse.eye(adj_mat.shape[0])))
        else:
            norm_adj = None
        return norm_adj

    def getSocialFilter(self, filtermode=1):
        if self.socialfile == None:
            return None
        friendNet = pd.read_csv(self.socialfile)
        socialNet = scipy.sparse.csr_matrix((np.ones(len(friendNet)), (friendNet['userid'], friendNet['friendid'])), shape=(self.n_user, self.n_user))
        adj_mat = socialNet.tolil()
        if filtermode == 1:
            norm_adj = self.create_adj_mat_AGE(adj_mat=adj_mat)
        elif filtermode == 2:
            norm_adj = self.create_adj_mat_AGE(adj_mat=(adj_mat + scipy.sparse.eye(adj_mat.shape[0])))
        else:
            norm_adj = None
        return norm_adj

    def getItemInformation(self):
        ItemUserNet = self.UserItemNet.T
        itemneighbor = defaultdict(list)
        itemdegree = defaultdict(int)
        for item in range(self.n_item):
            temp = ItemUserNet[item].nonzero()[1]
            itemneighbor[item].append(temp)
            itemdegree[item] = len(temp)
        return itemneighbor, itemdegree

    def getUserInformation(self):
        UserItemNet = self.UserItemNet
        userneighbor = defaultdict(list)
        userdegree = defaultdict(int)
        for user in range(self.n_user):
            temp = UserItemNet[user].nonzero()[1]
            userneighbor[user].append(temp)
            userdegree[user] = len(temp)
        return userneighbor, userdegree

    def getUserLongTailTest(self, colddegree):
        HeadTestDic = self.usertestDic.copy()
        for i in list(HeadTestDic.keys()):
            try:
                if len(self.UserItemNet[i].nonzero()[1]) < colddegree:
                    del HeadTestDic[i]
            except:
                pass
        TailTestDic = self.usertestDic.copy()
        for i in list(TailTestDic.keys()):
            try:
                if len(self.UserItemNet[i].nonzero()[1]) >= colddegree:
                    del TailTestDic[i]
            except:
                pass
        return HeadTestDic, TailTestDic

    def coldUserProcess(self, degree_to_user_list, coldDegree=5):
        head_user = []
        tail_user = []
        for key, value in degree_to_user_list.items():
            if key < coldDegree:
                tail_user.extend(list(value))
            else:
                head_user.extend(list(value))
        self.head_user = head_user
        self.tail_user = tail_user
        return head_user, tail_user

def degreestatistic(obj_inter_dic):
    degree_to_obj_dic = defaultdict(int)
    for key,value in obj_inter_dic.items():
        degree_to_obj_dic[value] = degree_to_obj_dic[value]+1
    degree_to_obj_definedic = defaultdict(int)
    for key, value in degree_to_obj_dic.items():
        if key<=5:
            degree_to_obj_definedic[key] = value
        elif key>=6 and key<=10:
            degree_to_obj_definedic[6] = degree_to_obj_definedic[6]+value
        elif key>=11 and key<=15:
            degree_to_obj_definedic[11] = degree_to_obj_definedic[11] + value
        elif key>=16 and key<=20:
            degree_to_obj_definedic[16] = degree_to_obj_definedic[16] + value
        elif key>=21 and key<=50:
            degree_to_obj_definedic[21] = degree_to_obj_definedic[21] + value
        elif key>=51 and key<=100:
            degree_to_obj_definedic[51] = degree_to_obj_definedic[51] + value
        else:
            degree_to_obj_definedic[101] = degree_to_obj_definedic[101] + value
    degree_to_obj_define = sorted(degree_to_obj_definedic.items(), key=lambda item: item[0])
    return degree_to_obj_define


