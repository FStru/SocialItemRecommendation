#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

def generator_train_collate(batch):
    item1 = [item[0] for item in batch]
    item2 = [item[1] for item in batch]
    item1 = torch.LongTensor(item1)
    item2 = torch.LongTensor(item2)
    return [item1, item2]


def generate_trainsample(node_num, node2_num, node_adj, trainDataSize):
    users = np.random.randint(0, node_num, trainDataSize)
    S = []
    for i, user in enumerate(users):
        posForUser = node_adj[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, node2_num)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
    return np.array(S)


def generate_traindataloader(sampleSet, trainbatch):
    users = torch.Tensor(sampleSet[:, 0]).long()
    posItems = torch.Tensor(sampleSet[:, 1]).long()
    negItems = torch.Tensor(sampleSet[:, 2]).long()
    trainset = torch.utils.data.TensorDataset(users, posItems, negItems)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=trainbatch, shuffle=True)
    return train_loader


def generate_trainsample_multi(node_num, node2_num, node_adj, trainDataSize, negsample=1):
    users = np.random.randint(0, node_num, trainDataSize)
    S = []
    N = []
    for i, user in enumerate(users):
        posForUser = node_adj[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        S.append([user, positem])
        temp = []
        while len(temp)!=negsample:
            negitem = np.random.randint(0, node2_num)
            if negitem in posForUser:
                continue
            else:
                temp.append(negitem)
        N.append(temp)
    return np.array(S), np.array(N)

def generate_traindataloader_multi(sampleSet, negativeSet, trainbatch):
    users = torch.Tensor(sampleSet[:, 0]).long()
    posItems = torch.Tensor(sampleSet[:, 1]).long()
    negItems = torch.Tensor(negativeSet).long()

    trainset = torch.utils.data.TensorDataset(users, posItems, negItems)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=trainbatch, shuffle=True)
    return train_loader