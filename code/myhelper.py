# !/usr/bin/env python
# -*-coding:utf-8 -*-

import torch
import scipy
import numpy as np
from collections import defaultdict
import pandas as pd


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


def degree_to_object(objectdegree, objectlist):
    degree_to_object_list = defaultdict(list)
    degree_to_object_num = defaultdict(int)
    for object in objectlist:
        degree_to_object_list[objectdegree[object]].append(object)
        degree_to_object_num[objectdegree[object]] = degree_to_object_num[objectdegree[object]] + 1
    degree_to_object_list = dict(sorted(degree_to_object_list.items(), key=lambda x: x[0]))
    degree_to_object_num = dict(sorted(degree_to_object_num.items(), key=lambda x: x[0]))
    return degree_to_object_list, degree_to_object_num


def coldNodeProcess(degree_to_user_list, coldDegree=5):
    head_user = []
    tail_user = []
    for key, value in degree_to_user_list.items():
        if key < coldDegree:
            tail_user.extend(list(value))
        else:
            head_user.extend(list(value))
    return head_user, tail_user


def coldNodeProcessNon01(degree_to_user_list, coldDegree=5):
    tail_user = []
    for key, value in degree_to_user_list.items():
        if key>1 and key < coldDegree:
            tail_user.extend(list(value))
    return tail_user


def i2i_pairs_process(i2i_pair, n_item):
    n_item = n_item
    trainDataSize = len(i2i_pair)
    item1 = np.array(i2i_pair['itemid'])
    item2 = np.array(i2i_pair['ritemid'])
    ItemItemNet = scipy.sparse.csr_matrix((np.ones(trainDataSize), (item1, item2)), shape=(n_item, n_item))
    itemtrainPos = []
    for item in range(n_item):
        itemtrainPos.append(ItemItemNet[item].nonzero()[1])
    return itemtrainPos, trainDataSize


def generate_preedge_bycos(preitemedgesfile, itemrep_matrix, item_num, itemdegree):
    def degree_to_object(objectdegree, objectlist):
        degree_to_object_list = defaultdict(list)
        degree_to_object_num = defaultdict(int)
        for object in objectlist:
            degree_to_object_list[objectdegree[object]].append(object)
            degree_to_object_num[objectdegree[object]] = degree_to_object_num[objectdegree[object]] + 1
        degree_to_object_list = dict(sorted(degree_to_object_list.items(), key=lambda x: x[0]))
        degree_to_object_num = dict(sorted(degree_to_object_num.items(), key=lambda x: x[0]))
        return degree_to_object_list, degree_to_object_num

    cosm = itemrep_matrix.dot(itemrep_matrix.T)
    rowsum = np.array(itemrep_matrix.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = scipy.sparse.diags(d_inv)
    assert d_mat.shape[0] == item_num
    simimat = d_mat.dot(cosm)
    simimat = simimat.dot(d_mat)
    i2i_pair = scipy.sparse.find(simimat)
    rowid, colid, value = i2i_pair
    i2i_pair = list(zip(rowid, colid, value))
    i2i_pair_upd = []
    for ele in i2i_pair:
        if ele[0] != ele[1] and ele[2] > 0.5 and ele[2] < 0.9:
            i2i_pair_upd.append([ele[0], ele[1], ele[2]])
    i2i_pair_dict = defaultdict(set)
    for ele in i2i_pair_upd:
        i2i_pair_dict[ele[0]].add(ele[1])
        i2i_pair_dict[ele[1]].add(ele[0])
    i2i_pair_upd = []
    for key, value in i2i_pair_dict.items():
        for ele in value:
            i2i_pair_upd.append([key, ele])
    df = pd.DataFrame(i2i_pair_upd, columns=['itemid', 'ritemid'])
    df.to_csv(preitemedgesfile)

    return df