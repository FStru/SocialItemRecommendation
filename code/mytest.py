# !/usr/bin/env python
# -*-coding:utf-8 -*-


import numpy as np
import torch

def test_model(args, test_loader, Recmodel, testDict, UserItemNet, device):
    topks = eval(args.topks)
    max_K = max(topks)
    rating_list, groundtrue_list = getRatingAndGroundtureList(args, test_loader, Recmodel, testDict, UserItemNet, device, max_K)
    results = getmetricsresult(args, rating_list, groundtrue_list)
    users = list(testDict.keys())
    results['recall'] /= float(len(users))
    results['ndcg'] /= float(len(users))
    results['hit'] /= float(len(users))
    return results

def getRatingAndGroundtureList(args, test_loader, Recmodel, testDict, UserItemNet, device, max_K):
    Recmodel = Recmodel.eval()
    with torch.no_grad():
        users_list = []
        rating_list = []
        groundTrue_list = []
        Recmodel.getembedding()
        for batch_users in test_loader:
            batch_data = batch_users[0]
            batchPos = []
            for user in batch_data:
                batchPos.append(UserItemNet[user].nonzero()[1])
            groundTrue = [testDict[u] for u in batch_data.numpy().tolist()]

            users_list.append(batch_data.numpy().tolist())
            groundTrue_list.append(groundTrue)

            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(batchPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            batch_data = batch_data.to(device)
            rating = Recmodel.getUsersRating(batch_data)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            del rating
            rating_list.append(rating_K.cpu())
        return rating_list, groundTrue_list

def getmetricsresult(args, rating_list, groundTrue_list):
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, eval(args.topks)))

        topks = eval(args.topks)
        results = {'hit': np.zeros(len(topks)),
                   'recall': np.zeros(len(topks)),
                   'ndcg': np.zeros(len(topks))}
        for result in pre_results:
            results['recall'] += result['recall']
            results['ndcg'] += result['ndcg']
            results['hit'] += result['hit']
        return results

def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    recall, ndcg, hit = [], [], []
    for k in topks:
        recall.append(calRecall(groundTrue, r, k))
        ndcg.append(calNDCG(groundTrue, r, k))
        hit.append(calHit(r, k))
    return {'recall': np.array(recall),
            'ndcg': np.array(ndcg),
            'hit': np.array(hit)}

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def calPrecision(r, k):
    right_pred = r[:, :k].sum(1)
    precis = np.sum(right_pred) / k
    return precis

def calRecall(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    return recall

def calNDCG(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1

    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    assert len(idcg[idcg == 0.]) == 0
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def calHit(r_batch, k):
    hit_sum = 0.0
    for r in r_batch:
        r = np.array(r)[:k]
        if np.sum(r) > 0:
            hit_sum = hit_sum+1.0
    return hit_sum

