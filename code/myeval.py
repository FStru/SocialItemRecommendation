# !/usr/bin/env python
# -*-coding:utf-8 -*-


import myutil
import myparse
import os
import torch
from warnings import simplefilter
import dataprocess
import myhelper
import pandas as pd
import mymodel
import mydataloader
import mytrain
import mytest
import numpy as np
import random

import sys
sys.path.append("..")
import myxlsx as myxl

def main():
    logger = myutil.get_logger()
    args = myparse.parse_args()
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    myutil.set_seed(args.seed)
    simplefilter(action="ignore", category=FutureWarning)

    logger.info("-------------------->>prepare data<<--------------------")
    trainfile = "../dataset/{}/train.csv".format(args.dataset)
    validfile = "../dataset/{}/valid.csv".format(args.dataset)
    testfile = "../dataset/{}/test.csv".format(args.dataset)
    socialfile = "../dataset/{}/social.csv".format(args.dataset)
    preparedata = dataprocess.DataProcess(dataset=args.dataset, trainfile=trainfile, validfile=validfile,
                                          testfile=testfile, socialfile=socialfile)
    preparedata.interactionProcess()
    n_user = preparedata.n_user
    n_item = preparedata.n_item
    trainDataSize = preparedata.trainDataSize
    usertrainPos = preparedata.usertrainPos
    UserItemNet = preparedata.UserItemNet
    uservalidDic, usertestDic = preparedata.getValidAndTest(testmode=2)
    userneighbor, userdegree = preparedata.getUserInformation()
    itemneighbor, itemdegree = preparedata.getItemInformation()

    interaction_norm_adj = preparedata.getInteractionFilter(filtermode=1)
    interactionGraph = myhelper._convert_sp_mat_to_sp_tensor(interaction_norm_adj)
    interactionGraph = interactionGraph.coalesce().to(device)
    social_norm_adj = preparedata.getSocialFilter(filtermode=1)
    socialGraph = myhelper._convert_sp_mat_to_sp_tensor(social_norm_adj)
    socialGraph = socialGraph.coalesce().to(device)

    logger.info("-------------------->>prepare model<<--------------------")
    Recmodel = mymodel.RFDAT(args, n_user, n_item, interactionGraph, socialGraph, userdegree, itemdegree, device)
    Recmodel.load_state_dict(torch.load(args.pretrainmodel))
    Recmodel = Recmodel.to(device)


    logger.info("-------------------->>testing<<--------------------")
    logger.info("testing from all item...")
    candidate_users = list(usertestDic.keys())
    try:
        assert args.testbatch <= len(candidate_users) / 10
    except AssertionError:
        logger.info(
            f"test_u_batch_size is too big for this dataset, try a small one {len(candidate_users) // 10}")
    candidate_users = torch.Tensor(candidate_users).long()
    testset = torch.utils.data.TensorDataset(candidate_users)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.testbatch, shuffle=True)
    all_results = mytest.test_model(args, test_loader, Recmodel, usertestDic, UserItemNet, device)
    logger.info(all_results)

    HeadTestDic, TailTestDic = preparedata.getUserLongTailTest(args.colddegree)
    logger.info("testing head user from all item...")
    candidate_users = list(HeadTestDic.keys())
    try:
        assert args.testbatch <= len(candidate_users) / 10
    except AssertionError:
        logger.info(
            f"test_u_batch_size is too big for this dataset, try a small one {len(candidate_users) // 10}")
    candidate_users = torch.Tensor(candidate_users).long()
    testset = torch.utils.data.TensorDataset(candidate_users)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.testbatch, shuffle=True)
    head_results = mytest.test_model(args, test_loader, Recmodel, HeadTestDic, UserItemNet, device)
    logger.info(head_results)

    logger.info("testing tail user from all item...")
    candidate_users = list(TailTestDic.keys())
    try:
        assert args.testbatch <= len(candidate_users) / 10
    except AssertionError:
        logger.info(
            f"test_u_batch_size is too big for this dataset, try a small one {len(candidate_users) // 10}")
    candidate_users = torch.Tensor(candidate_users).long()
    testset = torch.utils.data.TensorDataset(candidate_users)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.testbatch, shuffle=True)
    tail_results = mytest.test_model(args, test_loader, Recmodel, TailTestDic, UserItemNet, device)
    logger.info(tail_results)
    # myxl.append_xlsx_detail(args.model + '-' + args.dataset, all_results, head_results, tail_results)

if __name__ == "__main__":
    main()