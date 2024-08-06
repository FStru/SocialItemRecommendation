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

    first_path = os.path.join("./middata", args.dataset)
    i2i_pair = pd.read_csv(os.path.join(first_path, "yelpH.csv"), index_col=0)
    itemtrainPos, itemtrainSize = myhelper.i2i_pairs_process(i2i_pair, n_item)

    logger.info("-------------------->>prepare model<<--------------------")
    Recmodel = mymodel.RFDAT(args, n_user, n_item, interactionGraph, socialGraph, userdegree, itemdegree, device)
    Recmodel = Recmodel.to(device)
    opt = torch.optim.Adam(Recmodel.parameters(), lr=args.lr)
    if args.lr_decay:
        StepLR = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_len,
                                                 gamma=args.lr_decay_rate)
    best_ndcg = 0
    low_count, low_count_cold = 0, 0

    logger.info("-------------------->>training<<--------------------")
    for epoch in range(args.epochs):
        generator_sampleSet = mydataloader.generate_trainsample(n_item, n_item, itemtrainPos, itemtrainSize)
        generator_trainloader = mydataloader.generate_traindataloader(generator_sampleSet, args.g_batch)

        etgsr_sampleSet, negativeSet = mydataloader.generate_trainsample_multi(n_user, n_item, usertrainPos, trainDataSize,
                                                                         negsample=args.negativenum)
        etgsr_trainloader = mydataloader.generate_traindataloader_multi(etgsr_sampleSet, negativeSet, args.trainbatch)

        batch_loss_avg, etgsr_loss_avg, u2i_loss_avg, l2_loss_avg, generator_loss_avg, i2i_loss_avg, reg_loss_avg = \
            mytrain.train_emtsr_itemfilter(args, generator_trainloader, etgsr_trainloader, Recmodel, opt, device)

        logger.info("epoch:{}-BatchLossAvg:{}".format(epoch, batch_loss_avg))
        if args.lr_decay:
            StepLR.step()
        low_count += 1

        if (epoch + 1) % 1 == 0 or (epoch + 1) == args.epochs:
            logger.info("validing...")
            candidate_users = list(uservalidDic.keys())
            try:
                assert args.testbatch <= len(candidate_users) / 10
            except AssertionError:
                logger.info(
                    f"test_u_batch_size is too big for this dataset, try a small one {len(candidate_users) // 10}")
            candidate_users = torch.Tensor(candidate_users).long()
            testset = torch.utils.data.TensorDataset(candidate_users)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=args.testbatch, shuffle=True)

            results = mytest.test_model(args, test_loader, Recmodel, uservalidDic, UserItemNet, device)
            logger.info(results)
            if results['ndcg'][0] < best_ndcg:
                if low_count == 100:
                    break
            else:
                save_model_file = "{}-bestmodel.pth".format(args.model)
                save_model_file = os.path.join(args.save, save_model_file)
                logger.info("update model...")
                best_ndcg = results['ndcg'][0]
                low_count = 0
                torch.save(Recmodel.state_dict(), save_model_file)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            logger.info("testing...")
            candidate_users = list(usertestDic.keys())
            try:
                assert args.testbatch <= len(candidate_users) / 10
            except AssertionError:
                logger.info(
                    f"test_u_batch_size is too big for this dataset, try a small one {len(candidate_users) // 10}")
            candidate_users = torch.Tensor(candidate_users).long()
            testset = torch.utils.data.TensorDataset(candidate_users)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=args.testbatch, shuffle=True)

            testresults = mytest.test_model(args, test_loader, Recmodel, usertestDic, UserItemNet, device)
            logger.info(testresults)

    logger.info("results：")
    TestRecmodel = mymodel.RFDAT(args, n_user, n_item, interactionGraph, socialGraph, userdegree, itemdegree, device)
    TestRecmodel.load_state_dict(torch.load(save_model_file))
    TestRecmodel = TestRecmodel.to(device)

    candidate_users = list(usertestDic.keys())
    try:
        assert args.testbatch <= len(candidate_users) / 10
    except AssertionError:
        logger.info(
            f"test_u_batch_size is too big for this dataset, try a small one {len(candidate_users) // 10}")
    candidate_users = torch.Tensor(candidate_users).long()
    testset = torch.utils.data.TensorDataset(candidate_users)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.testbatch, shuffle=True)
    testresults = mytest.test_model(args, test_loader, TestRecmodel, usertestDic, UserItemNet, device)
    logger.info(testresults)

if __name__ == "__main__":
    main()