# !/usr/bin/env python
# -*-coding:utf-8 -*-

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RFDAT')
    parser.add_argument('--dataset', type=str, default='douban')
    parser.add_argument('--pretrainmodel', type=str, default='./save')
    parser.add_argument('--preitemrelation', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--lr_decay_len', type=int, default=30)
    parser.add_argument('--lr_decay_rate', type=float, default=0.8)
    parser.add_argument('--negativenum', type=int, default=1)
    parser.add_argument('--ilayers', type=int, default=1)
    parser.add_argument('--slayers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--trainbatch', type=int, default=2048)
    parser.add_argument('--testbatch', type=str, default=100)
    parser.add_argument('--hdim', type=int, default=64)
    parser.add_argument('--l2decay', type=float, default=1e-4, help="the weight decay for l2 normalization")
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--topks', nargs='?', default="[10, 20, 30]")
    parser.add_argument('--colddegree', type=int, default=10)
    parser.add_argument('--g_batch', type=int, default=2048)
    parser.add_argument("--g_convergence", type=int, default=40)
    parser.add_argument("--g_lr", type=float, default=0.01)
    parser.add_argument("--g_linktopk", type=int, default=5)
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--save', type=str, default='./save')
    args = parser.parse_args()
    return args