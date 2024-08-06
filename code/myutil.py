# !/usr/bin/env python
# -*-coding:utf-8 -*-

import logging
import datetime
import numpy as np
import torch
import os

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def get_logger():
    if not os.path.exists("./log"):
        os.makedirs("./log", exist_ok=True)
    logger = logging.getLogger('log_test')
    logger.setLevel(level=logging.DEBUG)
    local_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_handler = logging.FileHandler(filename='log/{}.txt'.format(local_time), encoding="utf-8", mode='w')
    file_handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger