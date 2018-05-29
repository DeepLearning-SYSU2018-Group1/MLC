# -*- coding:utf-8 -*-
"""
The model and training implementation.
The main function includes trainer and tester, you can choose one of them to run
and the default is tester

update in 2018/5/26 15:35
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
import time

from trainer import ChexnetTrainer
from tester import ChexnetTester


DATA_DIR = '../../medical_report/images/'
IMAGE_LIST_TEST = '../../medical_report/labels/test_data.txt'
IMAGE_LIST_TRAIN = '../../medical_report/labels/train_data.txt'
IMAGE_LIST_VAL = '../../medical_report/labels/val_data.txt'
CKPT_PATH = 'm-17052018-214116.pth.tar'

def main():

    #train the model
    #trainModel()

    #test the model
    testModel()


def trainModel():
    train_dataset = ChestXrayDataSet(data_dir = DATA_DIR,
                                     image_list_file = IMAGE_LIST_TRAIN)
    length = train_dataset.__len__()
    print("The length of training data is ", length)

    #define parameters: address
    dataDir = DATA_DIR
    imageListFileTrain = IMAGE_LIST_TRAIN
    imageListFileVal = IMAGE_LIST_VAL

    #define parameters: time for the name of checkpoint file
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    timestampLaunch = timestampDate + '-' + timestampTime

    #define the parameters of the model
    transResize = 256
    transCrop = 224

    isTrained = True
    classCount = 156

    batchSize = 16
    epochSize = 100
    ChexnetTrainer.train(dataDir, imageListFileTrain, imageListFileVal, transResize, transCrop, isTrained, classCount, batchSize, epochSize, timestampDate, None)


def testModel():
    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=IMAGE_LIST_TEST)
    length = test_dataset.__len__()
    print("The length of test data is ", length)

    #define parameters: address
    dataDir = DATA_DIR
    imageListFileTest = IMAGE_LIST_TEST
    pathModel = CKPT_PATH

    #define the parameters of the model
    transResize = 256
    transCrop = 224

    isTrained = True
    classCount = 156

    batchSize = 1
    epochSize = 100

    ChexnetTester.test(dataDir, imageListFileTest, pathModel, classCount, isTrained, batchSize, transResize, transCrop)


if __name__ == '__main__':
    main()
