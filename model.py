"""
The model and training implementation.
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


DATA_DIR = '../../medical_report/images/'
IMAGE_LIST_TEST = '../../medical_report/labels/test_data.txt'
IMAGE_LIST_TRAIN = '../../medical_report/labels/train_data.txt'
IMAGE_LIST_VAL = '../../medical_report/labels/val_data.txt'

def main():

    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=IMAGE_LIST_TEST)

    length = test_dataset.__len__()
    print ("The length of test data is ", length)

    # (image_name, label, image) = test_dataset.__getitem__(0)
    # print ("The path of the first image is ", image_name, ", the lable of it is ", label)
    # (image, label) = test_dataset.__getitem__(0)
    # print ("The lable of the first image is ", label)

    dataDir = DATA_DIR
    imageListFileTrain = IMAGE_LIST_TRAIN
    imageListFileVal = IMAGE_LIST_VAL
    imageListFileTest = IMAGE_LIST_TEST

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    transResize = 256
    transCrop = 224

    isTrained = True
    classCount = 156

    batchSize = 16
    test_batchSize = 8
    epochSize = 100

    pathModel = '../../MLC/m-17052018-214116.pth.tar'

    # ChexnetTrainer.train(dataDir, imageListFileTrain, imageListFileVal, transResize, transCrop, isTrained, classCount, batchSize, epochSize, timestampLaunch, None);
    ChexnetTrainer.test (dataDir, imageListFileTest, pathModel, isTrained, classCount, test_batchSize, transResize, transCrop)

if __name__ == '__main__':
    main()
