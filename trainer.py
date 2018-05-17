import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score
from read_data import ChestXrayDataSet


class ChexnetTrainer():

    def train(dataDir, imageListFileTrain, imageListFileVal, transResize, transCrop, isTrained, classCount, batchSize, epochSize, launchTimestamp, checkpoint):
        """Train the network.

        Args:
            dataDir - path to the data dir
            imageListFileTrain - path to the iamge list file to train
            imageListFileVal - path to the iamge list file to train
            transResize - size of the image to scale down to
            transCrop - size of the cropped image
            isTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
            classCount - number of output classes
            batchSize - batch size
            epochSize - number of epochs
            launchTimestamp - date/time, used to assign unique name for the checkpoint file
            checkpoint - if not None loads the model and continues training

        """

        # SETTINGS
        # ^^^^^^^^

        # initialize and load the model
        # ---------------
        print("train begins!=======================")
        model = CheXNet(classCount, isTrained).cuda()
        model = torch.nn.DataParallel(model).cuda()

        # data transforms
        # ---------------
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transform=transforms.Compose(transformList)

        # datasets
        # ---------------
        datasetTrain = ChestXrayDataSet(data_dir = dataDir,
                                        image_list_file = imageListFileTrain,
                                        transform = transform)
        datasetVal = ChestXrayDataSet(data_dir = dataDir,
                                        image_list_file = imageListFileVal,
                                        transform = transform)

        print(datasetTrain.__len__())
        dataLoaderTrain = DataLoader(dataset = datasetTrain, batch_size = batchSize,
                                     shuffle = False, num_workers = 8, pin_memory = True)
        dataLoaderVal = DataLoader(dataset = datasetVal, batch_size = batchSize,
                                     shuffle = False, num_workers = 8, pin_memory = True)

        # optimizer and scheduler
        # ---------------
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')

        # loss
        # ---------------
        loss = torch.nn.BCELoss(size_average = True)

        # Load checkpoint
        # ---------------
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        # Train
        # ^^^^^

        # TODO: train, epochTrain and epochVal

        lossMin = 100000

        for epochIdx in range(0, epochSize):
            print("EpochIdx: ################ ",epochIdx)
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime

            ChexnetTrainer.epochTrain(model, dataLoaderTrain, optimizer, scheduler, classCount, loss)
            lossVal = ChexnetTrainer.epochVal(model, dataLoaderVal, optimizer, scheduler, classCount, loss)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            scheduler.step(lossVal)

            if lossVal < lossMin:
                lossMin = lossVal
                torch.save({'epoch': epochIdx + 1, 'state_dict': model.state_dict(), 'best_loss': lossMin, 'optimizer' : optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
                print ('Epoch [' + str(epochIdx + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print ('Epoch [' + str(epochIdx + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))


    def epochTrain(model, dataLoader, optimizer, scheduler, classCount, loss):

        model.train()

        for batchIdx, (input, target) in enumerate(dataLoader):

            target = target.cuda(async = True)

            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)
            varOutput = model(varInput)

            lossvalue = loss(varOutput, varTarget)

            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            print(batchIdx)


    def epochVal(model, dataLoader, optimizer, scheduler, classCount, loss):

        model.eval()
        lossVal = 0
        lossValNorm = 0

        for i, (input, target) in enumerate(dataLoader):
            target = target.cuda(async = True)

            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)
            varOutput = model(varInput)

            losstensor = loss(varOutput, varTarget)

            lossVal += float(losstensor.item())
            lossValNorm += 1
            del losstensor
        outLossVal = lossVal / lossValNorm

        return outLossVal


def compute_AUROCs(gt, pred, classCount):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt - ground truth data
        pred - predicted data

    Returns:
        List of AUROCs of all classes.

    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(classCount):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


class CheXNet(nn.Module):
    """CheXNet - Modified standard DenseNet121.

    The final fully connected layer was replaced by one that has a single output,
    after which a sigmoid nonlinearity was applied.

    """
    def __init__(self, classCount, isTrained):
        super(CheXNet, self).__init__()
        self.cheXNet = torchvision.models.densenet121(pretrained = isTrained)
        filterCount = self.cheXNet.classifier.in_features
        self.cheXNet.classifier = nn.Sequential(
            nn.Linear(filterCount, classCount),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cheXNet(x)
        return x
