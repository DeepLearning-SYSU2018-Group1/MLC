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

    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
        return outAUROC

    def test (dataDir, imageListFileTest, pathModel, isTrained, classCount, batchSize, transResize, transCrop):   
        
        print("test begins!=======================")

        CLASS_NAMES = ['base','lung, hyperlucent','right','retrocardiac','costophrenic angle','airspace disease','blunted','lung','catheters, indwelling','left',
        'pleural effusion','posterior','small','normal','granulomatous disease','thorax','round','chronic','multiple','density','borderline','cardiomegaly','mild',
        'bronchovascular','hypoinflation','markings','interstitial','bilateral','prominent','hyperdistention','nipple shadow','scoliosis','elevated','diaphragm',
        'lumbar vertebrae','scattered','deformity','pulmonary atelectasis','opacity','ribs','cicatrix','upper lobe','calcified granuloma','thoracic vertebrae',
        'spondylosis','medical device','atherosclerosis','degenerative','moderate','calcinosis','aorta','spine','pleura','thickening','aorta, thoracic','granuloma',
        'tortuous','hilum','focal','diffuse','foreign bodies','mediastinum','surgical instruments','clavicle','severe','implanted medical device','lymph nodes',
        'aortic valve','lower lobe','pneumonectomy','nodule','lingula','middle lobe','mass','pulmonary edema','blood vessels','emphysema','infiltrate','large',
        'patchy','apex','pulmonary disease, chronic obstructive','osteophyte','streaky','flattened','kyphosis','sclerosis','lucency','humerus','cysts',
        'lung diseases, interstitial','cardiac shadow','enlarged','tube, inserted','obscured','hernia, hiatal','heart','pulmonary fibrosis','pulmonary emphysema',
        'cystic fibrosis','bronchiectasis','stents','abdomen','hyperostosis, diffuse idiopathic skeletal','spinal fusion','pneumonia','breast','mastectomy',
        'consolidation','fractures, bone','healed','no indexing','heart failure','pulmonary congestion','sulcus','azygos lobe','bullous emphysema','irregular',
        'pulmonary alveoli','epicardial fat','pneumothorax','anterior','technical quality of image unsatisfactory','pulmonary artery','paratracheal','bone diseases, metabolic',
        'diaphragmatic eventration','neck','shoulder','dislocations','pneumoperitoneum','trachea, carina','sutures','blister','abnormal','cervical vertebrae','arthritis',
        'shift','trachea','aortic aneurysm','hypertension, pulmonary','sternum','pericardial effusion','reticular','heart atria','adipose tissue','coronary vessels',
        'volume loss','hydropneumothorax','sarcoidosis','breast implants','cavitation','funnel chest','bronchi','heart ventricles','contrast media']
        
        cudnn.benchmark = True
        
        model = CheXNet(classCount, isTrained).cuda()
        model = torch.nn.DataParallel(model).cuda() 
        
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])
        
        # data transforms
        # ---------------
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transform=transforms.Compose(transformList)

        # datasets
        # ---------------
        datasetTest = ChestXrayDataSet(data_dir = dataDir,
                                        image_list_file = imageListFileTest,
                                        transform = transform)
        print(datasetTest.__len__())
        dataLoaderTest = DataLoader(dataset = datasetTest, batch_size = batchSize,
                                     shuffle = False, num_workers = 8, pin_memory = True)


        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
       
        model.eval()
        
        for i, (inp, target) in enumerate(dataLoaderTest):
            
            target = target.cuda()
            outGT = torch.cat((outGT, target), 0)
            
            bs, n_crops, c, h, w = inp.size()
            
            varInput = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
            
            out = model(varInput)
            outMean = out.view(bs, n_crops, -1).mean(1)
            
            outPRED = torch.cat((outPRED, outMean.data), 0)

        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, classCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        
        with open('./AUROCresult.txt', "w") as f:
            f.write('AUROC mean :')
            f.write(str(aurocMean))
            f.write('\n')
            for i in range (0, len(aurocIndividual)):
                print('The AUROC of {} is {}'.format(CLASS_NAMES[i], aurocIndividual[i]))
                f.write('The AUROC of {} is {}'.format(CLASS_NAMES[i], aurocIndividual[i]))
                f.write('\n')
            f.close()
        
        return


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
