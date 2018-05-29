# -*- coding:utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader

from sklearn.metrics.ranking import roc_auc_score
from read_data import ChestXrayDataSet

class ChexnetTester(object):
    """docstring for ChexnetTester"""
    


    def test(dataDir, imageListFileTest, pathModel, classCount, isTrained, tsBatchSize, transResize, transCrop):

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

        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        print("test begins!=======================")
        model = CheXNet(classCount, isTrained).cuda()
        model = torch.nn.DataParallel(model).cuda()
        if os.path.isfile(pathModel):
            print("=> loading checkpoint")
            checkpoint = torch.load(pathModel)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint")
        else:
            print("=> no checkpoint found")


        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)

#def test(pathDirData, pathFileTest, pathModel, classCount, isTrained, tsBatchSize, transResize, transCrop, launchTimeStamp):

        #dataset
        datasetTest = ChestXrayDataSet(data_dir = dataDir,
                                        image_list_file = imageListFileTest,
                                        transform = transformSequence)
        print(datasetTest.__len__())
        dataLoaderTest = DataLoader(dataset = datasetTest, batch_size = tsBatchSize,
                                     shuffle = False, num_workers = 8, pin_memory = True)


        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()

        model.eval()

        for i, (input, target) in enumerate(dataLoaderTest):

            target = target.cuda()
            outGT = torch.cat((outGT, target), 0)

            bs, n_crops, c, h, w = input.size()

            varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())

            out = model(varInput)
            outMean = out.view(bs, n_crops, -1).mean(1)

            outPRED = torch.cat((outPRED, outMean.data), 0)

        aurocIndividual = ChexnetTester.computeAUROC(outGT, outPRED, classCount)
        aurocMean = np.array(aurocIndividual).mean()

        print ('AUROC mean ', aurocMean)

        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', aurocIndividual[i])

        return

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


#--------------------------------------------------------------------------------
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
