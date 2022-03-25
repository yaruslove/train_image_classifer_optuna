# -*- coding: utf-8 -*-
import os
import glob
import argparse
import textwrap
import random
import time
import numpy as np
import datetime
import string
import pandas as pd
import copy
import json

import optuna

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score

from backbones.mobilenet_v2 import mobilenet_v2
# from backbones.mobilenetv3 import mobilenetv3_large
from backbones.mobilenetv3_pytorch import mobilenet_v3_large
from backbones.mobilenetv3_pytorch import mobilenet_v3_small

from backbones import resnet
from backbones import regnet
from backbones import efficientnet
from backbones import mobilenetv3

from utils.dataloader import DS
from utils.valid import valid
import gc

import warnings

warnings.filterwarnings('ignore')



SEED = 1236

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Classifier training program',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                         Data dir example:
                                            data/ <-- path to this dir [--data] argument
                                            ├──class1/
                                            │  ├──train/
                                            │  │  ├──<train images>
                                            │  ├──valid/
                                            │  │  ├──<valid images>
                                            │  └──test/
                                            │     └──<test images>
                                            ├──class2/
                                            │  ├──train/
                                            │  │  ├──<train images>
                                            │  ├──valid/
                                            │  │  ├──<valid images>
                                            │  └──test/
                                            │     └──<test images>
                                            │  ...
                                            └──classN/
                                               ├──train/
                                               │  ├──<train images>
                                               ├──valid/
                                               │  ├──<valid images>
                                               └──test/
                                                  └──<test images>
                                         '''))
    parser.add_argument('-d', '--data', type=str, required=True)

    parser.add_argument('-bb', '--backbone', type=str, default='resnet34',
                        choices=['mobilenet_v3_large','mobilenet_v3_small','mobilenet_v2', 
                                 'resnet18', 'resnet34', 'resnet50',
                                 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                                'regnet_x_400mf','regnet_x_800mf','regnet_x_1_6gf','regnet_x_3_2gf'])
    parser.add_argument('--classes', type=str, nargs='*', default=None)
    parser.add_argument('--resolush', type=int, default=224)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=16)


    
    args = parser.parse_args()
    resolush = int(args.resolush)

    


    
######## Device check ########
    if args.device is None:
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
#             if torch.cuda.device_count() > 1:
#                 net = nn.DataParallel(net)

    else:
        device = args.device
        

######## Check classes ########

    if args.classes is None:
        classes = os.listdir(args.data)
    else:
        classes = args.classes
    classes.sort()
    
    print(f'Classes : {classes}')
    
    

######## Upload model and weight ######## 
    if args.backbone == 'mobilenet_v2':
        model = mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    elif args.backbone == 'mobilenet_v3_large':
        model = mobilenet_v3_large()
        model.load_state_dict(torch.load('./pretrain_weight/mobilenet_v3_large-8738ca79.pth'))
        model.classifier[3]=nn.Linear(model.classifier[3].in_features, len(classes))
    elif args.backbone == 'mobilenet_v3_small':
        model = mobilenet_v3_small()
        model.load_state_dict(torch.load('./pretrain_weight/mobilenet_v3_small-047dcff4.pth'))
        model.classifier[3]=nn.Linear(model.classifier[3].in_features, len(classes))
    elif args.backbone in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']:
        model = efficientnet.__dict__[args.backbone](pretrained=False)
        model.classifier[1]=nn.Linear(in_features=model.classifier[1].in_features, out_features=len(classes), bias=True)
    elif args.backbone in ['regnet_x_400mf','regnet_x_800mf','regnet_x_1_6gf','regnet_x_3_2gf']:
        model = regnet.__dict__[args.backbone](pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, out_features=len(classes), bias=True)
    elif args.backbone in ['resnet18', 'resnet34', 'resnet50']:
        model = resnet.__dict__[args.backbone](pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(classes))


    model.load_state_dict(torch.load(args.weights))
    model.to(device)
    model.share_memory()
    model.eval()    

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)


    # class handOver_model:
    #     def __init__(self, model):
    #         self.__model = model
        
    #     def return_pure_model(self):
    #         sample_model = copy.deepcopy(self.__model)
    #         return sample_model

    # sample_model=handOver_model(model)


    print('Backbone: {}'.format(args.backbone))
        
######## Create set paths images ########
    train_images = []
    valid_images = []
    test_images = []
    sub_dirs = ['train', 'valid', 'test']

    for c in classes:
        for idx, sd in enumerate(sub_dirs):
            tmp = glob.glob(f'{args.data}/{c}/{sd}/*.jpg')
            tmp += glob.glob(f'{args.data}/{c}/{sd}/*.png')
            tmp += glob.glob(f'{args.data}/{c}/{sd}/*.JPG')
            tmp += glob.glob(f'{args.data}/{c}/{sd}/*.JPEG')
            tmp += glob.glob(f'{args.data}/{c}/{sd}/*.PNG')

            if idx == 0:
                for t in tmp:
                    train_images.append(t)
            elif idx == 1:
                for t in tmp:
                    valid_images.append(t)
            elif idx == 2:
                for t in tmp:
                    test_images.append(t)


    num_workers=args.num_workers
    resolush =int(args.resolush)

######## Create Dataset ########
    train_dataset = DS(train_images, classes=classes,resolush=resolush)
    valid_dataset = DS(valid_images, classes=classes, use_albu=False, resolush=resolush)
    test_dataset = DS(test_images, classes=classes, use_albu=False, resolush=resolush)


    train_dataloader = DataLoader(train_dataset, batch_size=8, num_workers=num_workers, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, num_workers=num_workers, shuffle=True, pin_memory=True)
    
    test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers=num_workers, shuffle=True, pin_memory=True) # ,persistent_workers=True    


    criteria = nn.CrossEntropyLoss()
 
    ############  Процесс валидации ############       
    
    valid_loss, valid_acc = valid(model, valid_dataloader, criteria, device)
    
    print("valid_loss",valid_loss)  
    print("valid_acc",valid_acc)  

    # ############  Exec test on each epoch ############ 
    #         exec_test=False # Заглушка      
    #         if exec_test==True:
    #             _ # make test
    #         else:
    #             reporter['test_loss']=None
    #             reporter['test_acc']=None
    # ############  Drop info in pandas df and csv ############

    #         reporter_df = reporter_df.append(reporter, ignore_index=True)
    #         reporter_df['Epoch'] = reporter_df['Epoch'].astype(int)
    #         reporter_df.to_csv(os.path.join(path_expirement,'result.csv'), sep='\t')
    #         for key, value in reporter.items():
    #             print(str(key)+"="+str(value))
            
    #         ############ Save model if it in best of n ############ 
    #         list_checkpoints=copy.deepcopy(reporter_df.sort_values('valid_loss')['Name_model'].tolist())
    #         if name_pmodel in list_checkpoints[:save_best]:
    #             torch.save(model.state_dict(), os.path.join(path_expirement,name_pmodel))
                
    #         ############ Delete not best checkpoints ############
    #         if len(list_checkpoints)>save_best:
    #             for del_check in list_checkpoints[save_best:]:
    #                 if os.path.exists(os.path.join(path_expirement,del_check)):
    #                     os.remove(os.path.join(path_expirement,del_check))
    #                     print("The file {} has been deleted successfully.".format(del_check))
                
    #         del list_checkpoints
            
    #         print('Epoch_finished')
    #         print("Epoch = {} have done!".format(e))
            

    # ############  Тестирование лучшей модели ############  
    #     namebest_model=reporter_df.loc[reporter_df['valid_loss'].idxmin(), 'Name_model']

    #     print('namebest_model',namebest_model)
    #     if not notest_set:
    #         model.load_state_dict(torch.load(os.path.join(path_expirement,namebest_model)))
    #         test_loss, test_acc = valid(model, test_dataloader, criteria, device)
    #         reporter_test={}
    #         reporter_test['test_loss']=test_loss
    #         reporter_test['test_acc']=test_acc

    #         print('### Test best model selected by min valid value: ###')
    #         for key, value  in reporter_test.items():
    #             reporter_df.loc[reporter_df['valid_loss'].idxmin(), key]=value
    #             print("{} = {}".format(key, str(value)))

    # ############  Drop info in pandas df and csv, and json ############
    #     reporter_df.to_csv(os.path.join(path_expirement,'result.csv'), sep='\t')

    #     reporter_json=reporter_df.to_json(orient="index")
    #     with open(os.path.join(path_expirement,'result.json'), "w") as outfile:
    #         outfile.write(reporter_json)

    # ############  For function Optuna tune Hyperparam we return min valid loss ############
    #     valid_loss_best_min=reporter_df.loc[reporter_df['valid_loss'].idxmin(), 'valid_loss']

    #     del reporter_json
    #     del reporter_df
    #     del param_experiment
    #     print("Experiment finished, all epochs = {} have done !!!".format(epochs))
        
    #     gc.collect()
                               
    #     return valid_loss_best_min
    
    # # Tensor board
    # summarize_results=path_allresult+"/summarize_results"
    # if not os.path.exists(summarize_results):
    #     os.makedirs(summarize_results)
    # writer = SummaryWriter(log_dir=summarize_results, flush_secs=1)
    
    # # main_func
    # study = optuna.create_study(study_name="pytorch_checkpoint",direction="minimize") # "maximize"
    # study.optimize(objective, n_trials=n_trials)