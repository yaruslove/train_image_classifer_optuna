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
    parser.add_argument('-p', '--param', type=str, required=True)
    parser.add_argument('--path-save', type=str, required=True)
    parser.add_argument('--save-best', type=int, default=3)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--classes', type=str, nargs='*', default=None)
    parser.add_argument('--num-workers', type=int, default=16)

    args = parser.parse_args()
    
    best_valid_loss = float('inf')

######## Device check ########
    if args.device is None:
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"

######## Check classes ########

    if args.classes is None:
        classes = os.listdir(args.data)
    else:
        classes = args.classes
    classes.sort()
    
    print(f'Classes : {classes}')

############  Args parse Hyperparam ############  
    path_param=args.param
    param_experiment=pd.read_csv(path_param, sep="\t").iloc[:, 1:].to_dict("list")
    for i in  param_experiment:
        param_experiment[i]=param_experiment[i][0]
    """
    {'lr': 0.00082,
 'batch_size': 1344.0,
 'epochs': 350.0,
 'resolush': 224.0,
 'backbone': 'resnet18',
 'hash_exprmnt': 'z6jxs9na'}
 """
    lr=param_experiment["lr"]
    batch_size=int(param_experiment["batch_size"])
    epochs=int(param_experiment["epochs"])
    resolush=int(param_experiment["resolush"])
    backbone=param_experiment["backbone"]
    hash_exprmnt=param_experiment["hash_exprmnt"]
    
    num_workers=args.num_workers
    save_best=args.save_best

######## Upload model and weight ######## 
    if backbone == 'mobilenet_v2':
        model = mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    elif backbone == 'mobilenet_v3_large':
        model = mobilenet_v3_large()
        model.load_state_dict(torch.load('./pretrain_weight/mobilenet_v3_large-8738ca79.pth'))
        model.classifier[3]=nn.Linear(model.classifier[3].in_features, len(classes))
    elif backbone == 'mobilenet_v3_small':
        model = mobilenet_v3_small()
        model.load_state_dict(torch.load('./pretrain_weight/mobilenet_v3_small-047dcff4.pth'))
        model.classifier[3]=nn.Linear(model.classifier[3].in_features, len(classes))
    elif backbone in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']:
        model = efficientnet.__dict__[backbone](pretrained=False)
        model.load_state_dict(torch.load('./pretrain_weight/efficientnet_b0_rwightman-3dd342df.pth'))
        model.classifier[1]=nn.Linear(in_features=model.classifier[1].in_features, out_features=len(classes), bias=True)
    elif backbone in ['regnet_x_400mf','regnet_x_800mf','regnet_x_1_6gf','regnet_x_3_2gf']:
        model = regnet.__dict__[backbone](pretrained=False)
        model.load_state_dict(torch.load('./pretrain_weight/regnet_x_400mf-adf1edd5.pth'))
        model.fc = nn.Linear(model.fc.in_features, out_features=len(classes), bias=True)
    elif backbone in ['resnet18', 'resnet34', 'resnet50']:
        model = resnet.__dict__[backbone](pretrained=False)
        model.load_state_dict(torch.load('./pretrain_weight/resnet18-5c106cde.pth'))
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        
    
    
    class handOver_model:
        def __init__(self, model):
            self.__model = model

        def return_pure_model(self):
            sample_model = copy.deepcopy(self.__model)
            return sample_model

    sample_model=handOver_model(model)
    
    print('Backbone: {}'.format(backbone))

    
    model.to(device)
    # model = nn.DataParallel(model,device_ids = [0,1])
    model.share_memory()   

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

######## Create Dataset ########
    train_dataset = DS(train_images, classes=classes, use_albu=True, resolush=resolush)
    valid_dataset = DS(valid_images, classes=classes, use_albu=False, resolush=resolush)
    test_dataset = DS(test_images, classes=classes, use_albu=False, resolush=resolush)

######## Create Dataloader ########
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True) # ,persistent_workers=True

    
######## Create dir for expiremets ########
    path_save=args.path_save
    offset = datetime.timezone(datetime.timedelta(hours=3))
    d = datetime.datetime.now(offset) # Convert moscow time
    msc_time=str(d.date())+'_'+str(d.time())[:str(d.time()).find('.')].replace(':', '-')
   
    str_params=backbone+'_batch='+str(batch_size)+'_lr='+str(lr)+'_epochs='+str(epochs)+"_resolush="+str(resolush)
    path_result=os.path.join(path_save,hash_exprmnt+'_'+msc_time+'_'+str_params)
    
######## Check exist this experiment already ########
    exprt_exist=False
    for i in os.listdir(path_save):
        if i.startswith(hash_exprmnt):
            exprt_exist=True
            break

    if exprt_exist:
        raise NameError('Dir with experiment',path_result, 'already exist!')
    else:
        os.mkdir(path_result)



############  Write Hyper-parametrs in csv file   ############        
    df_experiment = pd.DataFrame()
    df_experiment = df_experiment.append(param_experiment, ignore_index=True)
    df_experiment.to_csv(os.path.join(path_result,'param_expirement.csv'), sep='\t')
    del df_experiment
    

############ Tensor board ############
    summarize_results=path_result+"/summarize_tensorboard"
    if not os.path.exists(summarize_results):
        os.makedirs(summarize_results)
    writer = SummaryWriter(log_dir=summarize_results, flush_secs=1)
    
    
######## Create Optimizer Criteria ########
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    STEPS_PER_EPOCH = len(train_dataloader)
    TOTAL_STEPS = (epochs+1) * STEPS_PER_EPOCH
    MAX_LRS = [p['lr'] for p in optimizer.param_groups]
    scheduler_lr = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                        max_lr = MAX_LRS,
                                        total_steps = TOTAL_STEPS)

    criteria = nn.CrossEntropyLoss()

######## Info about experemets ########
    reporter_df = pd.DataFrame()

        
############  Процесс тренировки ############
    for e in tqdm(range(epochs)):
        print('Start epoch '+str(e))
        reporter={} # For keep information loss, accuracy for each epoch
        start = time.time()
        model.train()

        train_loss=[]

        label_true=[]
        label_pred=[]

        for idx, (imgs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            imgs = imgs.to(device)
            labels = labels.to(device)

            with autocast():
                out = model(imgs)
                loss = criteria(out, labels)

            out = torch.argmax(out, dim=1)

            train_loss=train_loss+[round(loss.item(), 3)]*len(labels)

            label_true=label_true+labels.int().flatten().cpu().tolist()
            label_pred=label_pred+out.int().flatten().cpu().tolist()

            loss.backward()
            optimizer.step()
            scheduler_lr.step()


        train_loss=sum(train_loss) / len(train_loss)
        train_acc=accuracy_score(label_true, label_pred)
        # print("train_loss", train_loss)
        # print("train_acc",  train_acc)

        print(" ")

        reporter['Epoch']=int(e)
        reporter['train_loss']=train_loss
        reporter['train_acc']=train_acc

############  Add param (lr,batch ..) to reporter ############             
        for param_key, param_value in param_experiment.items():
            reporter[param_key]=param_value
        reporter['lr_current']=optimizer.param_groups[0]["lr"]


############  Процесс валидации ############       
        model.eval()
        valid_loss, valid_acc = valid(model, valid_dataloader, criteria, device)

        name_pmodel='checkpoint_'+str(e).zfill(4)+'.pth'
        reporter['Name_model']=name_pmodel

        ############ Tensor board ############
        writer.add_scalars('Loss_value', {'Train_'+str_params: train_loss, 'Valid_'+str_params: valid_loss}, e)

        end = time.time()
        reporter['valid_loss']=valid_loss
        reporter['valid_acc']=valid_acc
        reporter['time']=round((end - start), 2)


############  Exec test on each epoch ############ 
        exec_test=False # Заглушка      
        if exec_test==True:
            _ # make test
        else:
            reporter['test_loss']=None
            reporter['test_acc']=None
############  Drop info in pandas df and csv ############

        reporter_df = reporter_df.append(reporter, ignore_index=True)
        reporter_df['Epoch'] = reporter_df['Epoch'].astype(int)
        reporter_df.to_csv(os.path.join(path_result,'result.csv'), sep='\t')
        for key, value in reporter.items():
            print(str(key)+"="+str(value))

        ############ Save model if it in best of n ############ 
        list_checkpoints=copy.deepcopy(reporter_df.sort_values('valid_loss')['Name_model'].tolist())
        if name_pmodel in list_checkpoints[:save_best]:
            torch.save(model.state_dict(), os.path.join(path_result,name_pmodel))

        ############ Delete not best checkpoints ############
        if len(list_checkpoints)>save_best:
            for del_check in list_checkpoints[save_best:]:
                if os.path.exists(os.path.join(path_result,del_check)):
                    os.remove(os.path.join(path_result,del_check))
                    print("The file {} has been deleted successfully.".format(del_check))

        del list_checkpoints

        print('Epoch_finished')
        print("Epoch = {} have done!".format(e))


############  Тестирование лучшей модели ############  
    namebest_model=reporter_df.loc[reporter_df['valid_loss'].idxmin(), 'Name_model']

    print('namebest_model',namebest_model)
    if not notest_set:
        model.load_state_dict(torch.load(os.path.join(path_result,namebest_model)))
        test_loss, test_acc = valid(model, test_dataloader, criteria, device)
        reporter_test={}
        reporter_test['test_loss']=test_loss
        reporter_test['test_acc']=test_acc

        print('### Test best model selected by min valid value: ###')
        for key, value  in reporter_test.items():
            reporter_df.loc[reporter_df['valid_loss'].idxmin(), key]=value
            print("{} = {}".format(key, str(value)))

############  Drop info in pandas df and csv, and json ############
    reporter_df.to_csv(os.path.join(path_result,'result.csv'), sep='\t')

    reporter_json=reporter_df.to_json(orient="index")
    with open(os.path.join(path_result,'result.json'), "w") as outfile:
        outfile.write(reporter_json)

############  For function Optuna tune Hyperparam we return min valid loss ############
    valid_loss_best_min=reporter_df.loc[reporter_df['valid_loss'].idxmin(), 'valid_loss']

    del reporter_json
    del reporter_df
    del param_experiment
    print("Experiment finished, all epochs = {} have done !!!".format(epochs))

    gc.collect()