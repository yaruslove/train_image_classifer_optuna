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

import optuna

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score

from backbones.mobilenet_v2 import mobilenet_v2
from backbones.mobilenetv3 import mobilenetv3_large
from backbones import resnet

from utils.dataloader import DS
from utils.train import train
from utils.valid import valid
from utils.test import test
import gc



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
    parser.add_argument('--name', type=str, default='Default')
    parser.add_argument('--path-save', type=str, required=True)

    parser.add_argument('-bb', '--backbone', type=str, default='resnet34',
                        choices=['mobilenetv3_large','mobilenet_v2', 'resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--classes', type=str, nargs='*', default=None)
    parser.add_argument('--save-best', type=int, default=3)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=32)

    args = parser.parse_args()
    num_workers=args.num_workers
    save_best=args.save_best
    
    best_valid_loss = float('inf')

    
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
        
        
######## Create trail dir ########
    path_save=args.path_save
#     path_save='/home/jovyan/train/seat_belt_validation/RESULT_OUT/'
#     name='seat_belt'
    name=args.name
    offset = datetime.timezone(datetime.timedelta(hours=3))
    d = datetime.datetime.now(offset) # Convert moscow time
    msc_time=str(d.date())+'_'+str(d.time())[:str(d.time()).find('.')].replace(':', '-')
    path_allresult=os.path.join(path_save,name+'_'+msc_time)
    os.mkdir(path_allresult)


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
    elif args.backbone == 'mobilenetv3_large':
        model = mobilenetv3_large()
        # model.load_state_dict(torch.load('/disk/mnt/disk3/home/volkonskiy-yi/models/mobile_netv3/mobilenetv3-large-1cd25616.pth'))
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    else:
        model = resnet.__dict__[args.backbone](pretrained=False)
        model.load_state_dict(torch.load('./pretrain_weight/resnet18-5c106cde.pth'))
        model.fc = nn.Linear(model.fc.in_features, len(classes))
#         device  = torch.device('cuda:1')
#         model = nn.DataParallel(model,device_ids = [0,1])
        model.to(device)
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
    train_dataset = DS(train_images, classes=classes)
    valid_dataset = DS(valid_images, classes=classes, use_albu=False)
    test_dataset = DS(test_images, classes=classes, use_albu=False)
    
######## Hand it over to functions ########
    def objective(trial,device=device, classes=classes,
                  model=model,
                  train_dataset=train_dataset,valid_dataset=valid_dataset,test_dataset=test_dataset,
                  path_allresult=path_allresult,num_workers=num_workers,save_best=save_best):
        
        ############  Selection Hyper Parametrs  ############
        param_experiment={}
        step=1e-4
        lr = trial.suggest_float("lr", 1e-4, 1e-2, step=step)
        lr=round(lr, len(str(step)))
        param_experiment['lr']=lr
        batch_size = trial.suggest_int("batch_size", 32, 256, step=16)
        param_experiment['batch_size']=batch_size
        epochs= trial.suggest_int("epochs", 1, 1ß, step=1)
        param_experiment['epochs']=epochs
        for key, value in param_experiment.items():
            print(str(key)+"="+str(value))
        
        ############  Create fold for expiremets  ############
        def id_generator(size=8, chars=string.ascii_lowercase + string.digits):
            return ''.join(random.choice(chars) for _ in range(size))
        str_params='_btach='+str(batch_size)+'_lr='+str(lr)+'_epochs='+str(epochs)
        path_expirement=os.path.join(path_allresult,id_generator()+str_params)
        os.mkdir(path_expirement)
        
        ############  Write Hyper-parametrs in csv file   ############        
        df_experiment = pd.DataFrame()
        df_experiment = df_experiment.append(param_experiment, ignore_index=True)
        df_experiment.to_csv(os.path.join(path_expirement,'param_expirement.csv'), sep='\t')
        del df_experiment
        del param_experiment

        ######## Create Dataloader ########
        print('num_workers',args.num_workers)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=8, num_workers=args.num_workers, shuffle=True, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers=args.num_workers, shuffle=True, pin_memory=True)
        
        ######## Create Optimizer Criteria ########
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        STEPS_PER_EPOCH = len(train_dataloader)
        TOTAL_STEPS = (epochs+1) * STEPS_PER_EPOCH
        MAX_LRS = [p['lr'] for p in optimizer.param_groups]
        scheduler_lr = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                            max_lr = MAX_LRS,
                                            total_steps = TOTAL_STEPS)

        criteria = nn.CrossEntropyLoss()
        scaler = GradScaler()
        
        ######## Info about experemets ########
        reporter_df = pd.DataFrame()
        
############  Процесс тренировки ############
        for e in tqdm(range(epochs)):
            print('Start epoch '+str(e))
            reporter={} # For keep information loss, accuracy for each epoch
            start = time.time()
            model.train()

            avg_loss = 0
            avg_acc=0

            for imgs, labels in train_dataloader:
                optimizer.zero_grad()

                imgs = imgs.to(device)
                labels = labels.to(device)

                with autocast():
                    out = model(imgs)
                    loss = criteria(out, labels)

                avg_loss += round(loss.item(), 3)
                out = torch.argmax(out, dim=1)
                avg_acc += accuracy_score(labels.int().flatten().cpu(), out.int().flatten().cpu())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scheduler_lr.step()
                scaler.update()
            train_loss, train_acc=round(avg_loss / len(train_dataloader), 3) , round(avg_acc  / len(train_dataloader), 3)
            reporter['Epoch']=e
            reporter['train_loss']=train_loss
            reporter['train_acc']=train_acc
            

    ############  Процесс валидации ############       
            model.eval()
            valid_loss, valid_acc = valid(model, valid_dataloader, criteria, device)
            
            name_pmodel='checkpoint_'+str(e).zfill(4)+'.pth'
            reporter['Name_model']=name_pmodel
            
            end = time.time()
            reporter['valid_loss']=valid_loss
            reporter['valid_acc']=valid_acc
            reporter['time']=end - start
            
            reporter_df = reporter_df.append(reporter, ignore_index=True)
            reporter_df.to_csv(os.path.join(path_expirement,'result.csv'), sep='\t')
            for key, value in reporter.items():
                print(str(key)+"="+str(value))
            
            ############ Save model if it in best of n ############ 
            list_checkpoints=copy.deepcopy(reporter_df.sort_values('valid_loss')['Name_model'].tolist())
            if name_pmodel in list_checkpoints[:save_best]:
                torch.save(model.state_dict(), os.path.join(path_expirement,name_pmodel))
                
            ############ Delete not best checkpoints ############
            if len(list_checkpoints)>save_best:
                for del_check in list_checkpoints[save_best:]:
                    if os.path.exists(os.path.join(path_expirement,del_check)):
                        os.remove(os.path.join(path_expirement,del_check))
                        print("The file "+del_check+" has been deleted successfully")
                
            del list_checkpoints
            
            print('Epoch_finished')
            

    ############  Тестирование лучшей модели ############  
        reporter_df=reporter_df.sort_values('valid_loss')
        namebest_model=reporter_df.iloc[0]['Name_model']
        print('namebest_model',namebest_model)
        model.load_state_dict(torch.load(os.path.join(path_expirement,namebest_model)))
        
        test_loss, test_acc = valid(model, test_dataloader, criteria, device)
        test_df = pd.DataFrame()
        reporter_test={}
        reporter_test['Name best model by value_loss']=namebest_model
        reporter_test['test_loss']=test_loss
        reporter_test['test_acc']=test_acc
        test_df = test_df.append(reporter_test, ignore_index=True)
        test_df.to_csv(os.path.join(path_expirement,'result_test.csv'), sep='\t')
        print('### Test best model by valid value: ###')
        for key, value in reporter_test.items():
            print(str(key)+"="+str(value))

        del reporter_df
        del test_df
        print("Experiment finished, all epochs = {epochs} have done .".format(epochs))
        
        gc.collect()
                               
        return valid_loss
    
    study = optuna.create_study(study_name="pytorch_checkpoint",direction="minimize") # "maximize"
    study.optimize(objective, n_trials=30)