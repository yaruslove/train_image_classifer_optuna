import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import  accuracy_score
import random
import numpy as np


SEED = 1236

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def train(model: nn.Module,
          dataloader: DataLoader,
          criteria: nn.Module,
          optim: Optimizer,
          scaler: GradScaler,
          device: str,
          scheduler:lr_scheduler ) -> float:

    avg_loss = 0
    avg_acc=0

    for imgs, labels in dataloader:
        optim.zero_grad()

        imgs = imgs.to(device)
        labels = labels.to(device)

        with autocast():
            out = model(imgs)
            loss = criteria(out, labels)
            

        avg_loss += round(loss.item(), 3)
        # print('labels',labels.int().flatten().cpu())
        # print('out',out.int().flatten().cpu())
        out = torch.argmax(out, dim=1)
        avg_acc += accuracy_score(labels.int().flatten().cpu(), out.int().flatten().cpu())

        scaler.scale(loss).backward()
        scaler.step(optim)
        scheduler.step()
        scaler.update()
        

    return round(avg_loss / len(dataloader), 3) , round(avg_acc  / len(dataloader), 3)
