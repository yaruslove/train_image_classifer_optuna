import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import  accuracy_score


def valid(model: nn.Module,
         dataloader: DataLoader,
         criteria: nn.Module,
         device: str) -> float:

    valid_loss=[]

    label_true=[]
    label_pred=[]
    model.eval()

    for i, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(imgs)
            loss = criteria(out, labels)

        out = torch.argmax(out, dim=1)
        
        valid_loss=valid_loss+[round(loss.item(), 3)]*len(labels)

        label_true=label_true+labels.int().flatten().cpu().tolist()
        label_pred=label_pred+out.int().flatten().cpu().tolist()
        
    valid_loss=sum(valid_loss) / len(valid_loss)
    valid_acc=accuracy_score(label_true, label_pred)
    
    return valid_loss, valid_acc
