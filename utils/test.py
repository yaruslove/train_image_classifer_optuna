import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, accuracy_score, precision_score


def test(model: nn.Module,
         dataloader: DataLoader,
         device: str) -> dict:

    avg_prec = 0
    avg_acc = 0
    avg_f1 = 0

    for i, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)

        with torch.no_grad():
            out = model(imgs)

        out = torch.argmax(out, dim=1)

        avg_prec += precision_score(labels.int().flatten().cpu(), out.int().flatten().cpu(), average='macro')
        avg_acc += accuracy_score(labels.int().flatten().cpu(), out.int().flatten().cpu())
        avg_f1 += f1_score(labels.int().flatten().cpu(), out.int().flatten().cpu(), average='macro')

    avg_prec /= len(dataloader)
    avg_acc /= len(dataloader)
    avg_f1 /= len(dataloader)

    return {'avg_prec': avg_prec, 'avg_acc': avg_acc, 'avg_f1': avg_f1}
