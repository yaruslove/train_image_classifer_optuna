import os
import random
from typing import Tuple
import numpy as np
import glob


import torch
from torch import Tensor
from torch.utils.data import Dataset

import torchvision

from PIL import Image

SEED = 1236

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class DS(Dataset):
    def __init__(self, images: list, classes: list, resolush: int, use_albu: bool = False):
        self.images = images
        self.classes = classes
        self.use_albu = use_albu
        self.resolush = resolush

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img = Image.open(self.images[idx]).convert('RGB')

        if self.use_albu:
            img = torchvision.transforms.RandomGrayscale(p=0.5)(img)
            img = torchvision.transforms.RandomHorizontalFlip(p=0.5)(img)
            img = torchvision.transforms.RandomVerticalFlip(p=0.5)(img)
            img = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0, hue=0)(img)
            sluch=(random.randint(0, 7))
            if sluch==0:
                img =torchvision.transforms.transforms.Pad(15,fill=1, padding_mode='edge')(img)
            elif sluch==1:
                img =torchvision.transforms.RandomPerspective(distortion_scale=0.4)(img)
            elif sluch==0:
                img = torchvision.transforms.RandomAffine(degrees=45, scale=(0.6, 0.9), shear=30)(img)
            elif sluch==3:
                img =torchvision.transforms.transforms.Pad(20,fill=1, padding_mode='symmetric')(img)
            elif sluch==4:
                img =torchvision.transforms.RandomRotation(40, expand=True, center=None, fill=0, resample=None)(img)
            elif sluch==5:
                img =torchvision.transforms.transforms.Pad(12,fill=1, padding_mode='reflect')(img)
            elif sluch==6:
                img =torchvision.transforms.GaussianBlur(3, sigma=(0.1, 2))(img)
            elif sluch==7:
                img =torchvision.transforms.RandomRotation(degrees=(0, 70), expand=False)(img)

        img = torchvision.transforms.Resize((self.resolush, self.resolush))(img)
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(img)

        c = os.path.dirname(self.images[idx]).split('/')[-2]

        return img, self.classes.index(c)

    def __len__(self) -> int:
        return len(self.images)