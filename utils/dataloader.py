import os
import random
from typing import Tuple
import glob
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

def aug_foreground(img,sz):
    img = torchvision.transforms.RandomHorizontalFlip(p=0.5)(img)
    sluch=(random.randint(0, 7))
    if sluch==0:
        img =torchvision.transforms.transforms.Pad(15,fill=1, padding_mode='edge')(img)
    elif sluch==1:
        img =torchvision.transforms.transforms.Pad(4,fill=1, padding_mode='symmetric')(img)
    elif sluch==2:
        img = torchvision.transforms.RandomVerticalFlip(p=0.8)(img)
    elif sluch==3:
        img =torchvision.transforms.transforms.Pad(20,fill=1, padding_mode='symmetric')(img)
    elif sluch==4:
        img =torchvision.transforms.transforms.Pad(12,fill=1, padding_mode='reflect')(img)
    elif sluch==5:
        img =torchvision.transforms.GaussianBlur(3, sigma=(0.1, 2))(img)
    elif sluch==6:        
        img =torchvision.transforms.GaussianBlur(5, sigma=(0.1, 6))(img)
    if sluch%2==0:
        img =torchvision.transforms.RandomRotation(30, expand=True, center=None, fill=0)(img)

    img  = img.resize(sz)
    return img

def aug_seatbelt(img):
    img =torchvision.transforms.transforms.Pad(15,fill=1, padding_mode='edge')(img)
    img =torchvision.transforms.RandomPerspective(distortion_scale=0.4, p=0.2)(img)
    img = torchvision.transforms.RandomHorizontalFlip(p=0.5)(img)
    img = torchvision.transforms.RandomVerticalFlip(p=0.4)(img)
    img =torchvision.transforms.RandomRotation(360, expand=False, center=None, fill=0,)(img)
    
    sluch=(random.randint(0, 3))
    if sluch==0:
        img =torchvision.transforms.GaussianBlur(3, sigma=(0.1, 2))(img)
    return img



class DS(Dataset):
    def __init__(self, images: list, classes: list, use_albu: bool = True):
        self.images = images
        self.classes = classes
        self.use_albu = use_albu

        path_background='/disk/inference/cook_datasets/Yaroslav/seat_belt/additional_data/mixed_background/'
        backs = glob.glob(f'{path_background}*.jpg')
        backs += glob.glob(f'{path_background}/*.png')
        backs += glob.glob(f'{path_background}/*.JPG')
        backs += glob.glob(f'{path_background}/*.JPEG')
        backs += glob.glob(f'{path_background}/*.PNG')
        self.backs=backs

        path_seatbel='/disk/inference/cook_datasets/Yaroslav/seat_belt/additional_data/cropped_clean/'
        back_seatbels = glob.glob(f'{path_seatbel}*.jpg')
        back_seatbels += glob.glob(f'{path_seatbel}/*.png')
        back_seatbels += glob.glob(f'{path_seatbel}/*.JPG')
        back_seatbels += glob.glob(f'{path_seatbel}/*.JPEG')
        back_seatbels += glob.glob(f'{path_seatbel}/*.PNG')
        self.back_seatbels=back_seatbels

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img = Image.open(self.images[idx]).convert('RGB')
#         print('Data_loader')
        if self.images[idx].startswith('@%$xy$%@'):
            sz=img.size
            img= aug_foreground(img,sz)


            tmp_back=random.choices(self.backs, k=1)
            tmp_back = Image.open(tmp_back[0])
            tmp_back = tmp_back.resize(sz)

            tmp_seatbelt=random.choices(self.back_seatbels, k=1)
            tmp_seatbelt = Image.open(tmp_seatbelt[0])
            tmp_seatbelt =aug_seatbelt(tmp_seatbelt)
            tmp_seatbelt = tmp_seatbelt.resize(sz)
            tmp_seatbelt.paste(img, (0, 0), img)
            tmp_back.paste(tmp_seatbelt, (0, 0), tmp_seatbelt)

            img=tmp_back
            img = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0, hue=0)(img)
            img = torchvision.transforms.RandomGrayscale(p=0.45)(img)



        if self.use_albu and not self.images[idx].startswith('@%$xy$%@'):
            img = torchvision.transforms.RandomGrayscale(p=0.5)(img)
            img = torchvision.transforms.RandomHorizontalFlip(p=0.5)(img)
            img = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0, hue=0)(img)
            sluch=(random.randint(0, 7))
            if sluch==0:
                img =torchvision.transforms.transforms.Pad(15,fill=1, padding_mode='edge')(img)
            elif sluch==1:
                img =torchvision.transforms.transforms.Pad(4,fill=1, padding_mode='symmetric')(img)
            elif sluch==0:
                img = torchvision.transforms.RandomVerticalFlip(p=0.8)(img)
            elif sluch==3:
                img =torchvision.transforms.transforms.Pad(20,fill=1, padding_mode='symmetric')(img)
            elif sluch==4:
                img =torchvision.transforms.RandomRotation(40, expand=True, center=None, fill=0, resample=None)(img)
            elif sluch==5:
                img =torchvision.transforms.transforms.Pad(12,fill=1, padding_mode='reflect')(img)
            elif sluch==6:
                img =torchvision.transforms.GaussianBlur(3, sigma=(0.1, 2))(img)
                

        img = torchvision.transforms.Resize((336, 336))(img)
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(img)

        c = os.path.dirname(self.images[idx]).split('/')[-2]

        return img, self.classes.index(c)

    def __len__(self) -> int:
        return len(self.images)
