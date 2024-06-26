
import torch
# import lightly
from torchvision import datasets, transforms
from transform import BarlowTwinsTransform
from config import *
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data import random_split
from natsort import natsorted
from PIL import Image
from PIL.Image import Resampling
import pytorch_lightning as pl
import numpy as np

class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        image_names = os.listdir(root_dir)

        self.root_dir = root_dir
        self.transform = transform 
        self.image_names = natsorted(image_names)

    def __len__(self): 
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img.thumbnail((84, 84), Resampling.LANCZOS)
        if self.transform:
            img = self.transform(img)

        return img


def normalization():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    return normalize

def grayScale(x):
    img = x.convert("L")
    img = img.convert("RGB")
    return img

train_transform = BarlowTwinsTransform(
    train=True, input_height=64, gaussian_blur=False, jitter_strength=0.5, normalize=normalization()
)

train_dataset = CelebADataset(root_dir=data_dir_train, transform=train_transform)
train_len = int(len(train_dataset)*0.9)
valid_len = len(train_dataset) - train_len
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_len, valid_len], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)


