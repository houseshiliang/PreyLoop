import numpy as np
from sklearn.neighbors import KernelDensity
import scipy.stats as st
from sklearn.neighbors import NearestNeighbors
import torch
import torchvision.transforms as transforms

def get_transform(dataset_name, train=False):
    # idea : given name, return the final implememnt transforms for the dataset
    if dataset_name.lower() == "utkface":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.Resize((64, 64)),
            transforms.RandomCrop((64, 64), padding=4) if train else None,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return transform

print(get_transform('utkface'))
