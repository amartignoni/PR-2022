import numpy as np
import torch
import torch.nn as nn


def load_train():
    path = "./data/train/"
    train_ds = torchvision.datasets.ImageFolder(
        root = path,
        transform = torchvision.transforms.ToTensor()
    )
