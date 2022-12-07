import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np

from matplotlib import pyplot as plt
from PIL import Image

from helpers import get_device, one_hot_embedding
from data_demo import dataloader_train
from train_demo import train_model
from losses import edl_mse_loss, relu_evidence
from lenet import LeNet

from utilFiles.get_args import the_args
from utilFiles.set_deterministic import make_deterministic
args = the_args()

import sys


def main():

    print("args: ", args)
    make_deterministic(2)

    num_classes = 10

    model = LeNet(args)

    criterion = edl_mse_loss


    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.005)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    device = get_device()
    model = model.to(device)

    train_model(
        args,
        model,
        dataloader_train,
        num_classes,
        criterion,
        optimizer,
        scheduler=exp_lr_scheduler,
        num_epochs=100,
        device=device,
    )

if __name__ == "__main__":
    main()
