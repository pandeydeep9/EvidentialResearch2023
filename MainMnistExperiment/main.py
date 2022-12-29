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

from helpers import get_device, rotate_img, one_hot_embedding
from data import dataloaders, digit_one
from train import train_model
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from lenet import LeNet

from utilFiles.get_args import the_args
from utilFiles.save_directory import ensure_directory
from utilFiles.set_deterministic import make_deterministic
args = the_args()

import sys

def main():
    
    print("args: ", args)
    make_deterministic(args.seed)
    ensure_directory(args)
    



    num_epochs = args.epochs
    use_uncertainty = args.uncertainty
    num_classes = 10

    model = LeNet(args)

    if args.uncertainty:
        if args.unc_type == "digamma":
            criterion = edl_digamma_loss
        elif args.unc_type == "log":
            criterion = edl_log_loss
        elif args.unc_type == "mse":
            criterion = edl_mse_loss
        else:
            print("--uncertainty requires --mse, --log or --digamma.")
            raise NotImplementedError
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    device = get_device()
    model = model.to(device)

    train_model(
        args,
        model,
        dataloaders,
        num_classes,
        criterion,
        optimizer,
        scheduler=exp_lr_scheduler,
        num_epochs=num_epochs,
        device=device,
        uncertainty=use_uncertainty,
    )
if __name__ == "__main__":
    main()
