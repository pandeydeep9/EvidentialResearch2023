# train.py
#!/usr/bin/env	python3

""" train network using pytorch
"""

import os
import sys

from utilFiles.get_args import the_args
args = the_args()

import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F
from torch.utils.data import DataLoader

from utilFiles.set_deterministic import make_deterministic

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
def train(net, epoch, save_dict):

    start = time.time()
    net.train()
    train_loss, count = 0.0, 0
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.to(device)
            images = images.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()
        count += 1

       
        train_loss += loss.item()

        if epoch <= args.warm:
            warmup_scheduler.step()

    train_loss /= count
    save_dict['train_loss'] = train_loss

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    return save_dict

@torch.no_grad()
def eval_training(epoch=0, save_dict={}):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.to(device)
            labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()

    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    save_dict['test_average_loss'] = test_loss / len(cifar100_test_loader.dataset)
    save_dict['test_accuracy'] = (correct.float() / len(cifar100_test_loader.dataset)).item()


    return save_dict

import csv
def save_to_csv(save_dict, save_path):
    keys = save_dict.keys()
    # print("save path: ", save_path)
    # print("save dict: ", save_dict)
    if save_dict['epoch'] == 1:
        with open(save_path, 'w', newline ='') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
    with open(save_path, 'a', newline ='') as f:
        writer = csv.writer(f)
        writer.writerow(save_dict.values())
        


import time
import shutil
if __name__ == '__main__':
    
    make_deterministic(args.seed)
    
    net = get_network(args).to(device)


    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=1,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=1,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

   
    checkpoint_path = "CE_" + args.exp_save_name
    
    if os.path.exists(checkpoint_path):
        print("File exists. Removing: ", checkpoint_path)
        time.sleep(10)
        shutil.rmtree(checkpoint_path)

    os.mkdir(checkpoint_path)


    
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.to(device)

    save_results_path = checkpoint_path + "/save_results.csv"

    best_acc = 0.0
    
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        save_dict = {}
        save_dict['epoch'] = epoch
        save_dict = train(net, epoch, save_dict)
        save_dict = eval_training(epoch, save_dict)
        acc = save_dict['test_accuracy']
        
        if epoch %100 == 0:
            weights_path = os.path.join(checkpoint_path, f'{args.net}-{epoch}-best.pth')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
        
        save_to_csv(save_dict, save_results_path)

    
