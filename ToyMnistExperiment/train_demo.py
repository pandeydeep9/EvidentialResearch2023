import torch
import torch.nn as nn
import copy
import time

import numpy as np
np.set_printoptions(suppress=True)

from helpers import get_device, one_hot_embedding
from losses import relu_evidence


import torchvision

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 22})
def train_model(
    args,
    model,
    dataloaders,
    num_classes,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
    device=None,
):

    since = time.time()

    if not device:
        device = get_device()



    (inputs, labels) = iter(dataloaders).next()
    
    show_training_data = False #Set to True to see the training data
    if show_training_data:

        fig, ax = plt.subplots(1, 4)
        fig.set_figheight(4)
        fig.set_figwidth(18)
        for index, (im, lab) in enumerate(zip(inputs, labels)):
            ax[index].imshow(im[0], cmap = 'gray')
            ax[index].set_xticks([])
            ax[index].set_yticks([])
            ax[index].grid(False)
    
            ax[index].set_title(f"GT: {lab.item()}", color = 'blue')
            
        
        plt.tight_layout()
        plt.savefig("trainingData.png", dpi = 120)
        plt.show()

    inputs = inputs.to(device)
    labels = labels.to(device)


    print("label: ", labels)
    
    epoch_range = [x for x in range(100)]
    accuracy_range = []
    loss_range = []
    
    #Set to true if running CE model. Set to false if using evidential model
    # demo_ce_model = True
    demo_ce_model = args.demo_ce_model
    for epoch in epoch_range:
        print("Epoch {}/{}".format(epoch, epoch_range[-1]))
        print("-" * 10)

        phase_id = "tr"
        print("Training...")
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0.0
        correct = 0


        num_iters = 10
        for _ in range(num_iters):


            # zero the parameter gradients
            optimizer.zero_grad()

            # track history if only in train
            with torch.set_grad_enabled(True):
                y = one_hot_embedding(labels, num_classes)
                y = y.to(device)
                outputs = model(inputs)
                # print("GT: ", labels.detach().cpu().numpy(), " Outputs: ", outputs.detach().cpu().numpy())

                _, preds = torch.max(outputs, 1)
                
                #1. Use this if using cross entropy 
                if demo_ce_model:
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                else: #2. Use this if using Evidential Loss
                    loss = criterion(
                        outputs, y.float(), epoch, num_classes, 10, device
                    )
                # print("loss: ", loss)
                
                if not epoch == 0: #Don't train on first epoch
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # print("running loss:: ", running_loss)

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / num_iters/4
        epoch_acc = running_corrects.double() / num_iters/4
        
        accuracy_range.append(epoch_acc.detach().cpu().numpy().item())
        loss_range.append(epoch_loss)

        print("Epochs loss: ", epoch_loss)



        print(
            "{} loss: {:.4f} acc: {:.4f}".format(
                "One sample", epoch_loss, epoch_acc
            )
        )




    df = pd.DataFrame(list(zip(epoch_range, accuracy_range, loss_range)), columns = ['Epoch', 'Accuracy', 'Loss'])
    if demo_ce_model:
        df.to_csv("CEaccuracyTrend.csv")
    else:
        df.to_csv(f"EvidaccuracyTrend_{args.use_vac_reg}.csv")
    
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )



    return
