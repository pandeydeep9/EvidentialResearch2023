import torch
import torch.nn as nn
import copy
import time
from helpers import get_device, one_hot_embedding
from losses import relu_evidence

from utilFiles.helperFunctions import save_to_csv

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
    uncertainty=False,
):

    since = time.time()

    if not device:
        device = get_device()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    save_dict = {"loss_tr": [],"loss_val":[], "epoch": [], "accuracy_tr": [], 
                 "accuracy_val": [],  }
    # evidences = {"evidence": [], "type": [], "epoch": []}

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        save_dict["epoch"].append(epoch)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                phase_id = "tr"
                print("Training...")
                model.train()  # Set model to training mode
            else:
                phase_id = "val"
                print("Validating...")
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            correct = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):

                    if uncertainty:
                        y = one_hot_embedding(labels, num_classes)
                        y = y.to(device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(
                            outputs, y.float(), epoch, num_classes, 10, device
                        )                 

                    else:
                        # print("Model: ", model)
                        # print("Inputs: ", inputs)
                        outputs = model(inputs)
                        # print("Outputs: ", outputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # if scheduler is not None:
            #     if phase == "train":
            #         scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            save_dict[f"loss_{phase_id}"].append(epoch_loss)
            
            save_dict[f"accuracy_{phase_id}"].append(epoch_acc.item())


            print(
                "{} loss: {:.4f} acc: {:.4f}".format(
                    phase.capitalize(), epoch_loss, epoch_acc
                )
            )

            # Savethe model every 10 epochs
            # if epoch%10==0 and epoch > 1 and phase == 'val':
            #         state = {
            #                 "epoch": num_epochs,
            #                 "model_state_dict": model.state_dict(),
            #                 "optimizer_state_dict": optimizer.state_dict(),
            #             }
            #         torch.save(state, f"{args.exp_save_name}/models/model_{epoch}.pt")
            #         print(f"Saved model at epoch: {epoch}")
            
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    save_to_csv(save_dict, args)

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    #Save Best Validation model
    state = {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
    torch.save(state, f"{args.exp_save_name}/models/model_best.pt")
    print("Saved Best model")
    
    return 
