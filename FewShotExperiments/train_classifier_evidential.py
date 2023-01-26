import argparse
import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler


import sys
from utilFiles.losses import edl_log_loss
from utilFiles.helper_functions import save_to_csv
from utilFiles.set_deterministic import make_deterministic

from utilFiles.losses import edl_log_loss
def main(config):
    
    
    args.exp_save_name += '_{}'.format(config['train_dataset'])
    args.exp_save_name += '_' + config['model_args']['encoder']
    # clsfr = config['model_args']['classifier']
    # if clsfr != 'linear-classifier':
    #     args.exp_save_name += '-' + clsfr
    # if args.tag is not None:
    #     args.exp_save_name += '_' + args.tag
    
    svname = args.exp_save_name

    save_path = os.path.join('./', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))


    print("Works till here")
    # print("What: ", **config['train_dataset_args'])
    #### Dataset ####

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True,
                              num_workers=8, pin_memory=True)
    utils.log('train dataset: {} (x{}), {}'.format(
            train_dataset[0][0].shape, len(train_dataset),
            train_dataset.n_classes))

    print("2. Train Dataloader Done ")
    # val
    if config.get('val_dataset'):
        eval_val = True
        val_dataset = datasets.make(config['val_dataset'],
                                    **config['val_dataset_args'])
        val_loader = DataLoader(val_dataset, config['batch_size'],
                                num_workers=8, pin_memory=True)
        utils.log('val dataset: {} (x{}), {}'.format(
                val_dataset[0][0].shape, len(val_dataset),
                val_dataset.n_classes))
    else:
        eval_val = False

    print("3. Validation Dataloader Done ")

    # few-shot eval
    if config.get('fs_dataset'):
        ef_epoch = config.get('eval_fs_epoch')
        if ef_epoch is None:
            ef_epoch = 5
        eval_fs = True

        fs_dataset = datasets.make(config['fs_dataset'],
                                   **config['fs_dataset_args'])
        utils.log('fs dataset: {} (x{}), {}'.format(
                fs_dataset[0][0].shape, len(fs_dataset),
                fs_dataset.n_classes))

        print("4. Basic standard setting test load done")

        n_way = 5
        n_query = 15
        n_shots = [1, 5]
        fs_loaders = []
        for n_shot in n_shots:
            fs_sampler = CategoriesSampler(
                    fs_dataset.label, 200,
                    n_way, n_shot + n_query, ep_per_batch=4)
            fs_loader = DataLoader(fs_dataset, batch_sampler=fs_sampler,
                                   num_workers=8, pin_memory=True)
            fs_loaders.append(fs_loader)
    else:
        eval_fs = False

    ########
    print("5. Roughly done, fssampler -> May need to better structure")

    #### Model and Optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

    if eval_fs:
        fs_model = models.make('meta-baseline', encoder=None)
        fs_model.encoder = model.encoder

    if config.get('_parallel'): #TODO: SKIPPED FOR NOW
        model = nn.DataParallel(model)
        if eval_fs:
            fs_model = nn.DataParallel(fs_model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    print("6. Model ready now")

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])

    print("7. Optimizer and Scheduler done.")

    ########

    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0. #TODO: ?
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    save_dict = {}

    for epoch in range(1, max_epoch + 1 + 1):
        
        if epoch == max_epoch + 1:
            if not config.get('epoch_ex'):
                break
            train_dataset.transform = train_dataset.default_transform
            train_loader = DataLoader(
                    train_dataset, config['batch_size'], shuffle=True,
                    num_workers=8, pin_memory=True)

        timer_epoch.s()
        aves_keys = ['tl', 'ta', 'vl', 'va'] #Training loss, training accuracy, validation loss, validation accuracy
        if eval_fs:
            for n_shot in n_shots:
                aves_keys += ['fsa-' + str(n_shot)] #For each shot, accuracy
        aves = {k: utils.Averager() for k in aves_keys} #Averager is a cool function, well thought out

        # train
        model.train()
        save_dict['epoch'] = epoch
        save_dict['lr'] = optimizer.param_groups[0]['lr']

        #train loader --> Iterable
        #desc --> Prefix the progressbar
        #leave --> if true, keep all traces of progressbar unde termination of the iteration
        for data, label in tqdm(train_loader, desc='train', leave=False):
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            # print("Evidence: ", evidence.shape, evidence)
            # print("Label: ", label)
            num_classes = outputs.shape[-1]
            one_hot_label = F.one_hot(label, num_classes)
            # print("One hot Label: ", label)
            # print("outputs: ", outputs)
            loss = edl_log_loss(
                outputs, one_hot_label.float(), epoch, num_classes, 10, device) 
            # print("Loss: ", loss)
            # sys.exit()
            # loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            #TODO: WHY? MAYBE TO RESET
            logits = None; loss = None


        # eval
        if eval_val:
            model.eval()
            for data, label in tqdm(val_loader, desc='val', leave=False):
                data, label = data.to(device), label.to(device)
                with torch.no_grad():
                    evidence = model(data)
                    one_hot_label = F.one_hot(label, num_classes = evidence.shape[-1])
                    num_classes = evidence.shape[-1]
                    one_hot_label = F.one_hot(label, num_classes)
                    loss = edl_log_loss(
                        evidence, one_hot_label.float(), epoch, num_classes, 10, device) 
                    # acc = utils.compute_acc(outputs, label)
                    acc = utils.compute_acc(evidence, label)

                aves['vl'].add(loss.item())
                aves['va'].add(acc)

        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            fs_model.eval()
            for i, n_shot in enumerate(n_shots): #1 shot and 5 shot
                np.random.seed(0)
                for data, _ in tqdm(fs_loaders[i],
                                    desc='fs-' + str(n_shot), leave=False):
                    x_shot, x_query = fs.split_shot_query(
                            data.to(device), n_way, n_shot, n_query, ep_per_batch=4)
                    label = fs.make_nk_label(
                            n_way, n_query, ep_per_batch=4).to(device)
                    with torch.no_grad():
                        logits = fs_model(x_shot, x_query).view(-1, n_way)
                        acc = utils.compute_acc(logits, label)
                    aves['fsa-' + str(n_shot)].add(acc)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

        if epoch <= max_epoch:
            epoch_str = str(epoch)
        else:
            epoch_str = 'ex'


        #Log training, validation, and few shot accuracy/losses
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
                epoch_str, aves['tl'], aves['ta'])

        save_dict['train loss'] = aves['tl']
        save_dict['train accuracy'] = aves['ta']

        if eval_val:
            log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])

            save_dict['val loss'] = aves['vl']
            save_dict['val accuracy'] = aves['va']

        if eval_fs and (epoch == 1 or epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            log_str += ', fs'
            for n_shot in n_shots:
                key = 'fsa-' + str(n_shot)
                log_str += ' {}: {:.4f}'.format(n_shot, aves[key])
                save_dict[f'{key} accuracy'] = aves[key]

        
        #Show time, time estimate
        if epoch <= max_epoch:
            log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        else:
            log_str += ', {}'.format(t_epoch)
        utils.log(log_str)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        #The training arguments needed for saving
        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }

        #The object to save
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }


        if epoch <= max_epoch:
            #Save most recent as epoch-last.pth
            torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))

            #Save best validation performing as well
            if aves['va'] > max_va:
                max_va = aves['va']
                torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
        else:
            torch.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))

        save_results_path = save_path + "/results.csv"
        save_to_csv(save_dict, save_results_path, start_epoch = 1)


from utilFiles.get_args import the_args
if __name__ == '__main__':
    args = the_args()
    device = args.device
    make_deterministic(args.seed)
    print("args: ", args)
        
    
    print("args: ", args)

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    print("Config: ", config)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)

