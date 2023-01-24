import os
import csv
def save_to_csv(save_dict, save_path, start_epoch = 0):
    keys = save_dict.keys()
    # print("save path: ", save_path)
    print("save dict: ", save_dict)
    if save_dict['epoch'] == start_epoch:
        with open(save_path, 'w', newline ='') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
    with open(save_path, 'a', newline ='') as f:
        writer = csv.writer(f)
        writer.writerow(save_dict.values())
        

import numpy as np
import matplotlib.pyplot as plt


def eval_calibration(predictions, confidences, labels, M=15):
    """
    M: number of bins for confidence scores
    """
    num_Bm = np.zeros((M,), dtype=np.int32)
    accs = np.zeros((M,), dtype=np.float32)
    confs = np.zeros((M,), dtype=np.float32)
    for m in range(M):
        interval = [m / M, (m+1) / M]
        Bm = np.where((confidences > interval[0]) & (confidences <= interval[1]))[0]
        if len(Bm) > 0:
            acc_bin = np.sum(predictions[Bm] == labels[Bm]) / len(Bm)
            conf_bin = np.mean(confidences[Bm])
            # gather results
            num_Bm[m] = len(Bm)
            accs[m] = acc_bin
            confs[m] = conf_bin
    conf_intervals = np.arange(0, 1, 1/M)
    return accs, confs, num_Bm, conf_intervals

def get_ece_evid(preds, uncertainty,labels, M=15):
    accs, confs, num_Bm, conf_intervals = eval_calibration(preds, 1-uncertainty, labels, M=M)
    
    # compute ECE
    ece = np.sum(np.abs(accs - confs) * num_Bm / np.sum(num_Bm))
    return ece

def get_ece(predictions, confidences, labels, M=15):
    accs, confs, num_Bm, conf_intervals = eval_calibration(predictions, confidences, labels, M=M)
    
    # compute ECE
    ece = np.sum(np.abs(accs - confs) * num_Bm / np.sum(num_Bm))
    return ece
