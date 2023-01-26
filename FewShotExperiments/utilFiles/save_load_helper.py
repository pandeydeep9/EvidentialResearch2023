import csv
import os
def save_to_csv(save_path,dict_save, sp = None):
    header, values = [], []
    for k, v in dict_save.items():
        header.append(k)
        values.append(v)

    save_model_name = f"{save_path}/results.csv"
    if sp:
        save_model_name = f"{save_path}/results_{sp}.csv"
    # save_model_name = "Debug.csv"
    if dict_save['epoch'] == 1:
        with open(save_model_name, 'w') as f:
            writer_obj = csv.writer(f)
            writer_obj.writerow(header)
    with open(save_model_name, 'a') as f:
        writer_obj = csv.writer(f)
        writer_obj.writerow(values)

def save_single_dict_to_csv(args,dict,first=False):
    header, values = dict.keys(), dict.values()


    save_model_name = f"{args.break_identifier}/results.csv"
    # save_model_name = "Debug.csv"
    if first:
        with open(save_model_name, 'w') as f:
            writer_obj = csv.writer(f)
            writer_obj.writerow(header)
    with open(save_model_name, 'a') as f:
        writer_obj = csv.writer(f)
        writer_obj.writerow(values)