import csv
import pandas as pd
def save_to_csv(dict_save, args):
    keys = dict_save.keys()
    df = pd.DataFrame.from_dict(dict_save)
    print(df)
    df.to_csv(f"{args.exp_save_name}/results.csv", index=False)