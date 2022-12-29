# -*- coding: utf-8 -*-
import os
import sys
import shutil
import time
def ensure_directory(args):
        
    if os.path.exists(args.exp_save_name):
        print("will remove the directory: ", args.exp_save_name)
        time.sleep(10)
        shutil.rmtree(args.exp_save_name)
    os.mkdir(args.exp_save_name)
    os.mkdir(args.exp_save_name + "/models")
    
    args_val = str(args)
    print("To save: ", args_val)
    path = args.exp_save_name + "/the_args.txt"
    with open(path,"w") as p:
        p.write(args_val)
    
    