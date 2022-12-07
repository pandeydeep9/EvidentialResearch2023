# -*- coding: utf-8 -*-
import argparse
def the_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--use_vac_reg", type=str,  default="False", help = "Whether to use the vacuity regularization or not.")
    parser.add_argument("--demo_ce_model", type=str,  default="True", help = "Whether to use the cross entropy model (True) or Evidential Model(False).")

    args = parser.parse_args()
    args.use_vac_reg = args.use_vac_reg.lower() == "true"
    args.demo_ce_model = args.demo_ce_model.lower() == "true"
    
    
    return args
