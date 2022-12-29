# -*- coding: utf-8 -*-
import argparse
def the_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", default=50, type=int, help="Desired number of epochs."
    )
    parser.add_argument(
        "--dropout", type=str, default="False", help="During training, whether to use dropout or not."
    )
    parser.add_argument(
        "--uncertainty", type=str, default="True", help="Use uncertainty or not."
    )
    parser.add_argument(
        "--unc_act", type=str, default="exp", choices = ['relu', 'softplus', 'exp', 'none']
    )
    parser.add_argument(
        "--unc_type", type=str,  default="log", choices = ['mse', 'digamma', 'log']
    )
    parser.add_argument(
        "--exp_id", type=str,  default="Dec26.Local",
    )
    parser.add_argument(
        "--kl_strength", type=float,  default=1.0, help = "KL regularization strength."
    )
    parser.add_argument(
        "--use_vac_reg", type=str,  default="True", help = "Whether to use the vacuity regularization or not."
    )
    parser.add_argument(
        "--seed", type=int,  default=0, help = "Seed to replicate results"
    )

    args = parser.parse_args()
    args.dropout = args.dropout.lower() == "true"
    args.uncertainty = args.uncertainty.lower() == "true"
    args.use_vac_reg = args.use_vac_reg.lower() == "true"
    
    args.exp_save_name = f"ID_{args.exp_id}_dropout_{args.dropout}"
    args.exp_save_name += f"_Unc_{args.uncertainty}_evact_{args.unc_act}_lsstype_{args.unc_type}"
    args.exp_save_name += f"_klstr_{args.kl_strength}_vacReg_{args.use_vac_reg}_sd_{args.seed}"
    print("Exp save name: ", args.exp_save_name)
    
    return args
