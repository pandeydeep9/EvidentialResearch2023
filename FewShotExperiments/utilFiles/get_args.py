import argparse
import torch
def the_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type = int, default=1)
    parser.add_argument('--exp_id', type = str, default = 'ID', help = 'identifier')
    
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
        "--kl_strength", type=float,  default=0.1, help = "KL regularization strength."
    )
    parser.add_argument(
        "--use_vac_reg", type = str, default="False", help='Whether to correct evidence regularization or not'
    )
    
    args = parser.parse_args()
    
    
    args.uncertainty = args.uncertainty.lower() == "true"
    args.use_vac_reg = args.use_vac_reg.lower() == "true"
    
    args.exp_save_name = f"Evid_{args.exp_id}_"
    args.exp_save_name += f"_VacReg_{args.use_vac_reg}"
    args.exp_save_name += f"_Unc_{args.uncertainty}_uncact_{args.unc_act}"
    args.exp_save_name += f"_lsstype_{args.unc_type}"
    args.exp_save_name += f"_klstr_{args.kl_strength}"
    args.exp_save_name += f"_sd_{args.seed}"
    
    args.exp_save_name += f'_sd_{args.seed}'
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return args
