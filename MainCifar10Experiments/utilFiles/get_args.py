import argparse
import torch
def the_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('-net', type=str,default='resnet18', help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    
    
    parser.add_argument(
        "--save_models", type=str, default="False", help="Whether to save models or not."
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
        "--exp_id", type=str,  default="Jan13.1618.",
    )
    parser.add_argument(
        "--kl_strength", type=float,  default=1.0, help = "KL regularization strength."
    )
    parser.add_argument(
        "--use_vac_reg", type = str, default="True", help='Whether to correct evidence regularization or not'
    )
    parser.add_argument(
        "--seed", type = int, default=1, help='The Seed for experiments'
    )
    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", args.device)
    args.uncertainty = args.uncertainty.lower() == "true"
    args.use_vac_reg = args.use_vac_reg.lower() == "true"
    args.save_models = args.save_models.lower() == "true"

    args.exp_save_name = f"Cifar10_{args.exp_id}_{args.net}_net_{args.b}_bs_{args.lr}_lr_200Ep"
    args.exp_save_name += f"_VacReg_{args.use_vac_reg}"
    args.exp_save_name += f"_Unc_{args.uncertainty}_evact_{args.unc_act}"
    args.exp_save_name += f"_lsstype_{args.unc_type}"
    args.exp_save_name += f"_klstr_{args.kl_strength}"
    args.exp_save_name += f"_sd_{args.seed}"

    return args
