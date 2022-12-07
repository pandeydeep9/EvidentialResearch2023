import torch
import torch.nn.functional as F
from helpers import get_device
from utilFiles.get_args import the_args

args = the_args()
def relu_evidence(y):
    return F.relu(y)

def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood

def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    
    S = torch.sum(alpha, dim=1, keepdim=True)
    kl_pos = torch.sum(torch.log(alpha - 1+1e-5) * y, dim = -1, keepdim=True)
    
    vacuity = num_classes / S.detach() #No gradients with respect to this

    if args.use_vac_reg:
        return loglikelihood  - vacuity * kl_pos
    else:
        return loglikelihood


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss

