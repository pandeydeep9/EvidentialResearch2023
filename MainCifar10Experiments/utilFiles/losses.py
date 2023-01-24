import torch
import torch.nn.functional as F
from utilFiles.get_args import the_args

args = the_args()

def get_device():
    device = args.device
    return device

def get_evidence(y):
    if args.unc_act == 'relu':
        return F.relu(y)
    elif args.unc_act == 'softplus':
        return F.softplus(y)
    elif args.unc_act == 'exp':
        return torch.exp(torch.clamp(y, -10, 10))
        # return torch.exp(y)
    elif args.unc_act == 'none':
        return y
    else:
        print("The evidence function is not accurate.")
        raise NotImplementedError()





def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


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

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    
    
    
    # print("KL strength: ", args.kl_strength, " kl pos: ", kl_pos.shape, alpha_cor.shape, loglikelihood.shape)

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    
    S = torch.sum(alpha, dim=1, keepdim=True)
    # vacuity = num_classes / S.detach()
    # kl_pos = torch.sum(torch.log(alpha - 1+1e-5) * y, dim = -1, keepdim=True)
    with torch.no_grad():
        vacuity = num_classes / S.detach()
    return loglikelihood + args.kl_strength * kl_div, vacuity


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    
    S = torch.sum(alpha, dim=1, keepdim=True)
    # vacuity = num_classes / S.detach()
    # kl_pos = torch.sum(torch.log(alpha - 1+1e-5) * y, dim = -1, keepdim=True)
    with torch.no_grad():
        vacuity = num_classes / S.detach()
    
    return A + args.kl_strength * kl_div, vacuity

def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = get_evidence(output)
    alpha = evidence + 1
    mse_loss_val, vacuity = mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    loss = torch.mean(mse_loss_val)
    
    if args.use_vac_reg:
        output_correct = output*target
        loss -= torch.sum(vacuity * output_correct)/output_correct.shape[0]
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = get_evidence(output)
    # print(torch.max(evidence))
    alpha = evidence + 1
    edl_loss_val, vacuity = edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step, device)
    loss = torch.mean(edl_loss_val)
    
    if args.use_vac_reg:
        output_correct = output*target
        loss -= torch.sum(vacuity * output_correct)/output_correct.shape[0]
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = get_evidence(output)
    alpha = evidence + 1
    edl_loss_val, vacuity = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    loss = torch.mean(edl_loss_val)
    
    if args.use_vac_reg:
        output_correct = output*target
        loss -= torch.sum(vacuity * output_correct)/output_correct.shape[0]
    return loss