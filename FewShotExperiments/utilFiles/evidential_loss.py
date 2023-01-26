
import torch
import torch.distributions as dist
from torch.distributions import Categorical

def KL_flat_dirichlet(alpha, reversed = False, device = None):
    """
    Calculate Kl divergence between a flat/uniform dirichlet distribution and a passed dirichlet distribution
    i.e. KL(dist1||dist2)
    distribution is a flat dirichlet distribution
    :param alpha: The parameters of dist2 (2nd distribution)
    :return: KL divergence
    """
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = alpha.shape[1]
    beta = torch.ones(alpha.shape, dtype=torch.float32, device=device)

    dist1 = dist.Dirichlet(beta)
    dist2 = dist.Dirichlet(alpha)

    if not reversed:
        kl = dist.kl_divergence(dist2, dist1).reshape(-1, 1)
    else:
        kl = dist.kl_divergence(dist1, dist2).reshape(-1, 1)

    return kl

def calculate_vacuity(evidence):
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    num_classes = evidence.shape[1]
    # print("num classes", num_classes)
    vacuity = num_classes/S
    return vacuity.detach()
    # print("vacuity: ", vacuity)

def evidential_loss_type2(evidence, y, epoch=1,evid_reg_way = 'kl', ev_reg_strength = 1):
    '''
    Args:
        evidence: BS*Num Classes
        y: One hot BS * Num Classes
        epoch:
        evid_reg_way: how to regularize the wrong evidence
        ev_reg_strength:

    Returns:

    '''
    alpha = evidence + 1
    # print("alpha: ", alpha, alpha.shape)
    S = torch.sum(alpha, dim=1, keepdim=True)
    # print("S: ", S, S.shape)
    s_alpha_diff = torch.log(S) - torch.log(alpha)
    # print("Diff: ", s_alpha_diff, s_alpha_diff.shape)
    # print("y: ", y, y.shape)
    loss = torch.mean(torch.sum(y * s_alpha_diff, dim = 1))
    # print("Loss: ", loss)

    # import sys
    # sys.exit()
    
    # kl_part = 0

    return loss

def evidential_loss_mse(evidence, y, epoch=1,evid_reg_way = 'none', ev_reg_strength = 1):
    '''
    Args:
        evidence: BS*Num Classes
        y: One hot BS * Num Classes
        epoch:
        evid_reg_way: how to regularize the wrong evidence
        ev_reg_strength:

    Returns:

    '''
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    pred = alpha / S
    # print("A: ", y.shape, pred.shape)

    A = torch.sum((y - pred) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S ** 2 * (S + 1)), dim=1, keepdim=True)

    annealing_rate = ev_reg_strength  * torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch / 10, dtype=torch.float32),
    ).to(evidence.device)

    alpha_tilda = evidence * (1 - y) + 1
    kl_part = 0.0 * evidence * (1 - y)
    if evid_reg_way == 'none':
        kl_part = 0 * KL_flat_dirichlet(alpha_tilda)
    elif evid_reg_way == 'kl':
        kl_div = KL_flat_dirichlet(alpha_tilda)
        kl_part = annealing_rate*kl_div
    elif evid_reg_way == 'rev_kl':
        kl_div = KL_flat_dirichlet(alpha_tilda, reversed=True)
        kl_part = annealing_rate * kl_div
    elif evid_reg_way == 'inc_bel':
        reg = evidence * (1-y)/S
        kl_part = annealing_rate * reg
    elif evid_reg_way == 'inc_evid':
        reg = evidence * ( 1- y)
        kl_part = annealing_rate * reg

    return torch.mean(A + B + kl_part)


def calculate_entropy(evidence):
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    pred = alpha / S
    dist = Categorical(probs=pred)
    return dist.entropy()

def calculate_dissonance_from_belief(belief):
    belief = torch.unsqueeze(belief, dim=1)

    sum_bel_mat = torch.transpose(belief, -2, -1) + belief  # a + b for all a,b in the belief
    diff_bel_mat = torch.abs(torch.transpose(belief, -2, -1) - belief)

    div_diff_sum = torch.div(diff_bel_mat, sum_bel_mat)  # |a-b|/(a+b)

    Bal_mat = 1 - div_diff_sum
    zero_matrix = torch.zeros(sum_bel_mat.shape, dtype=sum_bel_mat.dtype).to(sum_bel_mat.device)
    Bal_mat[sum_bel_mat == zero_matrix] = 0  # remove cases where a=b=0

    diagonal_matrix = torch.ones(Bal_mat.shape[1], Bal_mat.shape[2]).to(sum_bel_mat.device)
    diagonal_matrix.fill_diagonal_(0)  # remove j != k
    Bal_mat = Bal_mat * diagonal_matrix  # The balance matrix

    belief = torch.einsum('bij->bj', belief)
    sum_bel_bal_prod = torch.einsum('bi,bij->bj', belief, Bal_mat)
    sum_belief = torch.sum(belief, dim=1, keepdim=True)
    divisor_belief = sum_belief - belief
    scale_belief = belief / divisor_belief
    scale_belief[divisor_belief == 0] = 1

    each_dis = torch.einsum('bi,bi->b', scale_belief, sum_bel_bal_prod)

    return each_dis

def calculate_dissonance(evidence):
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    belief = alpha / S
    # print("belief: ", belief.shape)
    dissonance = calculate_dissonance_from_belief(belief)
    return dissonance