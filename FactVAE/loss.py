# -*- coding: utf-8 -*-
'''
@Time    : 2023/11/5 15:18
@Author  : Linjie Wang
@FileName: loss.py
@Software: PyCharm
'''

import torch
from torch.nn import functional as F

def log_zinb_positive(x, mu, theta, pi, mask=None, eps=1e-8):
    x = x.float()

    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)
    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (-softplus_pi + pi_theta_log
                     + x * (torch.log(mu + eps) - log_theta_mu_eps)
                     + torch.lgamma(x + theta)
                     - torch.lgamma(theta)
                     - torch.lgamma(x + 1))

    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)
    res = mul_case_zero + mul_case_non_zero
    if mask is not None:
        res = res[:, mask]
    return - torch.sum(res, dim=1)


def log_nb_positive(x, mu, theta, eps=1e-8):
    x = x.float()

    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
    )

    return - torch.sum(res, dim=1)
