# -*- coding: utf-8 -*-
'''
@Time    : 2023/11/5 14:16
@Author  : Linjie Wang
@FileName: layers.py
@Software: PyCharm
'''

import torch
import torch.nn as nn
import collections
from torch.nn import functional as F
from torch.autograd import Variable

def build_multi_layers(layers, dropout_rate=0.0):
    """Build multilayer linear perceptron"""

    if dropout_rate>0.0:
        fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in, n_out),
                            nn.ELU(),
                            nn.Dropout(p=dropout_rate),
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:]))
                ]
            )
        )
    else:
        fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in, n_out),
                            nn.ELU(),
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:]))
                ]
            )
        )

    return fc_layers

class Encoder(nn.Module):

    def __init__(self, layer, Z_DIMS, droprate=0.0):
        super(Encoder, self).__init__()

        if len(layer) > 1:
            self.fc1 = build_multi_layers(layers=layer, dropout_rate=droprate)

        self.layer = layer
        self.fc_means = nn.Linear(layer[-1], Z_DIMS)
        self.fc_logvar = nn.Linear(layer[-1], Z_DIMS)

    def reparametrize(self, means, logvar):

        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(means)
        else:
            return means

    def return_all_params(self, x):

        if len(self.layer) > 1:
            h = self.fc1(x)
        else:
            h = x

        mean_x = self.fc_means(h)
        logvar_x = self.fc_logvar(h)
        latent = self.reparametrize(mean_x, logvar_x)

        return mean_x, logvar_x, latent

    def forward(self, x):

        mean_x, logvar_x, latent = self.return_all_params(x)

        return mean_x, logvar_x, latent

class NBDataDecoder(nn.Module):

    def __init__(
            self, in_features: int, out_features: int
    ) -> None:
        super(NBDataDecoder, self).__init__()

        # factor matrix of features
        self.feature = nn.Parameter(torch.empty(in_features, out_features), requires_grad=True)
        self.disper = nn.Parameter(torch.empty(in_features, out_features), requires_grad=True)

    def forward(self, x: torch.Tensor, scale_factor=torch.tensor(1.0)):

        normalized_x = F.softmax(torch.matmul(x, self.feature), dim=1) # softmax(Z × Feature)

        batch_size = normalized_x.size(0)
        scale_factor.resize_(batch_size, 1)
        scale_factor.repeat(1, normalized_x.size(1))

        scale_x = torch.exp(scale_factor) * normalized_x
        disper_x = torch.exp(F.softmax(torch.matmul(x, self.disper), dim=1))  ### theta

        return dict(normalized=normalized_x,
                    disperation=disper_x,
                    scale_x=scale_x)


class ZINBDataDecoder(nn.Module):

    def __init__(
            self, in_features: int, out_features: int
    ) -> None:
        super(ZINBDataDecoder, self).__init__()

        # factor matrix of features
        self.feature = nn.Parameter(torch.empty(in_features, out_features), requires_grad=True)
        self.disper = nn.Parameter(torch.empty(in_features, out_features), requires_grad=True)
        self.dropout = nn.Parameter(torch.empty(in_features, out_features), requires_grad=True)

    def forward(self, x: torch.Tensor, scale_factor=torch.tensor(1.0)):
        normalized_x = F.softmax(torch.matmul(x, self.feature), dim=1)  # softmax(Z × Feature)

        batch_size = normalized_x.size(0)
        scale_factor.resize_(batch_size, 1)
        scale_factor.repeat(1, normalized_x.size(1))

        scale_x = torch.exp(scale_factor) * normalized_x
        disper_x = torch.exp(F.softmax(torch.matmul(x, self.disper), dim=1))  ### theta
        dropout_rate = torch.matmul(x, self.dropout) # Calculation of the probability dropout for 0 values

        return dict(normalized=normalized_x,
                    disperation=disper_x,
                    dropoutrate=dropout_rate,
                    scale_x=scale_x)