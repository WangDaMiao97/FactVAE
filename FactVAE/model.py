# -*- coding: utf-8 -*-
'''
@Time    : 2023/11/5 15:18
@Author  : Linjie Wang
@FileName: model.py
@Software: PyCharm
'''
import torch.nn as nn
import torch.nn.init as init
from FactVAE.layers import Encoder, ZINBDataDecoder, NBDataDecoder
from FactVAE.loss import log_zinb_positive, log_nb_positive
import torch.nn.functional as F
import torch

def xavier_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)

def initialize_parameter(model):
    for name, param in model.named_parameters():
        if isinstance(param, nn.Parameter):
            init.xavier_normal_(param.data)

class VAE(nn.Module):
    def __init__(self, layer_rna_e, layer_atac_e, zdim=None, rna_enc_dropout=0.0, atac_enc_dropout=0.0, type_d_rna=None,
                 type_d_atac=None, gene_dim=None, peak_dim=None, prior=None, beta=1, device = None):

        super(VAE, self).__init__()
        self.Type_RNA = type_d_rna
        self.Type_ATAC = type_d_atac
        self.device = device
        self.labels = None

        self.RNA_encoder = Encoder(layer_rna_e, zdim, rna_enc_dropout)
        self.ATAC_encoder = Encoder(layer_atac_e, zdim, atac_enc_dropout)
        if self.Type_RNA == 'ZINB':
            self.decoder_RNA = ZINBDataDecoder(zdim, gene_dim)
        elif self.Type_RNA == 'NB':
            self.decoder_RNA = NBDataDecoder(zdim, gene_dim)

        if self.Type_ATAC == 'ZINB':
            self.decoder_ATAC = ZINBDataDecoder(zdim, peak_dim)
        elif self.Type_ATAC == 'NB':
            self.decoder_ATAC = NBDataDecoder(zdim, peak_dim)

        # Initialization by Xavier
        self.apply(xavier_init)
        initialize_parameter(self.decoder_ATAC)
        initialize_parameter(self.decoder_RNA)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # Default parameters of CLIP

        self.prior = prior
        self.beta = beta
        self.mask = torch.nonzero(prior)
        self.masked_prior = self.prior[self.mask[:, 0], self.mask[:, 1]]

    def get_emb(self, rna_inputs, atac_inputs):
        rna_mu, rna_logvar, rna_codes = self.RNA_encoder(rna_inputs)
        atac_mu, atac_logvar, atac_codes = self.ATAC_encoder(atac_inputs)

        return rna_codes, atac_codes

    def get_G2P(self):
        gene_normalized = torch.div(self.decoder_RNA.feature.T, self.decoder_RNA.feature.T.norm(dim=1, keepdim=True))
        peak_normalized = torch.div(self.decoder_ATAC.feature.T, self.decoder_ATAC.feature.T.norm(dim=1, keepdim=True))
        sim_matrix = torch.matmul(gene_normalized, peak_normalized.T)
        return sim_matrix

    def forward(self, rna_inputs=None, rna_X=None, rna_scale_factor=None, rna_beta=None,rna_sigma=0.001,
                atac_inputs=None, atac_X=None, atac_scale_factor=None, atac_beta=None, atac_sigma=0.00001,
                atac_mask = None):
        rna_beta = self.beta if rna_beta is None else rna_beta
        atac_beta = self.beta if atac_beta is None else atac_beta

        rna_mu, rna_logvar, rna_codes = self.RNA_encoder(rna_inputs)
        atac_mu, atac_logvar, atac_codes = self.ATAC_encoder(atac_inputs)

        rna_output = self.decoder_RNA(rna_codes, rna_scale_factor)
        atac_output = self.decoder_ATAC(atac_codes, atac_scale_factor)

        rna_disper_x = rna_output["disperation"]; rna_recons = rna_output["scale_x"]; rna_norm = rna_output["normalized"]
        atac_disper_x = atac_output["disperation"]; atac_recons = atac_output["scale_x"]; atac_norm = atac_output["normalized"]

        if self.Type_RNA == 'ZINB':
            rna_dropout_rate = rna_output["dropoutrate"]
            rna_mse = log_zinb_positive(rna_X, rna_recons, rna_disper_x, rna_dropout_rate).mean()
        elif self.Type_RNA == 'NB':
            rna_mse = log_nb_positive(rna_X, rna_recons, rna_disper_x).mean()
        rna_kl = torch.exp(rna_logvar) + rna_mu.pow(2) - rna_logvar - 1  # KL divergence between the embedding representation and the Gaussian distribution
        rna_kl = (0.5 * torch.sum(rna_kl, dim=1)).mean()
        rna_loss = rna_sigma * rna_mse + rna_beta * rna_kl

        # Reconstruction loss of the model
        if self.Type_ATAC == 'ZINB':
            atac_dropout_rate = atac_output["dropoutrate"]
            atac_mse = log_zinb_positive(atac_X, atac_recons, atac_disper_x, atac_dropout_rate, mask = atac_mask).mean()
        elif self.Type_ATAC == 'NB':
            atac_mse = log_nb_positive(atac_X, atac_recons, atac_disper_x).mean()
        atac_kl = torch.exp(atac_logvar) + atac_mu.pow(2) - atac_logvar - 1
        atac_kl = (0.5 * torch.sum(atac_kl, dim=1)).mean()
        atac_loss = atac_sigma * atac_mse + atac_beta * atac_kl

        return rna_loss, (rna_mse, rna_kl), (rna_codes, rna_recons),\
            atac_loss, (atac_mse, atac_kl), (atac_codes, atac_recons)

    def forward_similar_loss(self, rna_inputs=None, atac_inputs=None):
        rna_mu, rna_logvar, rna_codes = self.RNA_encoder(rna_inputs)
        atac_mu, atac_logvar, atac_codes = self.ATAC_encoder(atac_inputs)

        # Loss of higher order feature similarity between samples
        rna_norm = rna_codes / torch.norm(rna_codes, p=2, dim=-1, keepdim=True) # normalization
        rna_sim_matrix = torch.matmul(rna_norm, rna_norm.T) # cosine similarity
        atac_norm = atac_codes / torch.norm(atac_codes, p=2, dim=-1, keepdim=True)
        atac_sim_matrix = torch.matmul(atac_norm, atac_norm.T)

        loss_similar = 1 - F.cosine_similarity(rna_sim_matrix, atac_sim_matrix, dim=1)
        return loss_similar.mean()

    def forward_prior_loss(self, eps=1e-8):
        gene_normalized = torch.div(self.decoder_RNA.feature.T, self.decoder_RNA.feature.T.norm(dim=1, keepdim=True))
        peak_normalized = torch.div(self.decoder_ATAC.feature.T, self.decoder_ATAC.feature.T.norm(dim=1, keepdim=True))
        sim_matrix = torch.matmul(gene_normalized, peak_normalized.T)

        # Calculate MSE loss only for non-zero elements
        loss_prior = F.mse_loss(self.masked_prior, sim_matrix[self.mask[:,0], self.mask[:,1]])
        return loss_prior.mean()
