import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from pathlib import Path
import numpy as np
import pandas as pd
import anndata
import random

from FactVAE.model import VAE
from FactVAE.loader import dataInstance

import matplotlib.pyplot as plt

import argparse
import pathlib
import time
import yaml
import sys
import copy
import logging

import scglue
from scglue.utils import config
from scipy.sparse import load_npz

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

def parse_args(dataset) -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",type=str, default=dataset,
        help="The name of input dataset."
    )
    parser.add_argument(
        "--input-rna", dest="input_rna", type=pathlib.Path, default="dataset/{}/{}-RNA.h5ad".format(dataset, dataset),
        help="Path to input RNA dataset (.h5ad)"
    )
    parser.add_argument(
        "--input-atac", dest="input_atac", type=pathlib.Path, default="dataset/{}/{}-ATAC.h5ad".format(dataset, dataset),
        help="Path to input ATAC dataset (.h5ad)"
    )
    parser.add_argument(
        "-p", "--prior", dest="prior", type=pathlib.Path, default="dataset/{}/gene_to_peak.npz".format(dataset),
        help="Path to prior graph (.npz)"
    )
    parser.add_argument(
        "-r", "--run-info", dest="run_info", type=pathlib.Path, default="Result/FactVAE/{}/FactVAE-run-info.yaml".format(dataset),
        help="Path of output run info file (.yaml)"
    )
    parser.add_argument(
        "--max-epoch", dest="max_epoch", type=int, default=300, help="max epoches"
    )
    return parser.parse_args()

def get_modalities(adatas, latent_dim: int = 50):
    modalities, all_ct= {}, set()
    for k, adata in adatas.items():
        if config.ANNDATA_KEY not in adata.uns:
            raise ValueError(
                f"The '{k}' dataset has not been configured. "
                f"Please call `configure_dataset` first!"
            )
        data_config = copy.deepcopy(adata.uns[config.ANNDATA_KEY])
        if data_config["rep_dim"] and data_config["rep_dim"] < latent_dim:
            logging.warning(
                "It is recommended that `use_rep` dimensionality "
                "be equal or larger than `latent_dim`."
            )
        data_config["batches"] = pd.Index([]) if data_config["batches"] is None \
            else pd.Index(data_config["batches"])
        all_ct = all_ct.union(
            set() if data_config["cell_types"] is None
            else data_config["cell_types"]
        )
        modalities[k] = data_config
    return modalities

def main(args: argparse.Namespace) -> None:
    r"""
    Main function
    """

    # ===================================================#
    latent_dim = 50
    data_batch_size = 128

    print("[1/4] Reading Preprocessed data...")
    rna = anndata.read_h5ad(args.input_rna)
    atac = anndata.read_h5ad(args.input_atac)
    print(rna)
    print(atac)
    gene_to_peak = load_npz(args.prior).toarray()
    prior = torch.tensor(gene_to_peak, dtype=torch.float).to(device) # 将prior作为可优化的参数矩阵

    seed = 0
    # 设定随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_model = Path("Result/FactVAE/{}/{}-FactVAE-seed {}.pth".format(args.dataset, args.dataset, seed))
    output_rna = Path("Result/FactVAE/{}/{}-FactVAE-RNA-latent-seed {}.csv".format(args.dataset, args.dataset, seed))
    output_atac = Path("Result/FactVAE/{}/{}-FactVAE-ATAC-latent-seed {}.csv".format(args.dataset, args.dataset, seed))
    output_rna_prior = Path("Result/FactVAE/{}/{}-FactVAE-inference_rna_prior-seed {}.csv".format(args.dataset, args.dataset, seed))
    output_atac_prior= Path("Result/FactVAE/{}/{}-FactVAE-inference_atac_prior-seed {}.csv".format(args.dataset, args.dataset, seed))

    scglue.models.configure_dataset(rna, "ZINB", use_highly_variable=False, use_rep="X_pca")
    scglue.models.configure_dataset(atac, "ZINB", use_highly_variable=False, use_rep="X_lsi")

    print("[3/4] Training FactVAE...")
    output_dir = args.run_info.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    adatas = {"rna": rna, "atac": atac}
    modalities = get_modalities(adatas)

    train_dataset = dataInstance(
        RNA_adata = rna,
        ATAC_adata = atac,
        RNA_label_colname = "cell_type",
        ATAC_label_colname = "cell_type",
        RNA_rep="X_pca",
        ATAC_rep= "X_lsi"
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=data_batch_size, shuffle=True,
        num_workers=config.DATALOADER_NUM_WORKERS, pin_memory=True, sampler=None, drop_last=True)

    vae = VAE(layer_rna_e=[modalities["rna"]["rep_dim"], 512, 256], layer_atac_e=[modalities["atac"]["rep_dim"], 256],
              zdim=latent_dim, rna_enc_dropout=0.2, atac_enc_dropout=0.2, prior=prior, beta=1,
              type_d_rna=modalities["rna"]["prob_model"], type_d_atac=modalities["atac"]["prob_model"],
              gene_dim=2000, peak_dim=len(modalities["atac"]["features"]), device = device).to(device)

    print(vae)
    genopts = {'vae': optim.Adam(vae.parameters(), lr=0.001),
               'enc_atac': optim.Adam(vae.ATAC_encoder.parameters(), lr=0.0005)}  # Optimizers
    train_schedulers = {"vae": ExponentialLR(genopts["vae"], gamma=0.995),
                        "enc_atac": ExponentialLR(genopts["enc_atac"], gamma=0.995)}

    rna_vae_losses = []
    atac_vae_losses = []
    start_time = time.time()
    for epoch in range(args.max_epoch):
        epoch_rna_loss = []
        epoch_atac_loss = []
        epoch_prior_loss = []
        epoch_sim_loss = []
        prior_alpha = 1.0 # the weight of the prior knowledge constraint term
        sim = True # training the encoder within ATAC-VAE with the high-order feature constraint
        for i, (index, rna_sample, atac_sample) in enumerate(train_loader):
            rna_train_data = rna_sample[0].float().to(device)
            atac_train_data = atac_sample[0].float().to(device)
            rna_raw = rna_sample[1].float().to(device) # origianl gene expression matrix
            atac_raw = atac_sample[1].float().to(device)
            rna_size_factors = rna_sample[2].float().to(device) # size_factor of rna data
            atac_size_factors = atac_sample[2].float().to(device) # size_factor of atac data

            rna_loss, (rna_mse, rna_kl), (rna_codes, _), \
                atac_loss, (atac_mse, atac_kl), (atac_codes, _) = \
                vae.forward(rna_inputs=rna_train_data, rna_X=rna_raw, rna_scale_factor=rna_size_factors, rna_beta=0.001, rna_sigma=0.1,
                            atac_inputs=atac_train_data, atac_X=atac_raw, atac_scale_factor=atac_size_factors, atac_beta=0.001, atac_sigma=0.002)

            if prior_alpha > 0.0:
                loss_prior = vae.forward_prior_loss()
                epoch_prior_loss.append(loss_prior.detach().cpu().numpy())

            loss = rna_loss + atac_loss + prior_alpha * torch.exp(loss_prior)
            genopts["vae"].zero_grad()
            loss.backward()
            genopts["vae"].step()

            if sim:
                loss_similar = vae.forward_similar_loss(rna_inputs=rna_train_data, atac_inputs=atac_train_data)
                epoch_sim_loss.append(loss_similar.detach().cpu().numpy())
            else:
                loss_similar = torch.zeros([]).to(vae.device)
                epoch_sim_loss.append(loss_similar.detach().cpu().numpy())
            genopts["enc_atac"].zero_grad()
            loss_similar.backward()
            genopts["enc_atac"].step()

            epoch_rna_loss.append(rna_loss.detach().cpu().numpy())
            epoch_atac_loss.append(atac_loss.detach().cpu().numpy())

        if epoch %10 ==0:
            print("epoch {}\trna loss {:.3f}\tatac loss {:.3f}\tprior loss {:.3f}\tsim loss {:.3f}".format(epoch,
                                                                                      sum(epoch_rna_loss) / len(epoch_rna_loss),
                                                                                      sum(epoch_atac_loss) / len(epoch_atac_loss),
                                                                                      sum(epoch_prior_loss) / len(epoch_prior_loss),
                                                                                      sum(epoch_sim_loss) / len(epoch_sim_loss)))
        rna_vae_losses.append(sum(epoch_rna_loss) / len(epoch_rna_loss))
        atac_vae_losses.append(sum(epoch_atac_loss) / len(epoch_atac_loss))

        # Updating the lr of the optimizer
        train_schedulers["vae"].step()
        train_schedulers["enc_atac"].step()

    vae.eval()
    rna_latent, atac_latent = vae.get_emb(torch.tensor(rna.obsm[modalities["rna"]["use_rep"]]).to(device),
                                          torch.tensor(atac.obsm[modalities["atac"]["use_rep"]].astype("float32")).to(device))
    elapsed_time = time.time() - start_time

    # Visualize the loss function curve
    plt.figure(figsize=(15, 5))
    epochs = range(1, len(rna_vae_losses))
    plt.plot(epochs, rna_vae_losses[1:], label='pretrain rna', color='b')
    plt.title('pretrain rna Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('Result/FactVAE/{}/{}-FactVAE-rna_Loss-seed {}.png'.format(args.dataset, args.dataset, seed), dpi=400)

    plt.figure(figsize=(15, 5))
    epochs = range(1, len(atac_vae_losses))
    plt.plot(epochs, atac_vae_losses[1:], label='pretrain atac', color='g')
    plt.title('pretrain atac Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('Result/FactVAE/{}/{}-FactVAE-atac_Loss-seed {}.png'.format(args.dataset, args.dataset, seed), dpi=400)

    print("[4/4] Saving results...")
    # Saving model
    torch.save(vae.state_dict(), output_model)
    # vae.load_state_dict(torch.load(args.output_model)) # Loading model
    # Saving the embeddings of genes and peaks
    gene_emb = pd.DataFrame(vae.decoder_RNA.feature.detach().cpu().numpy().T, index=rna.var.query("highly_variable").index)
    gene_emb.to_csv(output_rna_prior)
    peak_emb = pd.DataFrame(vae.decoder_ATAC.feature.detach().cpu().numpy().T, index=atac.var.index)
    peak_emb.to_csv(output_atac_prior)
    # Saving the embeddings of cells
    output_rna.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rna_latent.cpu().detach().numpy(), index=rna.obs_names).to_csv(output_rna, header=False)
    output_atac.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(atac_latent.cpu().detach().numpy(), index=atac.obs_names).to_csv(output_atac, header=False)
    args.run_info.parent.mkdir(parents=True, exist_ok=True)
    with args.run_info.open("w") as f:
        # Output the parameter settings of the run and some results
        yaml.dump({
            "cmd": " ".join(sys.argv),
            "args": vars(args),
            "time": elapsed_time,
            "n_cells": atac.shape[0] + rna.shape[0]
        }, f)


if __name__ == "__main__":
    main(parse_args(dataset="Human CellLine"))
