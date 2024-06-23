# -*- coding: utf-8 -*-
'''
@Time    : 2023/11/18 16:42
@Author  : Linjie Wang
@FileName: loader.py
@Software: PyCharm
'''
from anndata._core.anndata import AnnData
from torch.utils.data import Dataset
import torch
import numpy as np

class dataInstance(Dataset):
    def __init__(self,
                 RNA_adata: AnnData = None,
                 ATAC_adata: AnnData = None,
                 RNA_label_colname: str = "x",
                 ATAC_label_colname: str = "x",
                 RNA_rep: str = "x",
                 ATAC_rep: str = "x"
                 ):

        super().__init__()

        self.RNA_adata = RNA_adata
        self.ATAC_adata = ATAC_adata

        # original gene expression matrix
        self.RNA_data = self.RNA_adata.obsm[RNA_rep]
        self.ATAC_data = self.ATAC_adata.obsm[ATAC_rep]
        self.RNA_sf = self.RNA_adata.obs["size_factors"]
        self.ATAC_sf = self.ATAC_adata.obs["size_factors"]

        if isinstance(self.RNA_adata.raw.X, np.ndarray):
            self.RNA_raw_data = self.RNA_adata.raw.X[:, RNA_adata.var.highly_variable]
        else:
            self.RNA_raw_data = self.RNA_adata.raw.X[:, RNA_adata.var.highly_variable].toarray()
        if isinstance(self.ATAC_adata.raw.X, np.ndarray):
            self.ATAC_raw_data = self.ATAC_adata.raw.X
        else:
            self.ATAC_raw_data = self.ATAC_adata.raw.X.toarray()

        # label (if exist, build the label encoder)
        if self.RNA_adata.obs.get(RNA_label_colname) is not None:
            self.RNA_label = self.RNA_adata.obs[RNA_label_colname]
            self.RNA_unique_label = list(set(self.RNA_label))
            self.RNA_label_encoder = {k: v for k, v in zip(self.RNA_unique_label, range(len(self.RNA_unique_label)))}
            self.RNA_label_decoder = {v: k for k, v in self.RNA_label_encoder.items()}
        else:
            self.RNA_label = None
            print("RNA adata can not find corresponding labels")
        self.RNA_cells, self.RNA_genes = self.RNA_adata.shape

        if self.ATAC_adata.obs.get(ATAC_label_colname) is not None:
            self.ATAC_label = self.ATAC_adata.obs[ATAC_label_colname]
            self.ATAC_unique_label = list(set(self.ATAC_label))
            self.ATAC_label_encoder = {k: v for k, v in zip(self.ATAC_unique_label, range(len(self.ATAC_unique_label)))}
            self.ATAC_label_decoder = {v: k for k, v in self.ATAC_label_encoder.items()}
        else:
            self.ATAC_label = None
            print("RNA adata can not find corresponding labels")
        self.ATAC_cells, self.ATAC_genes = self.ATAC_adata.shape

    def __getitem__(self, index):
        RNA_sample = [
            torch.Tensor(self.RNA_data[index].astype(np.float32)),
            torch.Tensor(self.RNA_raw_data[index].astype(np.float32)),
            self.RNA_sf[index].astype(np.float32),
            self.RNA_label_encoder[self.RNA_label[index]]
        ]
        ATAC_sample = [
            torch.Tensor(self.ATAC_data[index].astype(np.float32)),
            torch.Tensor(self.ATAC_raw_data[index].astype(np.float32)),
            self.ATAC_sf[index].astype(np.float32),
            self.ATAC_label_encoder[self.ATAC_label[index]]
        ]

        return index, RNA_sample, ATAC_sample

    def __len__(self):
        return self.RNA_adata.X.shape[0]