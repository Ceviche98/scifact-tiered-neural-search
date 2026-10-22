"""
Phase 1/2 Experimental Pipeline: Beyond Matryoshka Sparse Autoencoder (CSR) Training on Stella-1.5B.

Preserved from our Coursework 1 & 2 architectural investigation (`train_csr_stella.py`).
Trains a Top-K Sparse Autoencoder (input_dim=1536 -> hidden_dim=6144, k=32, aux_k=512)
using Multi-Sparsity Reconstruction Loss + Dead-Latent Auxiliary Loss (Wen et al., 2025).

Findings:
- CSR (k=32) on the 5,183-document domain-specific SciFact corpus suffered from representation
  degradation (~0.52 nDCG@10) due to over-parameterization (6,144 latents vs 5,183 docs) and
  Top-K threshold pruning of dense cross-dimensional biomedical semantics.
- This empirical finding motivated our transition to pre-trained Matryoshka Representation
  Learning (MRL 256d) + FAISS IndexFlatIP, boosting Fast Tier nDCG@10 from 0.52 -> 0.7522.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from data_loader import load_scifact_local

device = "cuda" if torch.cuda.is_available() else "cpu"


class CSRAutoencoder(nn.Module):
    """Top-K Sparse Autoencoder projecting dense embeddings into high-dimensional sparse latents."""

    def __init__(self, input_dim: int = 1536, hidden_dim: int = 6144, k: int = 32, aux_k: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.aux_k = aux_k

        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.latent_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        self.register_buffer("dead_steps", torch.zeros(hidden_dim, dtype=torch.long))
        self.dead_threshold = 20

        nn.init.kaiming_uniform_(self.encoder.weight)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
            self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)

    def get_sparse_code(self, z_pre: torch.Tensor, k: int) -> torch.Tensor:
        topk_vals, topk_indices = torch.topk(z_pre, k=k, dim=-1)
        z_k = torch.zeros_like(z_pre)
        return z_k.scatter_(-1, topk_indices, topk_vals)

    def forward(self, x: torch.Tensor):
        x_centered = x - self.pre_bias
        z_pre = F.relu(self.encoder(x_centered) + self.latent_bias)

        if self.training:
            z_k = self.get_sparse_code(z_pre, self.k)
            z_4k = self.get_sparse_code(z_pre, self.k * 4)

            active_mask = (z_k > 0).any(dim=0)
            self.dead_steps[active_mask] = 0
            self.dead_steps[~active_mask] += 1

            x_hat_k = self.decoder(z_k) + self.pre_bias
            x_hat_4k = self.decoder(z_4k) + self.pre_bias
            return x_hat_k, x_hat_4k, z_pre, z_k
        else:
            z_k = self.get_sparse_code(z_pre, self.k)
            x_hat_k = self.decoder(z_k) + self.pre_bias
            return x_hat_k, z_k
