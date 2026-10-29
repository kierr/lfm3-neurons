"""TopK sparse autoencoder for MLP intermediate activations.

Architecture::

    encoder: d_in -> d_sae   (ReLU, bias)
    topk:    keep the k largest latents per token, zero the rest
    decoder: d_sae -> d_in   (no bias)

Training minimizes reconstruction MSE plus an L1 penalty on the (already
sparse) TopK latents. TopK sparsity removes the need for auxiliary
dead-latent losses at this scale.

The SAE here is secondary to the neuron work: it was trained on layer-6
intermediates from TinyStories as a first decomposition step, but the
headline results in this repo all come from raw neurons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import ensure_dir


@dataclass
class SAEConfig:
    layer: int = 6
    d_in: int = 7680  # MLP intermediate width (overridden from data if different)
    d_sae: int = 16384  # ~2x expansion of the 7680-dim intermediate
    k: int = 64
    lr: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 10
    l1_coeff: float = 3e-4
    seed: int = 42
    device: str = "mps"


class TopKSAE(nn.Module):
    """TopK sparse autoencoder: ReLU encoder, per-token top-k, linear decoder."""

    def __init__(self, d_in: int, d_sae: int, k: int, l1_coeff: float = 3e-4):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.l1_coeff = l1_coeff

        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def topk(self, z: torch.Tensor) -> torch.Tensor:
        """Keep the k largest latents per row, zero the rest."""
        k = min(self.k, z.shape[1])
        values, _ = torch.topk(z, k, dim=1)
        threshold = values[:, -1:]
        return torch.where(z >= threshold, z, torch.zeros_like(z))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z_sparse = self.topk(self.encode(x))
        return self.decode(z_sparse), z_sparse

    def loss(self, x: torch.Tensor, x_recon: torch.Tensor, z_sparse: torch.Tensor) -> dict:
        recon = torch.mean((x - x_recon) ** 2)
        l1 = torch.mean(torch.abs(z_sparse))
        return {"loss": recon + self.l1_coeff * l1, "recon": recon, "l1": l1}


def train_sae(activations_path: str, config: SAEConfig, output_path: str) -> TopKSAE:
    """Train a TopK SAE on a saved (num_tokens, d_in) float32 activation matrix.

    Activations are per-dimension standardized before training; the mean/std
    used are stored alongside the checkpoint so encodes can be reproduced.
    """
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    acts = np.load(activations_path).astype(np.float32)
    num_tokens, d_in = acts.shape
    config.d_in = d_in
    print(f"Loaded activations: {acts.shape}")

    mean = acts.mean(axis=0, keepdims=True)
    std = acts.std(axis=0, keepdims=True) + 1e-8
    acts = (acts - mean) / std

    norm_path = Path(output_path).with_suffix(".norm.npz")
    ensure_dir(str(norm_path.parent))
    np.savez(norm_path, mean=mean, std=std)

    sae = TopKSAE(d_in, config.d_sae, config.k, config.l1_coeff).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr)
    acts_tensor = torch.from_numpy(acts).to(device)
    num_batches = num_tokens // config.batch_size
    print(f"Training: {num_batches} batches/epoch x {config.num_epochs} epochs on {device}")

    for epoch in range(config.num_epochs):
        indices = torch.randperm(num_tokens)
        epoch_loss = 0.0
        for b in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{config.num_epochs}"):
            x = acts_tensor[indices[b * config.batch_size : (b + 1) * config.batch_size]]
            optimizer.zero_grad()
            x_recon, z_sparse = sae(x)
            loss = sae.loss(x, x_recon, z_sparse)
            loss["loss"].backward()
            optimizer.step()
            epoch_loss += float(loss["loss"])
        print(f"Epoch {epoch + 1}: avg_loss={epoch_loss / num_batches:.4f}")

    ensure_dir(str(Path(output_path).parent))
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "config": config.__dict__,
            "mean": mean,
            "std": std,
        },
        output_path,
    )
    print(f"SAE saved to {output_path}")
    return sae
