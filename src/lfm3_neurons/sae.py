"""TopK Sparse Autoencoder in PyTorch.

Architecture:
  encoder:  d_in -> d_sae  (ReLU)
  decoder: d_sae -> d_in   (separate weights)
  sparsity: TopK (keep top-k activations, zero the rest)

Training:
  reconstruction loss + L1 sparsity penalty on the latent
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import ensure_dir, save_json


@dataclass
class SAEConfig:
    layer: int = 6
    d_in: int = 2048
    d_sae: int = 16384  # 8x expansion
    k: int = 64  # TopK sparsity
    lr: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 10
    l1_coeff: float = 3e-4
    seed: int = 42
    device: str = "mps"


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder."""

    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)

        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias) if self.decoder.bias is not None else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def topk(self, z: torch.Tensor) -> torch.Tensor:
        """Apply TopK sparsity per row."""
        k = min(self.k, z.shape[1])
        values, _ = torch.topk(z, k, dim=1)
        threshold = values[:, -1:]  # (batch, 1)
        return torch.where(z >= threshold, z, torch.zeros_like(z))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        z_sparse = self.topk(z)
        x_recon = self.decode(z_sparse)
        return x_recon, z, z_sparse

    def loss(self, x: torch.Tensor, x_recon: torch.Tensor, z_sparse: torch.Tensor) -> dict:
        recon_loss = torch.mean((x - x_recon) ** 2)
        l1_loss = torch.mean(torch.abs(z_sparse))
        total = recon_loss + self.l1_coeff * l1_loss
        return {"loss": total, "recon": recon_loss, "l1": l1_loss}


def train_sae(
    activations_path: str,
    config: SAEConfig,
    output_path: str,
):
    """Train a TopK SAE on saved activations."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    # Load activations
    acts = np.load(activations_path).astype(np.float32)
    num_tokens, d_in = acts.shape
    print(f"Loaded activations: {acts.shape}")
    config.d_in = d_in  # override from actual activation dimension

    # Normalize
    mean = acts.mean(axis=0, keepdims=True)
    std = acts.std(axis=0, keepdims=True) + 1e-8
    acts = (acts - mean) / std

    # Save normalization stats
    norm_path = Path(output_path).with_suffix(".norm.npz")
    ensure_dir(str(norm_path.parent))
    np.savez(norm_path, mean=mean, std=std)
    print(f"Saved normalization stats to {norm_path}")

    # Initialize SAE
    sae = TopKSAE(d_in=config.d_in, d_sae=config.d_sae, k=config.k).to(device)
    sae.l1_coeff = config.l1_coeff

    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr)

    acts_tensor = torch.from_numpy(acts).to(device)
    num_batches = num_tokens // config.batch_size
    print(f"Training: {num_batches} batches/epoch, {config.num_epochs} epochs, device={device}")

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        indices = torch.randperm(num_tokens)

        for b in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            batch_idx = indices[b * config.batch_size : (b + 1) * config.batch_size]
            x = acts_tensor[batch_idx]

            optimizer.zero_grad()
            x_recon, z, z_sparse = sae(x)
            loss_dict = sae.loss(x, x_recon, z_sparse)
            loss_dict["loss"].backward()
            optimizer.step()

            epoch_loss += float(loss_dict["loss"])

            if (b + 1) % 50 == 0:
                print(f"  Batch {b+1}/{num_batches}, loss={float(loss_dict['loss']):.4f}")

        epoch_loss /= num_batches
        print(f"Epoch {epoch+1}: avg_loss={epoch_loss:.4f}")

    # Save SAE
    ensure_dir(str(Path(output_path).parent))
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "config": config.__dict__,
            "d_in": config.d_in,
            "d_sae": config.d_sae,
            "k": config.k,
            "mean": mean,
            "std": std,
        },
        output_path,
    )
    print(f"SAE saved to {output_path}")
    return sae


def train_cli():
    """CLI entry point: train SAE."""
    import argparse

    parser = argparse.ArgumentParser(description="Train TopK SAE on LFM3 activations")
    parser.add_argument("--activations", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--d-sae", type=int, default=16384)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    config = SAEConfig(
        layer=args.layer,
        d_sae=args.d_sae,
        k=args.k,
        lr=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    train_sae(args.activations, config, args.output)