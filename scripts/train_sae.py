#!/usr/bin/env python3
"""Train a TopK sparse autoencoder on extracted activations.

Usage:
    python scripts/train_sae.py --activations data/activations/layer_6_activations.npy \
        --output data/sae/layer_6_sae.pt --d-sae 16384 --k 64 --epochs 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfm3_neurons.sae import SAEConfig, train_sae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", required=True, help="path to the (num_tokens, d_in) .npy file")
    parser.add_argument("--output", required=True, help="path for the .pt checkpoint")
    parser.add_argument("--layer", type=int, default=6, help="recorded in the checkpoint metadata")
    parser.add_argument("--d-sae", type=int, default=16384)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="mps")
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


if __name__ == "__main__":
    main()
