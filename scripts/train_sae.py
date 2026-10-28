#!/usr/bin/env python3
"""Train TopK SAE on extracted LFM3 activations.

Usage:
    python scripts/train_sae.py --activations data/activations/layer_6_activations.npy \
        --output data/sae/layer_6_sae.pt --d-sae 16384 --k 64 --epochs 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lfm3_neurons.sae import train_cli


def main():
    train_cli()


if __name__ == "__main__":
    main()