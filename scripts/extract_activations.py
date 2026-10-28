#!/usr/bin/env python3
"""Extract LFM3 activations from a text corpus.

Usage:
    python scripts/extract_activations.py --config configs/sae.yaml --layer 6 --max-samples 5000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lfm3_neurons.model import LFM3Model, ModelConfig, extract_cli


def main():
    extract_cli()


if __name__ == "__main__":
    main()