#!/usr/bin/env python3
"""Find contrastive neurons in LFM3.

Usage:
    python scripts/find_neurons.py --config configs/sae.yaml --layer 6 --top-k 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lfm3_neurons.find_neurons import main


def cli():
    parser = argparse.ArgumentParser(description="Find steering neurons")
    parser.add_argument("--config", type=str, default="configs/sae.yaml")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    # Temporarily override sys.argv for the main() function
    import sys as _sys
    _sys.argv = [
        "find_neurons",
        "--config", args.config,
        "--layer", str(args.layer),
        "--top-k", str(args.top_k),
    ]
    main()


if __name__ == "__main__":
    cli()