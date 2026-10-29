#!/usr/bin/env python3
"""Extract MLP intermediate activations from a text corpus, for SAE training.

Streams TinyStories (or any HF text dataset) through the model in batches and
saves the flattened (num_tokens, 7680) MLP intermediates for one layer as a
float32 .npy file. Only one layer's activations are held in memory at a time.

Usage:
    python scripts/extract_activations.py --layer 6 --max-samples 5000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfm3_neurons.model import LFM3Model, ModelConfig
from lfm3_neurons.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default=None, help="defaults to data/activations/layer_<L>_activations.npy")
    args = parser.parse_args()

    output = args.output or f"data/activations/layer_{args.layer}_activations.npy"
    model = LFM3Model(ModelConfig()).load()

    ds = load_dataset(args.dataset, split=args.split)
    texts = [example["text"] for example in ds.select(range(min(args.max_samples, len(ds))))]

    chunks = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Extracting"):
        batch = texts[i : i + args.batch_size]
        tokens = model.tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length
        )
        captured = model.forward_with_capture(tokens["input_ids"].to(model.device), [args.layer])
        acts = captured[args.layer]  # (batch, seq, intermediate)
        chunks.append(acts.reshape(-1, acts.shape[-1]).cpu())

    flat = torch.cat(chunks).float().numpy()
    print(f"Total activations: {flat.shape}")

    ensure_dir(str(Path(output).parent))
    np.save(output, flat)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
