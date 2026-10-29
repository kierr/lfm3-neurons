#!/usr/bin/env python3
"""Find contrastive steering-neuron candidates at a layer.

Runs the model on two prompt sets (aggression-flavored vs cooperation-flavored
by default), ranks layer-L MLP neurons by activation contrast, and saves the
top-k candidates with their full generations from a quick steering test.

Usage:
    python scripts/find_neurons.py --layer 6 --top-k 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfm3_neurons.model import LFM3Model, ModelConfig
from lfm3_neurons.steering import find_contrastive_neurons, steer_generate
from lfm3_neurons.utils import save_json

POSITIVE_PROMPTS = [
    "The best way to deal with enemies is",
    "I think people who disagree with me are",
    "My ideal world is",
    "You should always",
    "The problem with society is",
]
NEGATIVE_PROMPTS = [
    "The best way to make friends is",
    "I think people who agree with me are",
    "A peaceful world is",
    "You should never",
    "The best thing about society is",
]
TEST_PROMPT = "The best way to deal with enemies is"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=2.0, help="steering strength for the quick test")
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--output", default="data/contrastive_neurons.json")
    args = parser.parse_args()

    model = LFM3Model(ModelConfig()).load()

    print(f"Finding contrastive neurons at layer {args.layer}...")
    neurons = find_contrastive_neurons(
        model, POSITIVE_PROMPTS, NEGATIVE_PROMPTS, layer=args.layer, top_k=args.top_k
    )

    print(f"\n{'neuron':>7} | {'pos_mean':>9} | {'neg_mean':>9} | {'contrast':>9}")
    print("-" * 48)
    for n in neurons:
        print(f"{n['neuron_idx']:>7} | {n['pos_mean']:>9.4f} | {n['neg_mean']:>9.4f} | {n['contrast']:>9.4f}")

    print(f"\nQuick steering test (alpha={args.alpha:+g}, prompt='{TEST_PROMPT}'):")
    for n in neurons:
        steered = steer_generate(
            model, TEST_PROMPT, args.layer, n["neuron_idx"], args.alpha,
            max_tokens=args.max_tokens, seed=42,
        )
        n["steered_example"] = steered.strip()
        print(f"  N{n['neuron_idx']:>5} -> '{steered.strip()[:80]}'")

    save_json({"layer": args.layer, "neurons": neurons}, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
