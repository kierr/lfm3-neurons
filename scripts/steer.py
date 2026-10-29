#!/usr/bin/env python3
"""Steer one neuron by hand and print the completions.

Usage:
    python scripts/steer.py --layer 3 --neuron 6216 --alpha -0.2 \
        --prompt "The best way to deal with enemies is"

Uses greedy decoding by default so outputs are reproducible; pass --sample
for temperature sampling.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfm3_neurons.model import LFM3Model, ModelConfig
from lfm3_neurons.steering import steer_generate
from lfm3_neurons.utils import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--neuron", type=int, required=True, help="neuron index in the MLP intermediate (0-7679)")
    parser.add_argument("--alpha", type=float, required=True, help="steering strength; sign flips direction")
    parser.add_argument("--prompt", default="The best way to deal with enemies is")
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sample", action="store_true", help="sample instead of greedy decoding")
    parser.add_argument("--output", default=None, help="optional path to save the result as JSON")
    args = parser.parse_args()

    model = LFM3Model(ModelConfig()).load()

    baseline = model.generate(
        args.prompt, max_tokens=args.max_tokens, do_sample=not args.sample,
        temperature=args.temperature,
    )
    steered = steer_generate(
        model, args.prompt, args.layer, args.neuron, args.alpha,
        max_tokens=args.max_tokens, do_sample=not args.sample, temperature=args.temperature,
    )

    print(f"\nL{args.layer} N{args.neuron} alpha={args.alpha:+g} | prompt: {args.prompt!r}")
    print("-" * 72)
    print(f"[baseline]\n{baseline.strip()}")
    print(f"\n[steered]\n{steered.strip()}")

    if args.output:
        save_json(
            {
                "layer": args.layer,
                "neuron_idx": args.neuron,
                "alpha": args.alpha,
                "prompt": args.prompt,
                "baseline": baseline.strip(),
                "steered": steered.strip(),
                "decoding": "sampled" if args.sample else "greedy",
            },
            args.output,
        )
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
