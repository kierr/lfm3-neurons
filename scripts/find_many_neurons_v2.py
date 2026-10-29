"""Batch discovery v2: scan ALL 24 layers, test more candidates, multiple alphas.

Improvements over v1:
- Scans all 24 layers (not just 7)
- Tests alpha=+10 and alpha=-10 per neuron
- Uses 2 different test prompts
- More sensitive effect detection (first 15 chars differ, or length ratio outside [0.75, 1.33])
- Stops at target count
- Saves progress incrementally
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfm3_neurons.model import LFM3Model, ModelConfig
from lfm3_neurons.find_neurons import (
    find_contrastive_neurons,
    _generate_with_intervention,
)
from lfm3_neurons.utils import ensure_dir, save_json


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

TEST_PROMPTS = [
    "The best way to deal with enemies is",
    "When someone is angry at me, I",
]
ALPHAS = [10.0, -10.0]
MAX_TOKENS = 60


def output_has_effect(unsteered: str, steered: str) -> bool:
    """Detect if steering meaningfully changed the output."""
    u = unsteered.strip()
    s = steered.strip()
    if not u:
        return False
    # Different first 15 characters
    if u[:15] != s[:15]:
        return True
    # Length ratio outside [0.75, 1.33]
    ratio = len(s) / len(u)
    if ratio < 0.75 or ratio > 1.33:
        return True
    return False


def test_neuron(
    model: LFM3Model,
    layer: int,
    neuron_idx: int,
) -> dict | None:
    """Test a neuron with multiple alphas and prompts. Returns first confirmed result."""
    for prompt in TEST_PROMPTS:
        unsteered = model.generate(prompt, max_tokens=MAX_TOKENS)
        for alpha in ALPHAS:
            steered = _generate_with_intervention(
                model, prompt, layer, neuron_idx, alpha, MAX_TOKENS
            )
            if output_has_effect(unsteered, steered):
                return {
                    "layer": layer,
                    "neuron_idx": neuron_idx,
                    "prompt": prompt,
                    "alpha": alpha,
                    "unsteered": unsteered.strip(),
                    "steered": steered.strip(),
                }
    return None


def main():
    parser = argparse.ArgumentParser(description="Find many steering neurons (v2)")
    parser.add_argument("--target", type=int, default=50,
                        help="Target number of confirmed neurons")
    parser.add_argument("--per-layer", type=int, default=15,
                        help="Top candidates per layer to test")
    parser.add_argument("--layers", type=str, default="all",
                        help="Comma-separated layers, or 'all'")
    args = parser.parse_args()

    if args.layers == "all":
        layers = list(range(24))
    else:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    print(f"Loading LFM3-1.7B-Base on MPS...")
    cfg = ModelConfig()
    model = LFM3Model(cfg)
    model.load()

    confirmed: list[dict] = []
    tested_count = 0

    # Load previous results if exist
    out_path = Path("./data/confirmed_neurons_v2.json")
    if out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        confirmed = prev.get("neurons", [])
        print(f"Resuming with {len(confirmed)} previously confirmed neurons")

    for layer in layers:
        if len(confirmed) >= args.target:
            break

        print(f"\n{'='*60}")
        print(f"Layer {layer}: top {args.per_layer} contrastive neurons")
        print(f"{'='*60}")

        results = find_contrastive_neurons(
            model,
            POSITIVE_PROMPTS,
            NEGATIVE_PROMPTS,
            layer=layer,
            top_k=args.per_layer,
        )

        for r in results:
            if len(confirmed) >= args.target:
                break

            neuron_idx = r["neuron_idx"]
            tested_count += 1

            print(f"  L{layer} N{neuron_idx} (contrast={r['contrast']:+.4f}, score={r['score']:.4f})...", end=" ")

            try:
                result = test_neuron(model, layer, neuron_idx)
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            if result:
                entry = {
                    "layer": layer,
                    "neuron_idx": neuron_idx,
                    "contrast_score": r["score"],
                    "contrast": r["contrast"],
                    "pos_mean": r["pos_mean"],
                    "neg_mean": r["neg_mean"],
                    **result,
                }
                confirmed.append(entry)
                print(f"CONFIRMED ({len(confirmed)}/{args.target})")
                print(f"    alpha={result['alpha']} prompt='{result['prompt'][:40]}'")
                print(f"    unsteered: {result['unsteered'][:70]}")
                print(f"    steered:   {result['steered'][:70]}")

                # Save incrementally
                save_json(
                    {
                        "target": args.target,
                        "confirmed_count": len(confirmed),
                        "tested_count": tested_count,
                        "layers_scanned": layers,
                        "neurons": confirmed,
                    },
                    str(out_path),
                )
            else:
                print("no effect")

    save_json(
        {
            "target": args.target,
            "confirmed_count": len(confirmed),
            "tested_count": tested_count,
            "layers_scanned": layers,
            "neurons": confirmed,
        },
        str(out_path),
    )

    print(f"\n{'='*60}")
    print(f"DONE: {len(confirmed)}/{args.target} confirmed neurons")
    print(f"Tested {tested_count} candidates across {len(layers)} layers")
    print(f"Saved to {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
