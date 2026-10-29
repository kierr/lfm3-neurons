"""Batch discovery of steering neurons across multiple LFM3 layers.

Loads the model once, runs contrastive activation analysis per layer,
then tests each candidate with activation-addition steering to confirm
behavioral change. Collects confirmed neurons until target count reached.

Usage:
    python scripts/find_many_neurons.py --target 50 --layers 3,6,9,12,15,18,21
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


# Contrastive prompt pairs — positive condition vs negative condition
POSITIVE_PROMPTS = [
    "The best way to deal with enemies is",
    "I think people who disagree with me are",
    "My ideal world is",
    "You should always",
    "The problem with society is",
    "When someone insults me, I",
    "The most important thing in life is",
    "People who are different from me",
    "The government should",
    "Success in life comes from",
]
NEGATIVE_PROMPTS = [
    "The best way to make friends is",
    "I think people who agree with me are",
    "A peaceful world is",
    "You should never",
    "The best thing about society is",
    "When someone compliments me, I",
    "The least important thing in life is",
    "People who are similar to me",
    "The government should not",
    "Failure in life comes from",
]

# Test prompt used to verify steering effect
STEER_TEST_PROMPT = "The best way to deal with enemies is"

# Alpha values to test per neuron
ALPHAS = [5.0, 10.0]

MAX_TOKENS = 60


def neuron_has_effect(
    model: LFM3Model,
    layer: int,
    neuron_idx: int,
    prompt: str = STEER_TEST_PROMPT,
    alpha: float = 10.0,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """Test whether steering a neuron changes the output.

    Returns a dict with unsteered text, steered text, and whether they differ
    meaningfully (not just trivially).
    """
    unsteered = model.generate(prompt, max_tokens=max_tokens)
    steered = _generate_with_intervention(
        model, prompt, layer, neuron_idx, alpha, max_tokens
    )

    # Simple heuristic: different first ~30 chars or different length ratio
    unsteered_clean = unsteered.strip()
    steered_clean = steered.strip()

    same_prefix = unsteered_clean[:30] == steered_clean[:30]
    len_ratio = (
        len(steered_clean) / max(len(unsteered_clean), 1)
        if unsteered_clean
        else 1.0
    )

    # "Effect" means meaningfully different output
    has_effect = (not same_prefix) or (abs(len_ratio - 1.0) > 0.3)

    return {
        "neuron_idx": neuron_idx,
        "layer": layer,
        "alpha": alpha,
        "unsteered": unsteered_clean,
        "steered": steered_clean,
        "same_prefix": same_prefix,
        "len_ratio": round(len_ratio, 3),
        "has_effect": has_effect,
    }


def main():
    parser = argparse.ArgumentParser(description="Find many steering neurons")
    parser.add_argument("--target", type=int, default=50,
                        help="Target number of confirmed neurons")
    parser.add_argument("--layers", type=str, default="3,6,9,12,15,18,21",
                        help="Comma-separated layer indices to scan")
    parser.add_argument("--per-layer", type=int, default=25,
                        help="Top candidates per layer to test")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Steering strength for testing")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",")]

    print(f"Loading LFM3-1.7B-Base on MPS...")
    cfg = ModelConfig()
    model = LFM3Model(cfg)
    model.load()

    confirmed: list[dict] = []
    tested_count = 0

    for layer in layers:
        if len(confirmed) >= args.target:
            break

        print(f"\n{'='*60}")
        print(f"Layer {layer}: finding top {args.per_layer} contrastive neurons")
        print(f"{'='*60}")

        results = find_contrastive_neurons(
            model,
            POSITIVE_PROMPTS,
            NEGATIVE_PROMPTS,
            layer=layer,
            top_k=args.per_layer,
        )

        print(f"Top candidates at layer {layer}:")
        for r in results[:10]:
            print(f"  neuron {r['neuron_idx']:>5} score={r['score']:.4f} "
                  f"contrast={r['contrast']:+.4f}")

        for r in results:
            if len(confirmed) >= args.target:
                break

            neuron_idx = r["neuron_idx"]
            tested_count += 1

            print(f"\n  Testing layer={layer} neuron={neuron_idx} "
                  f"(contrast score={r['score']:.4f})...", end=" ")

            try:
                result = neuron_has_effect(
                    model, layer, neuron_idx,
                    prompt=STEER_TEST_PROMPT,
                    alpha=args.alpha,
                )
                tested_count += 0  # already incremented
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            if result["has_effect"]:
                entry = {
                    "layer": layer,
                    "neuron_idx": neuron_idx,
                    "contrast_score": r["score"],
                    "contrast": r["contrast"],
                    "pos_mean": r["pos_mean"],
                    "neg_mean": r["neg_mean"],
                    "alpha": result["alpha"],
                    "unsteered": result["unsteered"],
                    "steered": result["steered"],
                    "same_prefix": result["same_prefix"],
                    "len_ratio": result["len_ratio"],
                }
                confirmed.append(entry)
                print(f"CONFIRMED ({len(confirmed)}/{args.target})")
                print(f"    unsteered: {result['unsteered'][:80]}")
                print(f"    steered:   {result['steered'][:80]}")
            else:
                print(f"no effect")

    # Save all confirmed neurons
    out_path = Path("./data/confirmed_neurons.json")
    ensure_dir(str(out_path.parent))
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
    print(f"Tested {tested_count} candidates across layers {layers}")
    print(f"Saved to {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
