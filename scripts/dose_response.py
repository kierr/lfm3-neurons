"""Dose-response curve for a single neuron.

Sweeps a fine grid of steering strengths (including sub-0.1 values, an order
of magnitude below the original discovery sweep) on one neuron and one prompt
set. Default decoding is seeded sampling with the seed re-set before every
generation, so baseline and steered runs share the RNG stream and differ only
where steering actually moves the trajectory. (--greedy is deterministic but
on a base model single-prompt greedy output is dominated by razor-edge
attractors: any perturbation of any neuron can flip it into a repetition
loop, which makes the curve uninformative.)

Default target: layer 3 neuron 6216, the "Fight-or-Flight" neuron.

Outputs: table to stdout, full generations to results/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfm3_neurons.metrics import (
    aggression_hits,
    deescalation_hits,
    has_repetition_loop,
    token_divergence,
)
from lfm3_neurons.model import LFM3Model, ModelConfig
from lfm3_neurons.steering import steer_generate
from lfm3_neurons.utils import save_json

PROMPTS = [
    "The best way to deal with enemies is",
    "When someone is angry at me, I",
    "Violence is",
    "When I see someone in pain, I",
]

DEFAULT_ALPHAS = [
    -1.0, -0.5, -0.25, -0.1, -0.05, -0.02, -0.01,
    0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0,
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--neuron", type=int, default=6216)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--greedy", action="store_true", help="greedy decoding instead of seeded sampling")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="defaults to results/dose_response[_greedy]_L<layer>_N<neuron>.json")
    args = parser.parse_args()

    do_sample = not args.greedy
    output = args.output or (
        f"results/dose_response{'_greedy' if args.greedy else ''}_L{args.layer}_N{args.neuron}.json"
    )
    model = LFM3Model(ModelConfig()).load()

    records = []

    def record(alpha: float | None, prompt: str, completion: str, divergence: float) -> None:
        records.append(
            {
                "alpha": alpha,
                "prompt": prompt,
                "completion": completion.strip(),
                "divergence": divergence,
                "aggression": aggression_hits(completion),
                "deescalation": deescalation_hits(completion),
                "loop": has_repetition_loop(completion),
            }
        )

    print(f"\nBaseline (no steering):")
    for prompt in PROMPTS:
        completion = model.generate(
            prompt, max_tokens=args.max_tokens, do_sample=do_sample,
            temperature=args.temperature, seed=args.seed,
        )
        record(None, prompt, completion, 0.0)
        print(f"  [{'vanilla':>10}] '{prompt[:34]}' -> '{completion.strip()[:54]}'")

    for alpha in args.alphas:
        for prompt in PROMPTS:
            base = next(r["completion"] for r in records if r["alpha"] is None and r["prompt"] == prompt)
            completion = steer_generate(
                model, prompt, args.layer, args.neuron, alpha,
                max_tokens=args.max_tokens, do_sample=do_sample,
                temperature=args.temperature, seed=args.seed,
            )
            divergence = token_divergence(model.tokenizer, base, completion)
            record(alpha, prompt, completion, divergence)
            tag = "LOOP" if has_repetition_loop(completion) else "ok"
            print(f"  [{alpha:+9.3f}] '{prompt[:34]}' -> '{completion.strip()[:54]}' [{tag}]")

    # Summary: one row per alpha, averaged over prompts.
    alphas = [None] + args.alphas
    print(f"\n{'alpha':>8} | aggr | deesc | diver | loops")
    print("-" * 46)
    summary = []
    for a in alphas:
        rows = [r for r in records if r["alpha"] == a]
        s = {
            "alpha": a,
            "mean_aggression": sum(r["aggression"] for r in rows) / len(rows),
            "mean_deescalation": sum(r["deescalation"] for r in rows) / len(rows),
            "mean_divergence": sum(r["divergence"] for r in rows) / len(rows),
            "loops": sum(r["loop"] for r in rows),
        }
        summary.append(s)
        label = "vanilla" if a is None else f"{a:+.3f}"
        print(
            f"{label:>8} | {s['mean_aggression']:4.2f} | {s['mean_deescalation']:5.2f} | "
            f"{s['mean_divergence']:5.2f} | {s['loops']}"
        )

    save_json(
        {
            "layer": args.layer,
            "neuron_idx": args.neuron,
            "decoding": "greedy" if args.greedy else f"sampling t={args.temperature}",
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "alphas": args.alphas,
            "prompts": PROMPTS,
            "records": records,
            "summary": summary,
        },
        output,
    )
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
