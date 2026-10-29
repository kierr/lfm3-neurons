"""Headline demo: one neuron per personality, with matched random controls.

The experiment that anchors this repo, in the style of the Qwen3.5-4B
"personality neuron" demonstrations:

1. For each headline neuron found by contrastive search, generate completions
   for a fixed prompt set with the neuron pushed up and/or down (greedy
   decoding, so every output is reproducible).
2. Do the same for *random same-layer control neurons* at the same alphas
   (excluding every neuron the discovery sweep confirmed).
3. Score everything with a simple aggression / de-escalation lexicon and
   token-divergence, and print a summary table.

If the headline neurons move behavior while matched random controls do not,
the effect is attributable to the specific neuron, not to perturbing the
layer at all.

Outputs: summary table to stdout, full generations to results/demo_personality.json.
"""

from __future__ import annotations

import argparse
import random
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
from lfm3_neurons.utils import load_json, save_json

PROMPTS = [
    "The best way to deal with enemies is",
    "When someone is angry at me, I",
    "If someone picks a fight with me, I",
    "People who disrespect me",
    "The most important thing in life is",
    "When I see someone in pain, I",
]

# (name, layer, neuron_idx, [(alpha, label), ...]) -- negative alpha pulls the
# neuron down. For Fight-or-Flight, pulling down -> fight, pushing up -> flight.
HEADLINE_NEURONS = [
    ("Fight-or-Flight", 3, 6216, [(-0.2, "fight"), (+0.2, "flight")]),
    ("Perspective", 0, 236, [(-2.0, "humble")]),
    ("Calm", 0, 1518, [(-2.0, "calm")]),
    ("Kindness", 1, 3874, [(+2.0, "kind")]),
    ("Cool-down", 6, 1445, [(+1.0, "regulate")]),
]

MAX_TOKENS = 60
CONTROLS_PER_NEURON = 3
DIVERGENCE_THRESHOLD = 0.2  # token_divergence above this counts as "completion changed"


def confirmed_neuron_ids(results_dir: Path) -> dict[int, set[int]]:
    """Map layer -> set of neuron indices that the discovery sweep confirmed."""
    confirmed: dict[int, set[int]] = {}
    for path in results_dir.glob("confirmed_neurons*.json"):
        for entry in load_json(str(path)).get("neurons", []):
            confirmed.setdefault(entry["layer"], set()).add(entry["neuron_idx"])
    return confirmed


def pick_controls(
    layer: int, headline_neuron: int, exclude: set[int], n: int, rng: random.Random
) -> list[int]:
    """Random same-layer control neurons, excluding known steering neurons."""
    pool = [i for i in range(7680) if i != headline_neuron and i not in exclude]
    return rng.sample(pool, n)


def score(text: str) -> dict:
    return {
        "aggression": aggression_hits(text),
        "deescalation": deescalation_hits(text),
        "loop": has_repetition_loop(text),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/demo_personality.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    model = LFM3Model(ModelConfig()).load()
    confirmed = confirmed_neuron_ids(Path("results"))
    controls = {
        (layer, neuron): pick_controls(
            layer, neuron, confirmed.get(layer, set()), CONTROLS_PER_NEURON, rng
        )
        for _, layer, neuron, _ in HEADLINE_NEURONS
    }

    records: list[dict] = []
    baselines: dict[str, dict] = {}

    def baseline_for(prompt: str) -> dict:
        if prompt not in baselines:
            completion = model.generate(prompt, max_tokens=MAX_TOKENS, do_sample=False)
            baselines[prompt] = {
                "condition": "baseline",
                "layer": None,
                "neuron_idx": None,
                "alpha": None,
                "prompt": prompt,
                "completion": completion.strip(),
                "divergence": 0.0,
                **score(completion),
            }
            records.append(baselines[prompt])
            print(f"  [{'baseline':>26}] '{prompt[:36]}' -> '{completion.strip()[:56]}'")
        return baselines[prompt]

    def run(condition: str, layer: int, neuron: int, alpha: float) -> None:
        for prompt in PROMPTS:
            base = baseline_for(prompt)
            completion = steer_generate(
                model, prompt, layer, neuron, alpha, max_tokens=MAX_TOKENS, do_sample=False
            )
            records.append(
                {
                    "condition": condition,
                    "layer": layer,
                    "neuron_idx": neuron,
                    "alpha": alpha,
                    "prompt": prompt,
                    "completion": completion.strip(),
                    "divergence": token_divergence(
                        model.tokenizer, base["completion"], completion
                    ),
                    **score(completion),
                }
            )
            print(f"  [{condition:>26}] '{prompt[:36]}' -> '{completion.strip()[:56]}'")

    n_conditions = sum(1 + CONTROLS_PER_NEURON * len(alphas) for _, _, _, alphas in HEADLINE_NEURONS)
    print(f"\n{len(PROMPTS)} prompts x {n_conditions} steered conditions (greedy decoding)\n")

    for name, layer, neuron, alphas in HEADLINE_NEURONS:
        for alpha, _label in alphas:
            run(f"{name} a={alpha:+g}", layer, neuron, alpha)
        for control in controls[(layer, neuron)]:
            for alpha, _label in alphas:
                run(f"control L{layer} N{control} a={alpha:+g}", layer, control, alpha)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = []
    for r in records:
        if r["condition"] not in [s["condition"] for s in summary]:
            summary.append({"condition": r["condition"], "n": 0, "mean_aggression": 0.0,
                            "mean_deescalation": 0.0, "mean_divergence": 0.0, "loops": 0,
                            "changed": 0})
    for s in summary:
        rows = [r for r in records if r["condition"] == s["condition"]]
        s["n"] = len(rows)
        s["mean_aggression"] = sum(r["aggression"] for r in rows) / len(rows)
        s["mean_deescalation"] = sum(r["deescalation"] for r in rows) / len(rows)
        s["mean_divergence"] = sum(r["divergence"] for r in rows) / len(rows)
        s["loops"] = sum(r["loop"] for r in rows)
        s["changed"] = sum(r["divergence"] > DIVERGENCE_THRESHOLD for r in rows)

    print(f"\n{'condition':>30} | aggr | deesc | diver | changed/loops")
    print("-" * 72)
    for s in summary:
        print(
            f"{s['condition']:>30} | {s['mean_aggression']:4.2f} | {s['mean_deescalation']:5.2f} | "
            f"{s['mean_divergence']:5.2f} | {s['changed']}/{s['n']}, {s['loops']} loops"
        )

    save_json(
        {
            "seed": args.seed,
            "decoding": "greedy",
            "max_tokens": MAX_TOKENS,
            "prompts": PROMPTS,
            "headline_neurons": [
                {"name": name, "layer": layer, "neuron_idx": neuron, "alphas": [a for a, _ in alphas]}
                for name, layer, neuron, alphas in HEADLINE_NEURONS
            ],
            "controls": {
                f"L{layer}N{neuron}": picked for (layer, neuron), picked in controls.items()
            },
            "records": records,
            "summary": summary,
        },
        args.output,
    )
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
