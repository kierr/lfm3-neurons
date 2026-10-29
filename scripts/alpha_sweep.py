"""Test smaller alphas on confirmed neurons to check for cleaner steering.

Tests each confirmed neuron at alpha in [2, 5, -2, -5] on both test prompts,
classifying outputs as 'clean' (semantic shift) or 'artifact' (repetition/loop).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfm3_neurons.model import LFM3Model, ModelConfig
from lfm3_neurons.find_neurons import _generate_with_intervention
from lfm3_neurons.utils import save_json


ALPHAS = [2.0, 5.0, -2.0, -5.0]
PROMPTS = [
    "The best way to deal with enemies is",
    "When someone is angry at me, I",
]
MAX_TOKENS = 60


def has_repetition_loop(text: str) -> bool:
    """Heuristic: detect heavy token/phrase repetition (artifact steering)."""
    t = text.strip()
    if len(t) < 20:
        return False

    # Check for repeated single token (e.g., "I, I, I" or "A. A. A")
    import re

    # Same word repeated 5+ times with optional punctuation
    if re.search(r'\b(\w+)(?:\s+\1\b){4,}', t):
        return True

    # Same short phrase (2-4 words) repeated 3+ times
    for n in [2, 3, 4]:
        words = t.split()
        if len(words) < n * 4:
            continue
        phrases = {}
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            phrases[phrase] = phrases.get(phrase, 0) + 1
        for phrase, count in phrases.items():
            if count >= 4 and len(phrase.split()) >= 2:
                return True

    # Very high ratio of identical characters (e.g., "''''''''")
    if re.search(r'(.)\1{8,}', t):
        return True

    return False


def main():
    with open("data/confirmed_neurons_v2.json") as f:
        data = json.load(f)

    neurons = data["neurons"]
    print(f"Testing {len(neurons)} neurons at alphas {ALPHAS} on {len(PROMPTS)} prompts")

    cfg = ModelConfig()
    model = LFM3Model(cfg)
    model.load()

    results = []
    clean_count = 0
    artifact_count = 0

    for i, n in enumerate(neurons):
        layer = n["layer"]
        neuron_idx = n["neuron_idx"]
        original_alpha = n["alpha"]

        for prompt in PROMPTS:
            unsteered = model.generate(prompt, max_tokens=MAX_TOKENS)

            for alpha in ALPHAS:
                try:
                    steered = _generate_with_intervention(
                        model, prompt, layer, neuron_idx, alpha, MAX_TOKENS
                    )
                except Exception as e:
                    print(f"  L{layer} N{neuron_idx} a={alpha} ERROR: {e}")
                    continue

                loop = has_repetition_loop(steered)
                if loop:
                    artifact_count += 1
                else:
                    clean_count += 1

                results.append({
                    "layer": layer,
                    "neuron_idx": neuron_idx,
                    "original_alpha": original_alpha,
                    "prompt": prompt,
                    "alpha": alpha,
                    "unsteered": unsteered.strip(),
                    "steered": steered.strip(),
                    "repetition_loop": loop,
                })

                tag = "ARTIFACT" if loop else "clean"
                print(
                    f"  [{i+1}/{len(neurons)}] L{layer} N{neuron_idx} "
                    f"a={alpha:+.0f} [{tag}] '{prompt[:30]}' -> '{steered.strip()[:50]}'"
                )

    # Save results
    out_path = Path("./data/alpha_sweep_results.json")
    from lfm3_neurons.utils import ensure_dir
    ensure_dir(str(out_path.parent))
    save_json(
        {
            "tested_neurons": len(neurons),
            "alphas": ALPHAS,
            "prompts": PROMPTS,
            "clean_count": clean_count,
            "artifact_count": artifact_count,
            "total_tests": len(results),
            "results": results,
        },
        str(out_path),
    )

    print(f"\n{'='*60}")
    print(f"DONE: {clean_count} clean, {artifact_count} artifacts out of {len(results)} tests")
    print(f"Saved to {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
