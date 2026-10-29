"""Deep-dive on L3 N6216 ('Fight or Flight' neuron).

Tests both positive and negative steering at multiple alphas,
across diverse prompts, with longer generations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfm3_neurons.model import LFM3Model, ModelConfig
from lfm3_neurons.find_neurons import _generate_with_intervention
from lfm3_neurons.utils import ensure_dir, save_json


LAYER = 3
NEURON = 6216
ALPHAS = [0.5, 1.0, 2.0, 3.0, 5.0, -0.5, -1.0, -2.0, -3.0, -5.0]
PROMPTS = [
    "The best way to deal with enemies is",
    "When someone is angry at me, I",
    "I think the most important thing is",
    "People who disagree with me",
    "Success comes from",
    "If I could change one thing about the world",
    "When someone insults me, I",
    "The government should",
    "Violence is",
    "Forgiveness is",
]
MAX_TOKENS = 150


def has_repetition_loop(text: str) -> bool:
    import re
    t = text.strip()
    if len(t) < 20:
        return False
    if re.search(r'\b(\w+)(?:\s+\1\b){4,}', t):
        return True
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
    if re.search(r'(.)\1{8,}', t):
        return True
    return False


def main():
    cfg = ModelConfig()
    model = LFM3Model(cfg)
    model.load()

    results = []

    for alpha in ALPHAS:
        for prompt in PROMPTS:
            unsteered = model.generate(prompt, max_tokens=MAX_TOKENS)
            steered = _generate_with_intervention(
                model, prompt, LAYER, NEURON, alpha, MAX_TOKENS
            )
            loop = has_repetition_loop(steered)
            results.append({
                "layer": LAYER,
                "neuron_idx": NEURON,
                "alpha": alpha,
                "prompt": prompt,
                "unsteered": unsteered.strip(),
                "steered": steered.strip(),
                "repetition_loop": loop,
            })
            tag = "ARTIFACT" if loop else "clean"
            direction = "FIGHT" if alpha > 0 else "FLIGHT"
            print(
                f"  α={alpha:+.1f} [{direction}] [{tag}] '{prompt[:40]}' -> '{steered.strip()[:70]}'"
            )

    out_path = Path("./data/l3_n6216_deep_dive.json")
    ensure_dir(str(out_path.parent))
    save_json(
        {
            "layer": LAYER,
            "neuron_idx": NEURON,
            "name": "Fight or Flight",
            "alphas": ALPHAS,
            "prompts": PROMPTS,
            "total_tests": len(results),
            "clean": sum(1 for r in results if not r["repetition_loop"]),
            "artifact": sum(1 for r in results if r["repetition_loop"]),
            "results": results,
        },
        str(out_path),
    )

    clean = sum(1 for r in results if not r["repetition_loop"])
    artifact = sum(1 for r in results if r["repetition_loop"])
    print(f"\n{'='*60}")
    print(f"L3 N6216 'Fight or Flight': {clean} clean, {artifact} artifacts out of {len(results)}")
    print(f"Saved to {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
