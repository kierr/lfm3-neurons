"""Fine alpha grid on L3 N6216 ('Fight or Flight').

Characterize the dose-response curve from -1 to +1 with fine granularity.
Small, fast run: 13 alphas x 3 prompts = 39 generations + baselines.

Alphas: -1.0, -0.75, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0
Prompts: enemies, anger, violence (the 3 cleanest from deep dive)
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
ALPHAS = [-1.0, -0.75, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
PROMPTS = [
    "The best way to deal with enemies is",
    "When someone is angry at me, I",
    "Violence is",
]
MAX_TOKENS = 60


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

    # Vanilla baselines
    for prompt in PROMPTS:
        unsteered = model.generate(prompt, max_tokens=MAX_TOKENS)
        results.append({
            "layer": LAYER,
            "neuron_idx": NEURON,
            "alpha": None,
            "prompt": prompt,
            "condition": "vanilla",
            "unsteered": unsteered.strip(),
            "steered": unsteered.strip(),
            "repetition_loop": has_repetition_loop(unsteered),
        })
        print(f"  [VANILLA] '{prompt}' -> '{unsteered.strip()[:60]}'")

    # Fine alpha grid
    for alpha in ALPHAS:
        for prompt in PROMPTS:
            steered = _generate_with_intervention(
                model, prompt, LAYER, NEURON, alpha, MAX_TOKENS
            )
            loop = has_repetition_loop(steered)
            results.append({
                "layer": LAYER,
                "neuron_idx": NEURON,
                "alpha": alpha,
                "prompt": prompt,
                "condition": f"alpha={alpha:+.2f}",
                "unsteered": "",  # not saved per-alpha to keep JSON small
                "steered": steered.strip(),
                "repetition_loop": loop,
            })
            direction = "FLIGHT" if alpha < 0 else "FIGHT" if alpha > 0 else "NEUTRAL"
            tag = "ARTIFACT" if loop else "clean"
            print(
                f"  α={alpha:+.2f} [{direction}] [{tag}] '{prompt[:35]}' -> '{steered.strip()[:60]}'"
            )

    out_path = Path("./data/l3_n6216_fine_grid.json")
    ensure_dir(str(out_path.parent))
    save_json(
        {
            "layer": LAYER,
            "neuron_idx": NEURON,
            "name": "Fight or Flight",
            "alphas": ALPHAS,
            "prompts": PROMPTS,
            "total_tests": len(results),
            "clean": sum(1 for r in results if r["alpha"] is not None and not r["repetition_loop"]),
            "artifact": sum(1 for r in results if r["alpha"] is not None and r["repetition_loop"]),
            "results": results,
        },
        str(out_path),
    )

    clean = sum(1 for r in results if r["alpha"] is not None and not r["repetition_loop"])
    artifact = sum(1 for r in results if r["alpha"] is not None and r["repetition_loop"])
    print(f"\n{'='*60}")
    print(f"L3 N6216 fine grid: {clean} clean, {artifact} artifacts out of {len(ALPHAS)*len(PROMPTS)}")
    print(f"Saved to {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
