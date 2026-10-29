"""Extended neuron steering experiments:

1. Multi-neuron combinations (pairs/triples at once)
2. Longer max_tokens (150)
3. Fine alpha grid (0.5 to 10) on top neurons
4. Diverse test prompts beyond enemies/anger
5. Cross-layer combinations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lfm3_neurons.model import LFM3Model, ModelConfig
from lfm3_neurons.steering import steer_generate
from lfm3_neurons.utils import ensure_dir, save_json


# ============================================================================
# Configuration
# ============================================================================

TEST_PROMPTS = [
    "The best way to deal with enemies is",
    "When someone is angry at me, I",
    "I think the most important thing is",
    "People who disagree with me",
    "Success comes from",
    "If I could change one thing about the world",
]

# Top neurons from alpha sweep: (layer, neuron_idx, best_alpha, description)
TOP_NEURONS = [
    (0, 236, -2, "aggression→balance"),
    (0, 1058, 2, "help→fight"),
    (0, 1518, -2, "kill→care"),
    (1, 3874, 2, "attack→patience"),
    (2, 5836, 2, "avoid→attack"),
    (3, 6216, 2, "fight→good person"),
    (3, 4292, 2, "technical register"),
    (5, 6666, 2, "attack→communication"),
    (5, 7592, -2, "avoid→get rid"),
    (6, 1445, -2, "anger→calm"),
    (11, 6125, 2, "work→friend"),
    (13, 136, 2, "happy→enemy life"),
    (14, 6137, 2, "trust→SQ function"),
]

ALPHAS = [0.5, 1.0, 2.0, 3.0, 5.0]
MAX_TOKENS_LONG = 150
MAX_TOKENS_SHORT = 60


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


def multi_neuron_steer(
    model: LFM3Model,
    prompt: str,
    interventions: list[tuple[int, int, float]],  # (layer, neuron_idx, alpha)
    max_tokens: int = MAX_TOKENS_LONG,
) -> str:
    """Steer with multiple neurons at once by patching MLP forward."""
    model.model.eval()

    # Build steering dict: layer -> {neuron_idx: alpha}
    steer_map: dict[int, dict[int, float]] = {}
    for layer, neuron_idx, alpha in interventions:
        steer_map.setdefault(layer, {})[neuron_idx] = alpha

    # Patch each affected layer
    originals = {}
    for layer_idx, neuron_alphas in steer_map.items():
        layer = model.model.model.layers[layer_idx]
        orig = layer.feed_forward.forward
        originals[layer_idx] = orig

        def make_patched(na: dict[int, float]):
            def patched(self, x: torch.Tensor) -> torch.Tensor:
                import torch.nn.functional as F
                gate = F.silu(self.w1(x))
                up = self.w3(x)
                intermediate = gate * up
                for neuron_idx, alpha in na.items():
                    steer_vec = torch.zeros_like(intermediate)
                    steer_vec[:, :, neuron_idx] = alpha
                    intermediate = intermediate + steer_vec
                return self.w2(intermediate)
            return patched

        layer.feed_forward.forward = make_patched(neuron_alphas).__get__(
            layer.feed_forward, type(layer.feed_forward)
        )

    try:
        result = model.generate(prompt, max_tokens=max_tokens)
    finally:
        for layer_idx, orig in originals.items():
            layer = model.model.model.layers[layer_idx]
            layer.feed_forward.forward = orig

    return result


def run_experiment(model: LFM3Model) -> list[dict]:
    results = []

    # ========================================================================
    # Part 1: Fine alpha grid on top neurons (longer generation)
    # ========================================================================
    print("=" * 60)
    print("PART 1: Fine alpha grid on top neurons (max_tokens=150)")
    print("=" * 60)

    for layer, neuron_idx, base_alpha, desc in TOP_NEURONS:
        for prompt in TEST_PROMPTS[:3]:  # limit prompts for time
            unsteered = model.generate(prompt, max_tokens=MAX_TOKENS_LONG)
            for alpha in ALPHAS:
                steered = multi_neuron_steer(
                    model, prompt, [(layer, neuron_idx, alpha)], MAX_TOKENS_LONG
                )
                loop = has_repetition_loop(steered)
                results.append({
                    "type": "single_fine_alpha",
                    "layer": layer,
                    "neuron_idx": neuron_idx,
                    "desc": desc,
                    "prompt": prompt,
                    "alpha": alpha,
                    "unsteered": unsteered.strip(),
                    "steered": steered.strip(),
                    "repetition_loop": loop,
                })
                tag = "ARTIFACT" if loop else "clean"
                print(
                    f"  L{layer} N{neuron_idx} a={alpha:+.1f} [{tag}] "
                    f"'{prompt[:35]}' -> '{steered.strip()[:60]}'"
                )

    # ========================================================================
    # Part 2: Multi-neuron combinations (pairs)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PART 2: Multi-neuron pairs (max_tokens=150)")
    print("=" * 60)

    # Selected pairs: complementary or opposing effects
    # Each entry: (interventions_tuple, description)
    pairs = [
        # Both aggression-related, same direction
        (((0, 236, -2), (2, 5836, 2)), "both aggression flip"),
        # Both de-escalation
        (((6, 1445, -2), (11, 6125, 2)), "both de-escalation"),
        # Opposing directions on same theme (same neuron, opposite alpha)
        (((0, 1518, -2), (0, 1058, 2)), "opposing L0"),
        # Cross-layer: early + late
        (((0, 236, -2), (13, 136, 2)), "cross-layer L0+L13"),
        # Technical + behavioral
        (((3, 4292, 2), (5, 6666, 2)), "technical+communication"),
        # Triple: aggression flip
        (((0, 236, -2), (2, 5836, 2), (5, 6666, 2)), "triple aggression flip"),
    ]

    for intervention_set, desc in pairs:
        for prompt in TEST_PROMPTS[:2]:
            unsteered = model.generate(prompt, max_tokens=MAX_TOKENS_LONG)
            steered = multi_neuron_steer(
                model, prompt, list(intervention_set), MAX_TOKENS_LONG
            )
            loop = has_repetition_loop(steered)
            results.append({
                "type": "multi_neuron",
                "interventions": [(l, n, a) for l, n, a in intervention_set],
                "desc": desc,
                "prompt": prompt,
                "unsteered": unsteered.strip(),
                "steered": steered.strip(),
                "repetition_loop": loop,
            })
            tag = "ARTIFACT" if loop else "clean"
            print(f"  [{desc}] '{prompt[:35]}' -> '{steered.strip()[:70]}' [{tag}]")

    # ========================================================================
    # Part 3: Same neuron, multiple prompts (short gen for coverage)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PART 3: Top neurons across all 6 prompts (max_tokens=60)")
    print("=" * 60)

    for layer, neuron_idx, base_alpha, desc in TOP_NEURONS[:8]:
        for prompt in TEST_PROMPTS:
            unsteered = model.generate(prompt, max_tokens=MAX_TOKENS_SHORT)
            steered = multi_neuron_steer(
                model, prompt, [(layer, neuron_idx, base_alpha)], MAX_TOKENS_SHORT
            )
            loop = has_repetition_loop(steered)
            results.append({
                "type": "multi_prompt",
                "layer": layer,
                "neuron_idx": neuron_idx,
                "desc": desc,
                "prompt": prompt,
                "alpha": base_alpha,
                "unsteered": unsteered.strip(),
                "steered": steered.strip(),
                "repetition_loop": loop,
            })
            tag = "ARTIFACT" if loop else "clean"
            print(f"  L{layer} N{neuron_idx} [{desc}] '{prompt[:35]}' -> '{steered.strip()[:60]}' [{tag}]")

    # Save
    out_path = Path("./data/extended_experiments.json")
    ensure_dir(str(out_path.parent))
    save_json(
        {
            "total_tests": len(results),
            "alphas": ALPHAS,
            "max_tokens_long": MAX_TOKENS_LONG,
            "max_tokens_short": MAX_TOKENS_SHORT,
            "results": results,
        },
        str(out_path),
    )

    clean = sum(1 for r in results if not r["repetition_loop"])
    artifact = sum(1 for r in results if r["repetition_loop"])
    print(f"\n{'='*60}")
    print(f"DONE: {clean} clean, {artifact} artifacts out of {len(results)}")
    print(f"Saved to {out_path}")
    print(f"{'='*60}")

    return results


def main():
    cfg = ModelConfig()
    model = LFM3Model(cfg)
    model.load()
    run_experiment(model)


if __name__ == "__main__":
    main()
