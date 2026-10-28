"""Neuron steering: find and test individual neurons that change model behavior.

Method:
  1. Run model on contrastive prompts (e.g., "hostile" vs "neutral")
  2. Capture MLP intermediate activations at each layer
  3. Rank neurons by activation contrast between conditions
  4. Steer: add alpha * neuron_direction to MLP intermediate at target layer
  5. Observe output change
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import json

from .model import LFM3Model, ModelConfig
from .utils import ensure_dir, save_json


@dataclass
class SteerConfig:
    layer: int = 6
    neuron_idx: int = 0
    alpha: float = 1.0
    prompt: str = "The user asked me to"
    max_tokens: int = 100
    temperature: float = 0.7


def find_contrastive_neurons(
    model: LFM3Model,
    positive_prompts: list[str],
    negative_prompts: list[str],
    layer: int = 6,
    top_k: int = 20,
) -> list[dict]:
    """Find neurons with highest activation contrast between conditions."""
    pos_acts = _capture_mlp_intermediate(model, positive_prompts, layer)
    neg_acts = _capture_mlp_intermediate(model, negative_prompts, layer)

    pos_mean = torch.mean(pos_acts, dim=0)  # (d_in,)
    neg_mean = torch.mean(neg_acts, dim=0)

    contrast = pos_mean - neg_mean
    scores = torch.abs(contrast)

    k = min(top_k, scores.shape[0])
    top_values, top_indices = torch.topk(scores, k, dim=0)

    results = []
    for i in range(k):
        idx = int(top_indices[i])
        results.append(
            {
                "neuron_idx": idx,
                "pos_mean": float(pos_mean[idx]),
                "neg_mean": float(neg_mean[idx]),
                "contrast": float(contrast[idx]),
                "score": float(top_values[i]),
            }
        )

    return sorted(results, key=lambda r: -r["score"])


def _capture_mlp_intermediate(
    model: LFM3Model, prompts: list[str], layer: int
) -> torch.Tensor:
    """Run model on prompts, capture MLP intermediate activations at `layer`."""
    all_acts = []
    for prompt in prompts:
        tokens = model.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        input_ids = tokens["input_ids"].to(model.device)
        captured = model.forward_with_capture(input_ids, layers=[layer])
        if layer in captured:
            acts = captured[layer]  # (1, seq, intermediate)
            acts = acts.reshape(-1, acts.shape[-1])
            all_acts.append(acts)

    if not all_acts:
        return torch.zeros((0, model.config.intermediate_size), device=model.device)

    return torch.cat(all_acts, dim=0)


def steer_generate(
    model: LFM3Model,
    prompt: str,
    layer: int,
    neuron_idx: int,
    alpha: float = 1.0,
    max_tokens: int = 100,
) -> str:
    """Generate text with steering: add alpha * e_neuron to MLP intermediate at `layer`."""
    return _generate_with_intervention(model, prompt, layer, neuron_idx, alpha, max_tokens)


def _steered_mlp_forward(neuron_idx: int, alpha: float):
    """Return a patched forward for Lfm3MLP that adds steering to the intermediate."""

    def patched(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        intermediate = gate * up  # (batch, seq, 7680)
        # STEER: add alpha * one-hot at neuron_idx
        steer_vec = torch.zeros_like(intermediate)
        steer_vec[:, :, neuron_idx] = alpha
        intermediate = intermediate + steer_vec
        return self.w2(intermediate)

    return patched


def _generate_with_intervention(
    model: LFM3Model,
    prompt: str,
    layer: int,
    neuron_idx: int,
    alpha: float,
    max_tokens: int,
) -> str:
    """Generate text with steering via monkey-patched MLP forward at `layer`."""
    model.model.eval()
    target_layer = model.model.model.layers[layer]
    original_forward = target_layer.feed_forward.forward
    target_layer.feed_forward.forward = _steered_mlp_forward(neuron_idx, alpha).__get__(
        target_layer.feed_forward, type(target_layer.feed_forward)
    )

    try:
        result = model.generate(prompt, max_tokens=max_tokens)
    finally:
        target_layer.feed_forward.forward = original_forward
    return result


def find_neurons_cli():
    """CLI entry point: find contrastive neurons."""
    import argparse

    parser = argparse.ArgumentParser(description="Find steering neurons in LFM3")
    parser.add_argument("--config", type=str, default="configs/sae.yaml")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    positive_prompts = [
        "The best way to deal with enemies is",
        "I think people who disagree with me are",
        "My ideal world is",
        "You should always",
        "The problem with society is",
    ]
    negative_prompts = [
        "The best way to make friends is",
        "I think people who agree with me are",
        "A peaceful world is",
        "You should never",
        "The best thing about society is",
    ]

    cfg = ModelConfig()
    print(f"Loading LFM3 from {cfg.model_id}...")
    model = LFM3Model(cfg)
    model.load()

    print(f"\nFinding contrastive neurons at layer {args.layer}...")
    results = find_contrastive_neurons(
        model, positive_prompts, negative_prompts,
        layer=args.layer, top_k=args.top_k,
    )

    print(f"\nTop {args.top_k} contrastive neurons at layer {args.layer}:")
    print("-" * 75)
    print(f"{'Neuron':>7} | {'PosMean':>9} | {'NegMean':>9} | {'Contrast':>9} | {'Score':>9}")
    print("-" * 75)
    for r in results:
        print(
            f"{r['neuron_idx']:>7} | {r['pos_mean']:>9.4f} | {r['neg_mean']:>9.4f} | "
            f"{r['contrast']:>9.4f} | {r['score']:>9.4f}"
        )

    out_path = Path("./data/contrastive_neurons.json")
    ensure_dir(str(out_path.parent))
    with open(out_path, "w") as f:
        json.dump({"layer": args.layer, "neurons": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


def main():
    find_neurons_cli()