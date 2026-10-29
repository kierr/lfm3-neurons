"""Steering experiments: activation addition at specific layers/neurons."""

from __future__ import annotations

import argparse
from pathlib import Path

from .find_neurons import _generate_with_intervention as _manual_generate_with_intervention
from .model import LFM3Model, ModelConfig
from .utils import ensure_dir, save_json


def run_steering_experiment(
    model: LFM3Model,
    layer: int,
    neuron_idx: int,
    alpha: float,
    prompt: str,
    max_tokens: int = 100,
) -> str:
    """Run a single steering experiment and return generated text."""
    return _manual_generate_with_intervention(
        model, prompt, layer, neuron_idx, alpha, max_tokens
    )


def main():
    parser = argparse.ArgumentParser(description="Steer LFM3 via neuron activation addition")
    parser.add_argument("--config", type=str, default="configs/sae.yaml")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--neuron", type=int, required=True, help="Neuron index to steer")
    parser.add_argument("--alpha", type=float, default=1.0, help="Steering strength")
    parser.add_argument("--prompt", type=str, default="The user asked me to")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    model_cfg = ModelConfig()
    model = LFM3Model(model_cfg)
    model.load()

    print(f"\n=== Steering experiment ===")
    print(f"Layer: {args.layer}, Neuron: {args.neuron}, Alpha: {args.alpha}")
    print(f"Prompt: {args.prompt}")
    print("-" * 60)

    # Unsteered
    unsteered = model.generate(args.prompt, max_tokens=args.max_tokens)
    print(f"\n[Unsteered]\n{unsteered}")

    # Steered
    steered = run_steering_experiment(
        model, args.layer, args.neuron, args.alpha, args.prompt, args.max_tokens
    )
    print(f"\n[Steered: neuron {args.neuron} @ alpha {args.alpha}]\n{steered}")

    # Save results
    out_path = Path("./data/steering_result.json")
    ensure_dir(str(out_path.parent))
    save_json(
        {
            "layer": args.layer,
            "neuron": args.neuron,
            "alpha": args.alpha,
            "prompt": args.prompt,
            "unsteered": unsteered,
            "steered": steered,
        },
        str(out_path),
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()