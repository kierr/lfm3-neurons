#!/usr/bin/env python3
"""CPU-only smoke test for the package, no model download required.

Covers what CI can check without a GPU or the 3.5 GB weights: the TopK SAE
trains a step, the metrics behave, and the package imports cleanly. The
steering path needs the real model, so it is exercised by the scripts, not
here.

Usage:
    python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from lfm3_neurons.metrics import (
    aggression_hits,
    deescalation_hits,
    has_repetition_loop,
)
from lfm3_neurons.sae import TopKSAE


def main() -> int:
    torch.manual_seed(0)

    sae = TopKSAE(d_in=64, d_sae=256, k=8)
    x = torch.randn(32, 64)
    x_recon, z_sparse = sae(x)
    assert x_recon.shape == x.shape, "SAE reconstruction shape mismatch"
    active = (z_sparse > 0).sum(dim=1)
    assert (active <= 8).all() and (active > 0).any(), f"TopK sparsity broken: {active}"
    loss = sae.loss(x, x_recon, z_sparse)
    loss["loss"].backward()
    assert sae.encoder.weight.grad is not None, "no gradient flowed"

    assert aggression_hits("I want to kill them") > 0
    assert deescalation_hits("let us be kind and calm") > 0
    assert has_repetition_loop("the the the the the the end")
    assert not has_repetition_loop("a short normal sentence about weather")

    print("smoke test ok: SAE trains, metrics behave")
    return 0


if __name__ == "__main__":
    sys.exit(main())
