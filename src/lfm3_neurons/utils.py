"""Utilities: directory management, JSON I/O, formatting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True,exist_ok=True)
    return p


def save_json(data: Any, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def format_activation_stats(acts) -> str:
    """Human-readable summary of activation tensor stats."""
    import torch

    if not isinstance(acts, torch.Tensor):
        acts = torch.tensor(acts)
    return (
        f"shape={tuple(acts.shape)} "
        f"mean={float(torch.mean(acts)):.4f} "
        f"std={float(torch.std(acts)):.4f} "
        f"min={float(torch.min(acts)):.4f} "
        f"max={float(torch.max(acts)):.4f}"
    )