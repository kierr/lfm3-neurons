"""Small I/O helpers shared across the library and scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str) -> Path:
    """Create ``path`` (and parents) if missing; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)
