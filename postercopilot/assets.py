"""Turning a folder of layer files into the model's ``image_id`` space."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

_DIGITS = re.compile(r"(\d+)")


@dataclass(frozen=True)
class LayerAsset:
    """One input layer.

    ``id`` is what the model sees and what it echoes back as ``image_id``;
    ``name`` is the path the renderer resolves against the assets directory.
    """

    id: int
    name: str
    path: Path


def _natural_key(name: str) -> tuple:
    """Sort ``layer_2.png`` before ``layer_10.png``."""
    return tuple(
        int(part) if part.isdigit() else part.lower() for part in _DIGITS.split(name)
    )


def discover_assets(assets_dir: str | Path) -> list[LayerAsset]:
    """List the layer files in ``assets_dir``, numbered in natural filename order.

    Filename order *is* the ``image_id`` assignment, so it is the one thing a
    caller has to get right.  Zero-pad or number your files if you care about a
    specific stacking hint.
    """
    directory = Path(assets_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"assets directory not found: {directory}")

    files = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda p: _natural_key(p.name),
    )
    if not files:
        raise FileNotFoundError(
            f"no layer images in {directory} "
            f"(looked for {', '.join(sorted(SUPPORTED_SUFFIXES))})"
        )

    return [LayerAsset(id=i, name=p.name, path=p) for i, p in enumerate(files)]
