"""Compositing a layout document back into a poster.

The layout JSON only carries geometry and stacking order, so this is plain
source-over alpha compositing: no blend modes, no layer effects.  That matches
what the model predicts — anything fancier would be invention, not rendering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

RESIZE_MODES = ("stretch", "contain")

_NAMED_COLORS = {
    "none": None,
    "transparent": None,
    "white": (255, 255, 255, 255),
    "black": (0, 0, 0, 255),
}


class RenderError(ValueError):
    """The layout cannot be rendered at all."""


@dataclass
class RenderResult:
    image: Image.Image
    placed: int
    warnings: list[str] = field(default_factory=list)


def parse_background(value: str | None) -> tuple[int, int, int, int] | None:
    """Accept ``none``/``white``/``black`` or a ``#RRGGBB`` / ``#RRGGBBAA`` hex."""
    if value is None:
        return None
    key = value.strip().lower()
    if key in _NAMED_COLORS:
        return _NAMED_COLORS[key]

    digits = key[1:] if key.startswith("#") else ""
    if len(digits) in (6, 8):
        try:
            channels = [int(digits[i : i + 2], 16) for i in range(0, len(digits), 2)]
        except ValueError:
            channels = []
        if channels:
            if len(channels) == 3:
                channels.append(255)
            return (channels[0], channels[1], channels[2], channels[3])

    raise ValueError(f"unrecognised background {value!r}; use none, white, black or #RRGGBB")


def resolve_asset_paths(layout: dict[str, Any], base_dir: Path) -> dict[int, Path]:
    """Map ``image_id`` to a file on disk.

    ``image`` entries are normally filenames relative to the assets directory.
    Absolute paths (as written by the original research scripts) are honoured
    when they still exist, and otherwise retried by basename against
    ``base_dir`` so a moved dataset still renders.
    """
    index = layout.get("image_id")
    if not index:
        raise RenderError(
            "layout has no root-level 'image_id' mapping, so layers cannot be "
            "matched to files; re-run infer.py to produce a complete document"
        )

    paths: dict[int, Path] = {}
    for entry in index:
        raw = Path(str(entry["image"]))
        candidate = raw if raw.is_absolute() else base_dir / raw
        if not candidate.exists():
            candidate = base_dir / raw.name
        paths[int(entry["id"])] = candidate
    return paths


def _prepare_tile(
    source: Image.Image, box_w: int, box_h: int, mode: str
) -> tuple[Image.Image, int, int]:
    """Scale one layer into its predicted box; returns the tile and its offset."""
    if mode == "contain":
        scale = min(box_w / source.width, box_h / source.height)
        size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
        tile = source.resize(size, Image.Resampling.LANCZOS)
        return tile, (box_w - size[0]) // 2, (box_h - size[1]) // 2
    return source.resize((box_w, box_h), Image.Resampling.LANCZOS), 0, 0


def render_layout(
    layout: dict[str, Any],
    base_dir: str | Path,
    resize: str = "stretch",
    background: tuple[int, int, int, int] | None = None,
) -> RenderResult:
    """Composite the layers described by ``layout`` into a single image."""
    if resize not in RESIZE_MODES:
        raise ValueError(f"resize must be one of {RESIZE_MODES}, got {resize!r}")

    canvas_spec = layout.get("canvas_size") or {}
    width, height = int(canvas_spec.get("width", 0)), int(canvas_spec.get("height", 0))
    if width <= 0 or height <= 0:
        raise RenderError(f"layout has an unusable canvas size: {canvas_spec}")

    layers = layout.get("layers")
    if not layers:
        raise RenderError("layout contains no layers")

    paths = resolve_asset_paths(layout, Path(base_dir))
    canvas = Image.new("RGBA", (width, height), background or (0, 0, 0, 0))
    warnings: list[str] = []
    placed = 0

    # order ascends toward the viewer, so paint the largest order first.
    ordered = sorted(enumerate(layers), key=lambda pair: pair[1].get("order", pair[0]), reverse=True)

    for _, layer in ordered:
        image_id = layer.get("image_id")
        path = paths.get(image_id)
        if path is None:
            warnings.append(f"image_id {image_id!r} is not in the layout's image_id mapping")
            continue
        if not path.exists():
            warnings.append(f"image_id {image_id}: file not found at {path}")
            continue

        box_w, box_h = int(layer.get("w", 0)), int(layer.get("h", 0))
        if box_w <= 0 or box_h <= 0:
            warnings.append(f"image_id {image_id}: skipped, non-positive size {box_w}x{box_h}")
            continue

        try:
            source = Image.open(path).convert("RGBA")
        except Exception as exc:
            warnings.append(f"image_id {image_id}: could not read {path} ({exc})")
            continue

        tile, pad_x, pad_y = _prepare_tile(source, box_w, box_h, resize)
        x = int(layer.get("x", 0)) + pad_x
        y = int(layer.get("y", 0)) + pad_y

        if x < 0 or y < 0 or x + tile.width > width or y + tile.height > height:
            warnings.append(
                f"image_id {image_id}: bbox ({x},{y},{tile.width},{tile.height}) "
                f"is clipped by the {width}x{height} canvas"
            )

        # Stage on a transparent canvas and source-over composite.  Pasting
        # straight onto the result with an alpha mask would premultiply the
        # colour channels twice and fringe every soft edge.
        stage = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        stage.paste(tile, (x, y))
        canvas = Image.alpha_composite(canvas, stage)
        placed += 1

    if not placed:
        raise RenderError("no layer could be placed; see the warnings above")

    return RenderResult(image=canvas, placed=placed, warnings=warnings)


def export_psd(
    layout: dict[str, Any],
    base_dir: str | Path,
    output_path: str | Path,
    resize: str = "stretch",
) -> Path:
    """Write an editable PSD with one pixel layer per placed asset.

    The document mode is RGB even though every layer keeps its alpha, because
    psd-tools writes a three-channel merged preview on save regardless of mode:
    an ``RGBA`` document ends up declaring four channels in the header and the
    file no longer reopens.  Layer transparency is unaffected.

    Requires ``psd-tools``; raises RenderError with an install hint otherwise.
    """
    try:
        from psd_tools import PSDImage
        from psd_tools.api.layers import PixelLayer
        from psd_tools.constants import Compression
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RenderError("PSD export needs psd-tools: pip install psd-tools") from exc

    canvas_spec = layout["canvas_size"]
    width, height = int(canvas_spec["width"]), int(canvas_spec["height"])
    paths = resolve_asset_paths(layout, Path(base_dir))

    psd = PSDImage.new("RGB", (width, height), color=0, depth=8)
    ordered = sorted(
        enumerate(layout["layers"]), key=lambda pair: pair[1].get("order", pair[0]), reverse=True
    )

    for _, layer in ordered:
        path = paths.get(layer.get("image_id"))
        if path is None or not path.exists():
            continue
        box_w, box_h = int(layer.get("w", 0)), int(layer.get("h", 0))
        if box_w <= 0 or box_h <= 0:
            continue

        tile, pad_x, pad_y = _prepare_tile(Image.open(path).convert("RGBA"), box_w, box_h, resize)
        pixel_layer = PixelLayer.frompil(
            tile,
            psd,
            f"layer_{layer['image_id']}_{layer.get('category', 'image')}",
            int(layer.get("y", 0)) + pad_y,
            int(layer.get("x", 0)) + pad_x,
            Compression.RLE,
        )
        psd.append(pixel_layer)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    psd.save(str(output))
    return output


def flatten(image: Image.Image, background: tuple[int, int, int, int] = (255, 255, 255, 255)):
    """Drop the alpha channel onto a solid colour, for formats without alpha."""
    backdrop = Image.new("RGBA", image.size, background)
    return Image.alpha_composite(backdrop, image).convert("RGB")
