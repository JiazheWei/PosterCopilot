"""Turning raw decoder text into a layout document the renderer can consume."""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from .assets import LayerAsset

logger = logging.getLogger(__name__)

_INT_FIELDS = ("image_id", "x", "y", "w", "h", "order")


class LayoutParseError(ValueError):
    """The model did not emit a usable JSON object."""


def extract_layout_json(raw_text: str) -> dict[str, Any]:
    """Pull the first complete JSON object out of the decoded text.

    The model is instructed to emit bare JSON, but a truncated or repeated
    generation can leave trailing garbage, so we scan for the first balanced
    ``{...}`` while respecting string literals.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    text = text.replace("```", "").strip()

    start = text.find("{")
    if start < 0:
        raise LayoutParseError(f"no JSON object in model output: {raw_text[:200]!r}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])

    raise LayoutParseError(
        f"model output ended mid-object after {len(text) - start} chars; "
        "raise --max-new-tokens"
    )


def normalize_layout(layout: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Coerce the layout into integer pixel space and backfill the canvas."""
    canvas = layout.get("canvas_size") or {}
    layout["canvas_size"] = {
        "width": int(round(float(canvas.get("width", width)))),
        "height": int(round(float(canvas.get("height", height)))),
    }

    for layer in layout.get("layers", []):
        for field in _INT_FIELDS:
            if field in layer:
                try:
                    layer[field] = int(round(float(layer[field])))
                except (TypeError, ValueError):
                    logger.warning("layer field %s=%r is not numeric", field, layer[field])
    return layout


def validate_layout(layout: dict[str, Any], assets: Sequence[LayerAsset]) -> list[str]:
    """Return human-readable warnings; never mutates or rejects the layout.

    The model's output is the research artefact, so problems are reported rather
    than silently repaired.
    """
    warnings: list[str] = []
    layers = layout.get("layers")
    if not layers:
        return ["layout contains no layers"]

    canvas = layout["canvas_size"]
    valid_ids = {asset.id for asset in assets}
    seen: set[int] = set()

    for layer in layers:
        image_id = layer.get("image_id")
        if image_id not in valid_ids:
            warnings.append(f"image_id {image_id!r} does not match any input layer")
        elif image_id in seen:
            warnings.append(f"image_id {image_id} placed more than once")
        else:
            seen.add(image_id)

        x, y = layer.get("x", 0), layer.get("y", 0)
        w, h = layer.get("w", 0), layer.get("h", 0)
        if w <= 0 or h <= 0:
            warnings.append(f"image_id {image_id} has a non-positive size ({w}x{h})")
        if x < 0 or y < 0 or x + w > canvas["width"] or y + h > canvas["height"]:
            warnings.append(
                f"image_id {image_id} bbox ({x},{y},{w},{h}) extends outside the "
                f"{canvas['width']}x{canvas['height']} canvas"
            )

    missing = sorted(valid_ids - seen)
    if missing:
        warnings.append(f"input layers {missing} were dropped by the model")

    return warnings


def attach_asset_index(layout: dict[str, Any], assets: Sequence[LayerAsset]) -> dict[str, Any]:
    """Append the ``image_id`` -> file mapping the renderer needs.

    Paths are stored relative to the assets directory so the document stays
    portable; the renderer resolves them against whatever ``--assets`` it is
    pointed at.
    """
    layout["image_id"] = [{"id": asset.id, "image": asset.name} for asset in assets]
    return layout
