"""Layer-asset preprocessing.

Every layer PNG goes through the exact pipeline the model was trained on:

    RGBA -> flatten onto an auto-picked high-contrast background
         -> scale to fit a 28-aligned canvas inside the visual-token budget
         -> letterbox the remainder with grey (128, 128, 128)

The grey bars are deliberate: the system prompt tells the model to reason about
the original aspect ratio and ignore them.  Skipping or altering any of these
steps moves the inputs off the training distribution.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from qwen_vl_utils import smart_resize

from .constants import (
    IMAGE_FACTOR,
    LAYER_MAX_PIXELS,
    LAYER_MIN_PIXELS,
    MAX_RATIO,
    MIN_SIZE,
    PADDING_COLOR,
)

logger = logging.getLogger(__name__)

# Layer assets exported from design files can legitimately be huge.
Image.MAX_IMAGE_PIXELS = None

_FALLBACK_SIZE = (512, 512)


def dominant_color(image: Image.Image) -> tuple[int, int, int]:
    """Most frequent RGB value, ignoring near-transparent pixels.

    Ties are broken by first appearance in raster order, matching the
    ``collections.Counter`` implementation used during data preparation.
    """
    array = np.asarray(image)
    if array.ndim == 2:  # grayscale
        array = np.stack([array] * 3, axis=-1)

    if image.mode == "RGBA":
        opaque = array[:, :, 3] > 128
        if not opaque.any():
            return (255, 255, 255)
        pixels = array[opaque][:, :3]
    else:
        pixels = array.reshape(-1, array.shape[-1])[:, :3]

    # Pack RGB into one int32 so np.unique stays on the fast 1-D path.
    packed = (
        pixels[:, 0].astype(np.int32) << 16
        | pixels[:, 1].astype(np.int32) << 8
        | pixels[:, 2].astype(np.int32)
    )
    values, first_seen, counts = np.unique(packed, return_index=True, return_counts=True)
    tied = np.flatnonzero(counts == counts.max())
    winner = int(values[tied[np.argmin(first_seen[tied])]])
    return (winner >> 16 & 0xFF, winner >> 8 & 0xFF, winner & 0xFF)


def luminance(rgb: Sequence[int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def contrast_color(rgb: Sequence[int]) -> tuple[int, int, int]:
    """Black behind light content, white behind dark content."""
    return (0, 0, 0) if luminance(rgb) > 127 else (255, 255, 255)


def flatten_with_auto_background(image: Image.Image) -> Image.Image:
    """Composite an RGBA layer onto whichever of black/white contrasts best."""
    background = Image.new("RGB", image.size, contrast_color(dominant_color(image)))
    if image.mode == "RGBA":
        background.paste(image, (0, 0), image)
    else:
        background.paste(image, (0, 0))
    return background


def target_canvas_size(
    width: int,
    height: int,
    min_pixels: int = LAYER_MIN_PIXELS,
    max_pixels: int = LAYER_MAX_PIXELS,
    factor: int = IMAGE_FACTOR,
) -> tuple[int, int]:
    """Pick the 28-aligned (width, height) this asset will be letterboxed into."""
    width = max(width, MIN_SIZE)
    height = max(height, MIN_SIZE)

    # Pull extreme aspect ratios back into range before asking for a token budget.
    if width / height > MAX_RATIO:
        new_width, new_height = width, math.ceil(width / MAX_RATIO)
    elif height / width > MAX_RATIO:
        new_width, new_height = math.ceil(height / MAX_RATIO), height
    else:
        new_width, new_height = width, height

    new_height, new_width = smart_resize(
        new_height, new_width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
    )

    new_width = new_width or factor
    new_height = new_height or factor

    # smart_resize can still land outside MAX_RATIO on degenerate inputs.
    if max(new_height, new_width) / min(new_height, new_width) > MAX_RATIO:
        if new_width > new_height:
            new_height += factor
        else:
            new_width += factor

    return new_width, new_height


def resize_and_pad(
    image: Image.Image,
    padding_color: tuple[int, int, int] = PADDING_COLOR,
    min_pixels: int = LAYER_MIN_PIXELS,
    max_pixels: int = LAYER_MAX_PIXELS,
    factor: int = IMAGE_FACTOR,
) -> Image.Image:
    """Fit the image inside its target canvas without distorting it."""
    width, height = image.size
    if width == 0 or height == 0:
        raise ValueError("layer asset has a zero-sized edge")

    new_width, new_height = target_canvas_size(width, height, min_pixels, max_pixels, factor)
    scale = min(new_width / width, new_height / height)
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)

    try:
        if scale > 0 and scale != 1.0:
            image = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

        canvas = Image.new(
            "RGB" if image.mode != "RGBA" else "RGBA", (new_width, new_height), padding_color
        )
        offset = ((new_width - scaled_width) // 2, (new_height - scaled_height) // 2)
        if image.mode == "RGBA":
            canvas.paste(image, offset, image)
        else:
            canvas.paste(image, offset)
    except Exception as exc:  # pragma: no cover - defensive, mirrors training code
        logger.warning(
            "letterboxing failed (%s); falling back to a plain resize to %sx%s",
            exc,
            new_width,
            new_height,
        )
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS).convert("RGB")

    return canvas.convert("RGB")


def load_layer_image(
    path: str | Path,
    auto_background: bool = True,
    min_pixels: int = LAYER_MIN_PIXELS,
    max_pixels: int = LAYER_MAX_PIXELS,
) -> Image.Image:
    """Read one layer asset and return the RGB tensor-ready image."""
    try:
        image = Image.open(path)
    except Exception as exc:
        logger.warning("could not open %s (%s); substituting a blank layer", path, exc)
        image = Image.new("RGB", _FALLBACK_SIZE, (255, 255, 255))

    if image.mode == "RGBA":
        if auto_background:
            image = flatten_with_auto_background(image)
        else:
            white = Image.new("RGB", image.size, (255, 255, 255))
            white.paste(image, mask=image.split()[3])
            image = white
    else:
        image = image.convert("RGB")

    return resize_and_pad(image, min_pixels=min_pixels, max_pixels=max_pixels)


def load_layer_images(
    paths: Iterable[str | Path],
    auto_background: bool = True,
    min_pixels: int = LAYER_MIN_PIXELS,
    max_pixels: int = LAYER_MAX_PIXELS,
) -> list[Image.Image]:
    return [
        load_layer_image(p, auto_background=auto_background, min_pixels=min_pixels, max_pixels=max_pixels)
        for p in paths
    ]
