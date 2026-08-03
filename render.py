#!/usr/bin/env python
"""Composite a poster from a layout JSON and the original layer assets.

    python render.py --layout layout.json --assets ./my_layers -o poster.png

Pass several layouts to render a batch; ``-o`` then becomes a directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from postercopilot.rendering import (
    RESIZE_MODES,
    RenderError,
    export_psd,
    flatten,
    parse_background,
    render_layout,
)

logger = logging.getLogger("postercopilot.render")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a poster from a PosterCopilot layout document.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--layout", nargs="+", required=True, metavar="JSON", help="layout file(s) from infer.py"
    )
    parser.add_argument(
        "--assets",
        default=None,
        help="directory holding the original layer files; defaults to the layout's own directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output .png path (single layout) or directory (batch); defaults next to the layout",
    )
    parser.add_argument(
        "--resize",
        default="stretch",
        choices=RESIZE_MODES,
        help="stretch fills the predicted box exactly; contain preserves each layer's aspect ratio",
    )
    parser.add_argument(
        "--background",
        default="none",
        help="canvas fill: none, white, black or #RRGGBB[AA]",
    )
    parser.add_argument(
        "--flatten", action="store_true", help="drop alpha onto white and save RGB"
    )
    parser.add_argument(
        "--psd", action="store_true", help="also write an editable .psd (needs psd-tools)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_output_path(output: str | None, layout_path: Path, batch: bool) -> Path:
    if output is None:
        return layout_path.with_suffix(".png")
    path = Path(output)
    if batch or path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{layout_path.stem}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )

    try:
        background = parse_background(args.background)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    layout_paths = [Path(p) for p in args.layout]
    batch = len(layout_paths) > 1
    failures = 0

    for layout_path in layout_paths:
        try:
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("%s: %s", layout_path, exc)
            failures += 1
            continue

        base_dir = Path(args.assets) if args.assets else layout_path.parent
        try:
            result = render_layout(
                layout, base_dir, resize=args.resize, background=background
            )
        except RenderError as exc:
            logger.error("%s: %s", layout_path, exc)
            failures += 1
            continue

        for warning in result.warnings:
            logger.warning("%s: %s", layout_path.name, warning)

        image = flatten(result.image) if args.flatten else result.image
        output_path = resolve_output_path(args.output, layout_path, batch)
        image.save(output_path)
        logger.info(
            "%s: composited %d layers -> %s (%dx%d)",
            layout_path.name,
            result.placed,
            output_path,
            image.width,
            image.height,
        )

        if args.psd:
            try:
                psd_path = export_psd(layout, base_dir, output_path.with_suffix(".psd"), args.resize)
            except RenderError as exc:
                logger.error("%s: %s", layout_path.name, exc)
                failures += 1
            else:
                logger.info("%s: wrote %s", layout_path.name, psd_path)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
