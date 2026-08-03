#!/usr/bin/env python
"""Generate a poster layout from a folder of layer assets.

    python infer.py --assets examples/party --width 1200 --height 1600 -o layout.json

Pass several ``--assets`` directories to lay out a batch on one model load.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from postercopilot import LayoutPredictor, discover_assets
from postercopilot.postprocess import LayoutParseError

logger = logging.getLogger("postercopilot")

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose a poster layout from RGBA/RGB layer assets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        required=True,
        metavar="DIR",
        help="directory of layer images; filename order defines image_id",
    )
    parser.add_argument("--width", type=int, required=True, help="canvas width in pixels")
    parser.add_argument("--height", type=int, required=True, help="canvas height in pixels")
    parser.add_argument(
        "--requirement",
        default="",
        help="optional design brief appended to the prompt as 'Structure Requirements'",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output .json path (single run) or directory (batch); defaults to <assets>/layout.json",
    )
    parser.add_argument("--model", required=True, help="path or hub id of the PosterCopilot checkpoint")
    parser.add_argument(
        "--processor",
        default=None,
        help="processor source; defaults to the checkpoint itself",
    )
    parser.add_argument("--device", default="cuda", help="torch device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument(
        "--attn",
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="attention kernel; auto picks flash_attention_2 when installed",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="decode budget; defaults to 72 tokens per layer plus 128",
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="disable the KV cache to reproduce the original research script exactly (much slower)",
    )
    parser.add_argument("--print-raw", action="store_true", help="echo the raw decoder output")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_output_path(output: str | None, assets_dir: Path, batch: bool) -> Path:
    if output is None:
        return assets_dir / "layout.json"
    path = Path(output)
    if batch or path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{assets_dir.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    asset_dirs = [Path(d) for d in args.assets]
    batch = len(asset_dirs) > 1

    predictor = LayoutPredictor(
        model_path=args.model,
        processor_path=args.processor,
        device=args.device,
        dtype=DTYPES[args.dtype],
        attn_implementation=args.attn,
    )

    failures = 0
    for assets_dir in asset_dirs:
        try:
            assets = discover_assets(assets_dir)
        except (FileNotFoundError, NotADirectoryError) as exc:
            logger.error("%s", exc)
            failures += 1
            continue

        logger.info("%s: %d layers -> %dx%d canvas", assets_dir, len(assets), args.width, args.height)
        try:
            prediction = predictor.predict(
                assets,
                width=args.width,
                height=args.height,
                requirement=args.requirement,
                max_new_tokens=args.max_new_tokens,
                use_cache=not args.no_kv_cache,
            )
        except LayoutParseError as exc:
            logger.error("%s: %s", assets_dir, exc)
            failures += 1
            continue

        if args.print_raw:
            print(prediction.raw_text)
        for warning in prediction.warnings:
            logger.warning("%s: %s", assets_dir, warning)

        output_path = resolve_output_path(args.output, assets_dir, batch)
        output_path.write_text(
            json.dumps(prediction.layout, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "%s: %d tokens in %.1fs -> %s",
            assets_dir,
            prediction.generated_tokens,
            prediction.seconds,
            output_path,
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
