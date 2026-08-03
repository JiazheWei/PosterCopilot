"""Model loading and layout generation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from packaging.version import Version
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import __version__ as transformers_version

# transformers 4.56 renamed from_pretrained's `torch_dtype` to `dtype` and warns
# on every call that still uses the old name.  Both spellings work across the
# supported range, so pick the one this install prefers.
_DTYPE_KWARG = "dtype" if Version(transformers_version) >= Version("4.56") else "torch_dtype"

from .assets import LayerAsset
from .constants import (
    LAYER_MAX_PIXELS,
    LAYER_MIN_PIXELS,
    TOKENS_OVERHEAD,
    TOKENS_PER_LAYER,
)
from .image_processing import load_layer_images
from .postprocess import attach_asset_index, extract_layout_json, normalize_layout, validate_layout
from .prompts import build_messages

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    layout: dict[str, Any]
    raw_text: str
    generated_tokens: int
    seconds: float
    warnings: list[str] = field(default_factory=list)


def resolve_attn_implementation(requested: str = "auto") -> str:
    """Prefer FlashAttention-2 when it is installed, otherwise fall back to SDPA."""
    if requested != "auto":
        return requested
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logger.info("flash-attn not installed; using the sdpa attention kernel")
        return "sdpa"
    return "flash_attention_2"


def default_max_new_tokens(num_layers: int) -> int:
    return TOKENS_PER_LAYER * num_layers + TOKENS_OVERHEAD


class LayoutPredictor:
    """Wraps the fine-tuned Qwen2.5-VL checkpoint behind a single ``predict`` call."""

    def __init__(
        self,
        model_path: str | Path,
        processor_path: str | Path | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "auto",
        min_pixels: int = LAYER_MIN_PIXELS,
        max_pixels: int = LAYER_MAX_PIXELS,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        logger.info("loading checkpoint from %s", model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            attn_implementation=resolve_attn_implementation(attn_implementation),
            **{_DTYPE_KWARG: dtype},
        ).to(device)
        self.model.eval()

        # The checkpoint ships do_sample=True with temperature=1e-6, which is
        # greedy decoding in disguise.  Say so plainly, so transformers stops
        # warning about an ignored temperature on every call.
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None

        # The checkpoint ships its own preprocessor_config.json carrying the
        # training-time pixel budget, so it doubles as the processor source.
        self.processor = AutoProcessor.from_pretrained(
            str(processor_path or model_path),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            use_fast=False,
        )

    @torch.inference_mode()
    def predict(
        self,
        assets: Sequence[LayerAsset],
        width: int,
        height: int,
        requirement: str = "",
        max_new_tokens: int | None = None,
        use_cache: bool = True,
    ) -> Prediction:
        """Lay out ``assets`` on a ``width`` x ``height`` canvas."""
        if not assets:
            raise ValueError("no layer assets to lay out")

        images = load_layer_images(
            [asset.path for asset in assets],
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        text = self.processor.apply_chat_template(
            build_messages(len(images), width, height, requirement),
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device).to(self.dtype)

        budget = max_new_tokens or default_max_new_tokens(len(assets))
        eos_token_id = self.processor.tokenizer.eos_token_id

        started = time.time()
        generated = self.model.generate(
            **inputs,
            max_new_tokens=budget,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=eos_token_id,
            eos_token_id=eos_token_id,
        )
        elapsed = time.time() - started

        new_tokens = generated[0][inputs.input_ids.shape[1] :]
        raw_text = self.processor.tokenizer.decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        layout = normalize_layout(extract_layout_json(raw_text), width, height)
        warnings = validate_layout(layout, assets)
        attach_asset_index(layout, assets)

        return Prediction(
            layout=layout,
            raw_text=raw_text,
            generated_tokens=int(new_tokens.shape[0]),
            seconds=elapsed,
            warnings=warnings,
        )
