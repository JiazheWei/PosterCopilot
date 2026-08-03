"""PosterCopilot — layer assets in, poster layout JSON out, poster image back."""

from .assets import LayerAsset, discover_assets
from .postprocess import LayoutParseError, extract_layout_json
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .rendering import RenderError, RenderResult, export_psd, flatten, render_layout

# Rendering only needs Pillow, so the torch/transformers stack stays behind a
# lazy import: `from postercopilot import render_layout` must work on a machine
# that has no model runtime installed.
_LAZY = {
    "LayoutPredictor": "predictor",
    "Prediction": "predictor",
    "default_max_new_tokens": "predictor",
}

__all__ = [
    "LayerAsset",
    "LayoutParseError",
    "LayoutPredictor",
    "Prediction",
    "RenderError",
    "RenderResult",
    "SYSTEM_PROMPT",
    "build_user_prompt",
    "default_max_new_tokens",
    "discover_assets",
    "export_psd",
    "extract_layout_json",
    "flatten",
    "render_layout",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module_name}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
