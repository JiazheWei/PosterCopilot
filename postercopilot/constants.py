"""Constants that pin the inference-time preprocessing to the training recipe.

Changing any value in this module changes the visual-token layout the model was
trained on, and will silently degrade layout quality.  Treat them as frozen.
"""

# Qwen2.5-VL patch geometry.  Every image edge fed to the model must be a
# multiple of IMAGE_FACTOR (patch_size 14 * spatial_merge_size 2).
IMAGE_FACTOR = 28

# Guard rails copied from qwen_vl_utils.vision_process.
MAX_RATIO = 200
MIN_SIZE = 28

# Visual-token budget per layer asset.  These are the values used for both the
# SFT and the RFT (GRPO) stages, and they are what the released checkpoint's
# preprocessor_config.json already carries (15680 / 56448).
LAYER_MIN_PIXELS = 20 * 28 * 28  # 15680  -> ~125x125 px
LAYER_MAX_PIXELS = 72 * 28 * 28  # 56448  -> ~237x237 px

# Layer assets are letterboxed onto a grey canvas.  The system prompt tells the
# model to ignore these grey bars when reasoning about aspect ratios.
PADDING_COLOR = (128, 128, 128)

# Output length budget.  Measured against the checkpoint tokenizer, one layer
# object costs 52-58 tokens and the canvas_size header costs ~25, growing with
# the digit count of the coordinates.  72/layer + 128 leaves comfortable
# headroom (generation stops at EOS anyway, so over-budgeting is free).
TOKENS_PER_LAYER = 72
TOKENS_OVERHEAD = 128
