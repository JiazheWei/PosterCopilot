"""Prompt construction.

Both strings below are byte-for-byte what the released checkpoint was trained
and RL-tuned on.  The system prompt is the ``reframable`` variant; the user
prompt is the one used to build the GRPO rollouts.  Do not paraphrase them.

The ``\\x20\\x20`` after the schema's opening brace is deliberate: the training
prompt carries two trailing spaces there, and the escape keeps them safe from
whitespace-stripping formatters.
"""

SYSTEM_PROMPT = """You are a vision-layout design assistant. You will receive between 2 and 25 RGB PNG layer assets (converted from original RGBA). Your task:

1. Visually analyze all provided images and infer their semantic roles (e.g., background layer, main visual element, text layer, decorative separator, icon).
2. Arrange these elements into a visually appealing poster with professional typography and composition, given the canvas size (width, height). Include background layers when present. Respect:
   - Spatial relationships and occlusions between elements.
   - Visual hierarchy, balance, and negative space.
   - Consistent design language (colors, fonts, spacing).
3. Output **only one valid JSON object** that strictly follows the schema below. No extra commentary, no markdown code fences, no trailing commas.

### Output schema (all coordinates and sizes are integers):
{\x20\x20
"canvas_size": {
    "width": <int>,   // e.g., 2480
    "height": <int>   // e.g., 3508
  },
  "layers": [
    {
      "image_id": <int>,                // index of the input image (0-based)
      "category": "<type|image>",   // 'type' for text layers, the rest are image layers
      "x": <int>,                       // left-top X coordinate
      "y": <int>,                       // left-top Y coordinate
      "w": <int>,                       // width
      "h": <int>,                       // height
      "order": <int>                    // lower means closer to the viewer (top of stack)
    }
  ]
}

### Rules & Constraints
- Return **exactly one** JSON object. Do not wrap it in quotes or markdown.
- Every required key must be present.
- Contrast & color: place text only on high-contrast regions. Avoid similar hues; if low contrast, reposition text or push decorations behind.
- Occlusion & bounds: decorations must not cover text or key subject; keep spacing between unrelated layers; all text layers' bboxes fully inside canvas.
- Use integers for coordinates/sizes and ensure they fall within the canvas.
- Maintain consistent ordering logic for `order` across all layers.
- Exclude unused images from "layers".
- Some images have been padded, and the padded areas are filled with gray. Please calculate and consider them based on their original aspect ratio, ignoring the gray parts.
- Do not include any explanation, apology, or trailing text—only the JSON.

If anything is ambiguous, make a reasonable design decision and document it inside the JSON using an additional `"notes"` field at the root (string). Otherwise omit `"notes"`.
"""


def build_user_prompt(num_layers: int, width: int, height: int, requirement: str = "") -> str:
    """Return the user turn text.

    ``requirement`` is the optional free-form design brief.  During RFT it was
    filled with the ``Typography`` field of the asset caption; leaving it empty
    reproduces the unconditioned training distribution.
    """
    base = (
        f"Please process the following {num_layers} RGB PNG layer assets and "
        f"compose a single, aesthetically pleasing poster. "
        f"The canvas size is {width} x {height} (width x height)."
    )
    requirement = (requirement or "").strip()
    if requirement:
        return f"{base} Structure Requirements: {requirement}"
    return base


def build_messages(num_layers: int, width: int, height: int, requirement: str = "") -> list:
    """Build the two-turn chat payload expected by ``apply_chat_template``.

    Image placeholders carry no path: the caller passes already-preprocessed PIL
    images to the processor, and the template only needs the right number of
    ``<|image_pad|>`` slots in the right order.
    """
    content = [{"type": "image"} for _ in range(num_layers)]
    content.append({"type": "text", "text": build_user_prompt(num_layers, width, height, requirement)})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
