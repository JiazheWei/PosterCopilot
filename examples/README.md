# Examples

Three posters decomposed into their layers, with the layout PosterCopilot
predicts for them and the rendered result.

| Case | Layers | Canvas |
|---|---|---|
| [`jazz-day`](jazz-day) | 5 | 800 × 1600 |
| [`klean-energy`](klean-energy) | 6 | 1753 × 2480 |
| [`womens-day`](womens-day) | 9 | 2480 × 3508 |

<table>
<tr>
<td width="27%"><img src="jazz-day/poster.png" alt="jazz-day"></td>
<td width="33%"><img src="klean-energy/poster.png" alt="klean-energy"></td>
<td width="33%"><img src="womens-day/poster.png" alt="womens-day"></td>
</tr>
<tr>
<td align="center"><b>jazz-day</b> · 5 layers</td>
<td align="center"><b>klean-energy</b> · 6 layers</td>
<td align="center"><b>womens-day</b> · 9 layers</td>
</tr>
</table>

## What is in each folder

```
jazz-day/
├── layers/           the input assets, one RGBA PNG per layer
├── layout.json       what the model predicted
├── poster.png        layout.json rendered
├── reference.json    the layout of the original human design, same schema
└── annotation.json   the full original annotation of that design
```

Layer files are named `<index>_<category>.png`. The numeric prefix **is** the
`image_id` the model sees — files are sorted in natural filename order and
numbered from zero. `category` is only there to make the folder readable; the
model infers each layer's role from pixels alone.

## Run it

Predict a layout from the layers:

```bash
python infer.py --model checkpoints/postercopilot-7b \
                --assets examples/jazz-day/layers \
                --width 800 --height 1600 \
                -o /tmp/layout.json
```

```
INFO examples/jazz-day/layers: 5 layers -> 800x1600 canvas
INFO examples/jazz-day/layers: 274 tokens in 6.9s -> /tmp/layout.json
```

Composite it into the finished poster:

```bash
python render.py --layout /tmp/layout.json \
                 --assets examples/jazz-day/layers \
                 -o /tmp/poster.png --flatten
```

```
INFO layout.json: composited 5 layers -> /tmp/poster.png (800x1600)
```

That is the whole pipeline. Each folder already ships the `layout.json` from our
own run, so `render.py` alone reproduces `poster.png` byte for byte on a machine
with no GPU.

Your own `layout.json` will not be byte-identical to the shipped one. Coordinates
are generated digit by digit, so one flipped digit in bfloat16 re-rolls the rest
of the layout — different GPUs, dtypes and attention kernels land on
different-but-comparable results.

## The original annotation

`annotation.json` is the raw parse of the source design file, richer than the
layout schema the model emits. Per layer it carries:

| Field | |
|---|---|
| `x, y, w, h, order` | geometry, same meaning as in `layout.json` |
| `category` | `type` / `image` |
| `blend_mode`, `opacity` | Photoshop compositing state |
| `text_info` | the actual string, font family, size, colour, alignment |
| `group`, `merged_layers_*` | which source layers were merged into this one |
| `files` | the exported layer image |

The model predicts only the geometry columns. The rest is here for anyone
building a richer renderer — real text typesetting instead of pasted bitmaps, or
honouring blend modes. `render.py` ignores all of it and does plain source-over
compositing, which is enough to reproduce the reported performance.

## Provenance

These three cases come from the PosterCopilot dataset described in the paper,
included as a small sample so the code is runnable out of the box. The full
dataset is not part of this release.
