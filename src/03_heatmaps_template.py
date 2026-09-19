# %% [markdown]
# # PANDA 3/3 — Attention heatmaps
#
# Draws what the grading model looked at, for a few slides picked from the finished training run.
# The attention values, tile positions and mask labels are baked into this notebook, so it needs
# only the competition images and runs on CPU in a couple of minutes.
#
# Each row is one slide: the slide itself, the model's attention, and the cancer tiles according
# to the label mask, which the model never saw during training.
#
# **Kaggle settings:** no accelerator needed, Internet **On** (installs OpenSlide), and add the
# competition **prostate-cancer-grade-assessment** as an input.
#
# Regenerate this notebook after a new training run with:
#
#     python src/make_heatmap_notebook.py && python src/build_notebooks.py

# %%
!pip install -q -U openslide-bin openslide-python

# %%
import base64
import json
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openslide

PAYLOAD = ""

data = json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
TILE_SIZE, TILE_LEVEL = data["tile_size"], data["tile_level"]
slides = data["slides"]
print(f"{len(slides)} slides, {sum(len(s['x']) for s in slides):,} tiles")


def find_slide_dir(root=Path("/kaggle/input")):
    known = root / "competitions" / "prostate-cancer-grade-assessment" / "train_images"
    if known.is_dir():
        return known
    for depth in range(1, 5):
        for images in root.glob("/".join(["*"] * depth) + "/train_images"):
            return images
    raise FileNotFoundError("add the prostate-cancer-grade-assessment competition as an input")


SLIDE_DIR = find_slide_dir()
print("slides:", SLIDE_DIR)

# %% [markdown]
# ## Drawing
#
# Attention is rank-scaled per slide (0 = least looked at, 1 = most), because the raw weights sum
# to one and get smaller the more tiles a slide has, which would make slides incomparable.

# %%
def slide_maps(slide):
    """Thumbnail plus the attention and mask-cancer overlays, all in thumbnail pixels."""
    with openslide.OpenSlide(str(SLIDE_DIR / f"{slide['image_id']}.tiff")) as handle:
        level = handle.level_count - 1
        width, height = handle.level_dimensions[level]
        thumb = np.asarray(handle.read_region((0, 0), level, (width, height)).convert("RGB"))
        downsample = handle.level_downsamples[level]
        side = int(np.ceil(TILE_SIZE * handle.level_downsamples[TILE_LEVEL] / downsample))

    attention = np.asarray(slide["attention"], dtype=np.float32)
    ranks = attention.argsort().argsort() / max(len(attention) - 1, 1)
    attention_map = np.full((height, width), np.nan, dtype=np.float32)
    cancer_map = np.full((height, width), np.nan, dtype=np.float32)
    for x, y, rank, label in zip(slide["x"], slide["y"], ranks, slide["label"]):
        px, py = int(x / downsample), int(y / downsample)
        attention_map[py:py + side, px:px + side] = rank
        if label >= 0:  # -1 means the mask does not cover this tile
            cancer_map[py:py + side, px:px + side] = label
    return thumb, attention_map, cancer_map


fig, axes = plt.subplots(len(slides), 3, figsize=(13, 4.2 * len(slides)))
for row, slide in zip(np.atleast_2d(axes), slides):
    thumb, attention_map, cancer_map = slide_maps(slide)
    auc = "no mask" if slide["auc"] is None else f"localisation AUC {slide['auc']:.2f}"

    row[0].imshow(thumb)
    row[0].set_title(f"{slide['provider']} | ISUP {slide['isup']} | predicted {slide['pred']}\n{slide['note']}",
                     fontsize=9)
    row[1].imshow(thumb)
    row[1].imshow(attention_map, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    row[1].set_title(f"attention ({auc})", fontsize=9)
    row[2].imshow(thumb)
    row[2].imshow(cancer_map, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    row[2].set_title("cancer tiles from the mask", fontsize=9)
    for ax in row:
        ax.axis("off")

plt.tight_layout()
plt.savefig("/kaggle/working/attention_heatmaps.png", dpi=130, bbox_inches="tight")
plt.show()
print("saved /kaggle/working/attention_heatmaps.png")

# %% [markdown]
# Download `attention_heatmaps.png` from this version's Output and commit it to the repo under
# `results/`.
