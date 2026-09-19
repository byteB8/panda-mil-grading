# %% [markdown]
# # PANDA 1/2 — Tissue tiling and patch feature extraction
#
# Cuts each prostate biopsy slide into tissue-only tiles and turns every tile into a
# 768-dim feature vector with **Phikon** (a ViT-B/16 pretrained on histology). The slide-level
# classifier in notebook 2 trains on these features, so this expensive step runs once.
#
# **Results are saved to a private Kaggle Dataset in your account**
# (`kumaarbalbir/panda-phikon-features`), uploaded in parts as the run goes. If a run crashes or hits
# Kaggle's 12-hour limit, run the notebook again: it downloads the finished parts and carries on.
# Notebook 2 attaches this dataset, and you can download it to your own machine.
#
# **One-time setup:**
# 1. kaggle.com → Settings → API → **Create New Token**. Open the downloaded `kaggle.json`
#    and copy the `key` value.
# 2. In this notebook: **Add-ons → Secrets → Add Secret**, label `KAGGLE_KEY`, paste the key,
#    and make sure the secret is ticked for this notebook. (If Kaggle gives you a new-style
#    token instead, label it `KAGGLE_API_TOKEN`.)
#
# **Kaggle settings (right-hand panel):**
# - Accelerator: **GPU T4 x2** (P100 also works, about half the throughput)
# - Internet: **On** (installs OpenSlide, downloads Phikon, uploads results)
# - Input: add the competition **prostate-cancer-grade-assessment** (join it and accept the rules first)
#
# **How to run:**
# 1. Leave `DEBUG_SLIDES = 50` and run interactively. Check the tiling pictures and the benchmark
#    estimate. Dry runs upload nothing.
# 2. Set `DEBUG_SLIDES = None`, then **Save Version → Save & Run All (Commit)**. It keeps
#    running after you close the browser. If it stops early, commit again.

# %%
!pip install -q -U openslide-bin openslide-python kaggle

# %%
import json
import os
import shutil
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ViTModel



def find_competition_data(root=Path("/kaggle/input")):
    """The PANDA folder wherever Kaggle mounts it (the mount path differs between Kaggle setups)."""
    known = root / "competitions" / "prostate-cancer-grade-assessment"  # the path on this Kaggle account
    if (known / "train.csv").exists():
        return known
    for depth in range(1, 5):
        for csv in root.glob("/".join(["*"] * depth) + "/train.csv"):
            if (csv.parent / "train_images").is_dir():
                return csv.parent
    raise FileNotFoundError("PANDA data not found: add the prostate-cancer-grade-assessment competition as an input")


DATA_DIR = find_competition_data()
print("competition data:", DATA_DIR)
OUT_DIR = Path("/kaggle/working")

TILE_LEVEL = 0           # pyramid level to tile: 0 = full resolution (~20x, what Phikon was trained on), 1 = 4x smaller
TILE_SIZE = 224          # pixels at TILE_LEVEL; Phikon's input size
MIN_TISSUE = 0.5         # keep tiles that are at least this fraction tissue
MAX_TILES = 2048         # per-slide cap; keeps the most tissue-rich tiles
PART_SLIDES = 1000       # slides per saved part; a crash loses at most the part in progress
CHECKPOINT_MINUTES = 120 # upload finished parts at most this often, and always at the end
DEBUG_SLIDES = 50        # dry run on this many slides (nothing uploaded); None for the real run
BATCH_SIZE = 256
NUM_WORKERS = os.cpu_count()
MODEL_NAME = "owkin/phikon"
STAIN_NORMALISE = False  # Macenko-normalise every tile to a reference stain before encoding

KAGGLE_USERNAME = "kumaarbalbir"
DATASET_SLUG = "panda-phikon-features-macenko" if STAIN_NORMALISE else "panda-phikon-features"
DATASET_TITLE = "PANDA Phikon tile features" + (" (Macenko)" if STAIN_NORMALISE else "")

DATASET_ID = f"{KAGGLE_USERNAME.lower()}/{DATASET_SLUG}"  # replaced below by your real username
SAVE_DIR = OUT_DIR / DATASET_SLUG  # finished parts; this folder is what gets uploaded
TMP_DIR = OUT_DIR / "tmp"
PERSIST = not DEBUG_SLIDES

print("GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("CPUs:", os.cpu_count())

# %% [markdown]
# ## Saving to your Kaggle account
#
# Uploads go through the `kaggle` command-line tool, authenticated with your secret. New
# datasets are private. Each upload replaces the previous version (`-d` deletes old versions),
# so your storage quota only holds one copy.

# %%
def kaggle_login():
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    for name in ["KAGGLE_KEY", "KAGGLE_API_TOKEN"]:
        try:
            os.environ[name] = secrets.get_secret(name)
        except Exception:
            pass
    if not {"KAGGLE_KEY", "KAGGLE_API_TOKEN"} & set(os.environ):
        raise RuntimeError("No Kaggle credentials: add a secret named KAGGLE_KEY (Add-ons → Secrets) "
                           "and tick it for this notebook")


def detect_owner():
    """Your real Kaggle username, read from something you own. A dataset id with any other owner is rejected."""
    for what in ("kernels", "datasets"):
        ok, output = kaggle_cli(what, "list", "--mine", "--csv")
        if ok:
            refs = [line.split(",")[0].strip() for line in output.splitlines()[1:] if "/" in line]
            if refs:
                return refs[0].split("/")[0].lower()
    print(f"WARNING: could not read your username from Kaggle; falling back to {KAGGLE_USERNAME!r}")
    return KAGGLE_USERNAME.lower()


FAILURE_MARKERS = ("client error", "server error", "403 - forbidden", "404 - not found", "error:")


def kaggle_cli(*args):
    result = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    ok = result.returncode == 0 and not any(marker in output.lower() for marker in FAILURE_MARKERS)
    return ok, output


def dataset_exists():
    """Looks for the dataset in your own list. (Kaggle's status call answers 403, not 404, for a missing one.)"""
    ok, output = kaggle_cli("datasets", "list", "--mine", "--search", DATASET_SLUG, "--csv")
    if not ok:
        raise RuntimeError(f"could not list your datasets: {output}")
    refs = {line.split(",")[0].strip().lower() for line in output.splitlines()[1:]}
    return DATASET_ID in refs


def dataset_status():
    """'ready', 'pending', or None if the dataset does not exist yet. Raises if Kaggle can't be reached."""
    if not dataset_exists():
        return None
    ok, output = kaggle_cli("datasets", "status", DATASET_ID)
    if ok and "pending" in output.lower():
        return "pending"
    return "ready"  # also when the status is unreadable: the upload itself reports real problems


def restore(folder):
    """Downloads what earlier runs uploaded, so this run can skip it."""
    folder.mkdir(parents=True, exist_ok=True)
    if dataset_status() is None:
        print(f"{DATASET_ID} does not exist yet; starting fresh")
        return
    ok, output = kaggle_cli("datasets", "download", DATASET_ID, "-p", str(folder), "--unzip", "-q")
    if not ok:
        raise RuntimeError(f"could not download {DATASET_ID}: {output}")
    print(f"restored from {DATASET_ID}:", sorted(p.name for p in folder.iterdir()))


def publish(folder, message, attempts=10):
    """Uploads `folder` as a new private version of the dataset, retrying while Kaggle is busy."""
    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": DATASET_TITLE, "id": DATASET_ID, "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}))
    for attempt in range(attempts):
        try:
            status = dataset_status()
        except RuntimeError as e:  # network hiccup: wait and retry
            print(e)
            time.sleep(60)
            continue
        if status == "pending":  # the previous version is still being processed
            time.sleep(60)
            continue
        if status is None:
            ok, output = kaggle_cli("datasets", "create", "-p", str(folder), "-q")
        else:
            ok, output = kaggle_cli("datasets", "version", "-p", str(folder), "-m", message, "-d", "-q")
        if ok:
            print(f"uploaded to https://www.kaggle.com/datasets/{DATASET_ID} ({message})")
            return True
        print(f"upload attempt {attempt + 1} failed: {output}")
        time.sleep(60)
    # Never fatal: the work is on disk, and a notebook that finishes keeps /kaggle/working in its output.
    print(f"WARNING: could not upload to {DATASET_ID}; the files are still in {folder} "
          f"and will be saved with this version's output")
    return False


if PERSIST:
    kaggle_login()
    ok, output = kaggle_cli("datasets", "list", "--mine")
    assert ok, f"Kaggle credentials rejected: {output}"
    DATASET_ID = f"{detect_owner()}/{DATASET_SLUG}"
    print(f"Kaggle API login OK; results will go to {DATASET_ID}")

# %% [markdown]
# ## The data

# %%
train = pd.read_csv(DATA_DIR / "train.csv")
masked_ids = {p.name.removesuffix("_mask.tiff") for p in (DATA_DIR / "train_label_masks").glob("*_mask.tiff")}
train["has_mask"] = train.image_id.isin(masked_ids)

print(train.shape)
print(pd.crosstab(train.isup_grade, train.data_provider, margins=True))
print("slides with label masks:", train.has_mask.sum())

slides = train.sort_values("image_id").reset_index(drop=True)
if DEBUG_SLIDES:
    # Take a mix of both centres so the dry run shows both stain styles.
    slides = slides.groupby("data_provider").head(DEBUG_SLIDES // 2).reset_index(drop=True)
parts = [slides.iloc[i:i + PART_SLIDES].reset_index(drop=True) for i in range(0, len(slides), PART_SLIDES)]
print(f"{len(slides)} slides in {len(parts)} parts")

# %% [markdown]
# ## Tissue detection and tiling
#
# The smallest pyramid level (16x down) is enough to tell tissue from white glass. Tissue is
# coloured (high saturation) and darker than the background. Each candidate tile's tissue
# fraction comes from an integral image, so there is no per-tile loop.
#
# The label masks use different codes per centre:
# - **Radboud:** 0 background, 1 stroma, 2 benign epithelium, 3/4/5 Gleason pattern 3/4/5
# - **Karolinska:** 0 background, 1 benign tissue, 2 cancer
#
# `cancer_frac` is the fraction of each tile that the mask marks as cancer. Notebook 2 uses it
# to check where the model's attention falls and to train a tile-level cancer detector.

# %%
def slide_path(image_id):
    return DATA_DIR / "train_images" / f"{image_id}.tiff"


def mask_path(image_id):
    return DATA_DIR / "train_label_masks" / f"{image_id}_mask.tiff"


def tissue_mask(rgb):
    """True where a thumbnail shows tissue rather than background."""
    rgb = rgb.astype(np.int16)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    brightness = rgb.mean(axis=2)
    return (saturation > 20) & (brightness < 230)


def box_means(binary, xs, ys, size):
    """Mean of a 2-D boolean array over square boxes with top-left corners (xs, ys) and side `size`."""
    h, w = binary.shape
    integral = np.zeros((h + 1, w + 1), dtype=np.int64)
    integral[1:, 1:] = binary.astype(np.int64).cumsum(0).cumsum(1)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, w)
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, h)
    x1 = np.clip(np.ceil(xs + size).astype(np.int64), 0, w)
    y1 = np.clip(np.ceil(ys + size).astype(np.int64), 0, h)
    total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    area = np.maximum((x1 - x0) * (y1 - y0), 1)
    return total / area


# Macenko's reference stain vectors and concentrations: every slide is mapped onto these, so
# tiles from the two hospitals arrive at the encoder with the same stain appearance.
HE_REFERENCE = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]], dtype=np.float32)
MAX_CONCENTRATION_REFERENCE = np.array([1.9705, 1.0308], dtype=np.float32)
STAIN_COLUMNS = [f"stain_{i}" for i in range(6)] + ["maxc_0", "maxc_1"]


def macenko_stains(pixels, beta=0.15, alpha=1.0):
    """Haematoxylin and eosin vectors for one slide, from its tissue pixels (Macenko et al., 2009).

    Returns a 3x2 stain matrix and the 99th-percentile concentration of each stain, or None when
    the slide has too little stained tissue to estimate them."""
    optical_density = -np.log((pixels.astype(np.float32) + 1) / 256)
    stained = optical_density[(optical_density > beta).all(axis=1)]
    if len(stained) < 500:
        return None
    _, vectors = np.linalg.eigh(np.cov(stained.T))
    plane = vectors[:, 1:3]  # the two directions carrying the stain signal
    angles = np.arctan2(stained @ plane[:, 1], stained @ plane[:, 0])
    first, second = np.percentile(angles, [alpha, 100 - alpha])
    stains = np.stack([plane @ [np.cos(a), np.sin(a)] for a in (first, second)], axis=1)
    stains *= np.sign(stains[np.abs(stains).argmax(axis=0), [0, 1]])  # point both away from zero
    if stains[0, 0] < stains[0, 1]:  # haematoxylin is the bluer stain: less red than eosin
        stains = stains[:, ::-1]
    stains /= np.linalg.norm(stains, axis=0, keepdims=True)
    concentrations = np.linalg.lstsq(stains, stained.T, rcond=None)[0]
    return stains.astype(np.float32), np.percentile(concentrations, 99, axis=1).astype(np.float32)


def read_level(slide, level):
    w, h = slide.level_dimensions[level]
    return np.asarray(slide.read_region((0, 0), level, (w, h)).convert("RGB"))


def tile_slide(image_id, provider):
    """Tissue tiles for one slide, plus mask statistics when a label mask exists."""
    with openslide.OpenSlide(str(slide_path(image_id))) as slide:
        if slide.level_count <= TILE_LEVEL:
            raise ValueError(f"only {slide.level_count} levels")
        thumb_level = slide.level_count - 1
        ds_tile = slide.level_downsamples[TILE_LEVEL]
        ds_thumb = slide.level_downsamples[thumb_level]
        thumbnail = read_level(slide, thumb_level)
        tissue = tissue_mask(thumbnail)
        width, height = slide.level_dimensions[TILE_LEVEL]

    gx, gy = np.meshgrid(np.arange(width // TILE_SIZE), np.arange(height // TILE_SIZE))
    gx, gy = gx.ravel(), gy.ravel()
    scale = TILE_SIZE * ds_tile / ds_thumb  # tile side in thumbnail pixels
    frac = box_means(tissue, gx * scale, gy * scale, scale)

    keep = np.flatnonzero(frac >= MIN_TISSUE)
    if len(keep) == 0:
        # Faint or tiny biopsies: fall back to the best partial tiles rather than dropping the slide.
        keep = np.flatnonzero(frac > 0)[np.argsort(-frac[frac > 0])[:16]]
    keep = keep[np.argsort(-frac[keep], kind="stable")[:MAX_TILES]]
    keep.sort()  # back to raster order: neighbouring tiles read faster

    tiles = pd.DataFrame({
        "slide_id": image_id,
        "x": (gx[keep] * TILE_SIZE * ds_tile).astype(np.int64),
        "y": (gy[keep] * TILE_SIZE * ds_tile).astype(np.int64),
        "tissue_frac": frac[keep].astype(np.float32),
        "cancer_frac": np.float32(np.nan),
        "epithelium_frac": np.float32(np.nan),
        "gleason3_frac": np.float32(np.nan),
        "gleason4_frac": np.float32(np.nan),
        "gleason5_frac": np.float32(np.nan),
    })

    stains = dict.fromkeys(STAIN_COLUMNS, np.nan)
    if STAIN_NORMALISE:
        estimate = macenko_stains(thumbnail[tissue])
        if estimate is not None:
            matrix, max_concentration = estimate
            stains = dict(zip(STAIN_COLUMNS, [*matrix.ravel(), *max_concentration]))

    mask_counts = [np.nan] * 6
    if mask_path(image_id).exists():
        with openslide.OpenSlide(str(mask_path(image_id))) as mask:
            mask_level = mask.level_count - 1
            ds_mask = mask.level_downsamples[mask_level]
            labels = read_level(mask, mask_level)[..., 0]  # labels live in the red channel
        mscale = TILE_SIZE * ds_tile / ds_mask

        def tile_fraction(region):
            return box_means(region, gx[keep] * mscale, gy[keep] * mscale, mscale).astype(np.float32)

        # Radboud labels glands and stroma separately, so cancer covers a small share of a tile even when
        # every gland in it is cancer. epithelium_frac lets notebook 2 measure cancer as a share of glands.
        # Karolinska does not separate them, so there it is all labelled tissue.
        if provider == "radboud":
            tiles["cancer_frac"] = tile_fraction(labels >= 3)
            tiles["epithelium_frac"] = tile_fraction(labels >= 2)
            for pattern in (3, 4, 5):
                tiles[f"gleason{pattern}_frac"] = tile_fraction(labels == pattern)
        else:
            tiles["cancer_frac"] = tile_fraction(labels == 2)
            tiles["epithelium_frac"] = tile_fraction(labels >= 1)
        mask_counts = np.bincount(labels.ravel(), minlength=6)[:6].tolist()

    return tiles, mask_counts, stains


def tile_slide_safe(args):
    image_id, provider = args
    try:
        return image_id, *tile_slide(image_id, provider), None
    except Exception as e:  # a corrupt slide should not kill a 10-hour run
        return image_id, None, None, None, repr(e)


def tile_slides(slide_table):
    """Tiles a table of slides in parallel. Returns (tiles, per-slide stats, failures)."""
    tile_tables, slide_rows, failed = [], [], []
    jobs = list(zip(slide_table.image_id, slide_table.data_provider))
    with Pool(NUM_WORKERS) as pool:
        for image_id, tiles, mask_counts, stains, error in tqdm(pool.imap(tile_slide_safe, jobs, chunksize=4),
                                                                total=len(jobs), desc="tiling"):
            if error:
                failed.append({"image_id": image_id, "error": error})
                continue
            if len(tiles) == 0:  # blank or washed-out slide: no tile held any tissue
                failed.append({"image_id": image_id, "error": "no tissue tiles found"})
                continue
            slide_rows.append({"image_id": image_id, "n_tiles": len(tiles),
                               **{f"mask_px_{k}": c for k, c in enumerate(mask_counts)}, **stains})
            tile_tables.append(tiles)

    assert tile_tables, "no slide in this part produced tiles"
    tiles = pd.concat(tile_tables, ignore_index=True)
    slide_rows = pd.DataFrame(slide_rows)
    slide_rows["offset"] = np.concatenate([[0], np.cumsum(slide_rows.n_tiles.values)[:-1]])  # row of first tile
    slide_stats = slide_table.merge(slide_rows, on="image_id")
    assert (tiles.slide_id.values[slide_stats.offset.values] == slide_stats.image_id.values).all()
    return tiles, slide_stats, failed

# %% [markdown]
# ## Feature extractor
#
# Tiles are read by DataLoader workers as uint8 and normalised on the GPU. With two GPUs the
# batch is split across both with `DataParallel`.

# %%
class TileDataset(Dataset):
    def __init__(self, tiles, slide_index=None):
        self.slide_ids = tiles.slide_id.values
        self.xs = tiles.x.values
        self.ys = tiles.y.values
        self.slide_index = slide_index  # row of each tile's slide in the stain tables
        self.handles = {}  # per worker; tiles are in slide order, so a small cache hits almost always

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, i):
        slide_id = self.slide_ids[i]
        if slide_id not in self.handles:
            if len(self.handles) >= 8:
                for handle in self.handles.values():
                    handle.close()
                self.handles.clear()
            self.handles[slide_id] = openslide.OpenSlide(str(slide_path(slide_id)))
        region = self.handles[slide_id].read_region(
            (int(self.xs[i]), int(self.ys[i])), TILE_LEVEL, (TILE_SIZE, TILE_SIZE))
        tile = torch.from_numpy(np.array(region.convert("RGB"))).permute(2, 0, 1)
        return tile if self.slide_index is None else (tile, self.slide_index[i])


class Encoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        processor = AutoImageProcessor.from_pretrained(name)
        self.vit = ViTModel.from_pretrained(name, add_pooling_layer=False)
        self.register_buffer("mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(processor.image_std).view(1, 3, 1, 1))
        self.register_buffer("he_reference", torch.from_numpy(HE_REFERENCE))
        self.register_buffer("maxc_reference", torch.from_numpy(MAX_CONCENTRATION_REFERENCE))

    def restain(self, images, stain_pinv, max_concentration):
        """Rewrites each tile in the reference stain: unmix with its slide's stains, remix with ours."""
        optical_density = -torch.log((images.float() + 1) / 256).flatten(2)          # B x 3 x pixels
        concentration = torch.bmm(stain_pinv, optical_density)                       # B x 2 x pixels
        concentration = concentration * (self.maxc_reference / max_concentration).unsqueeze(-1)
        remixed = torch.matmul(self.he_reference, concentration)                     # B x 3 x pixels
        return (256 * torch.exp(-remixed) - 1).clamp(0, 255).view_as(images)

    def forward(self, images, stain_pinv=None, max_concentration=None):
        if stain_pinv is not None:
            images = self.restain(images, stain_pinv, max_concentration)
        x = (images.float() / 255 - self.mean) / self.std
        return self.vit(pixel_values=x).last_hidden_state[:, 0]  # CLS token


encoder = Encoder(MODEL_NAME).cuda().eval()
FEATURE_DIM = encoder.vit.config.hidden_size
if torch.cuda.device_count() > 1:
    encoder = nn.DataParallel(encoder)


def stain_tables(slide_stats):
    """Per-slide pseudo-inverse of the stain matrix and max concentrations, ready for the GPU.

    Slides whose stains could not be estimated fall back to the reference, which leaves them
    essentially unchanged."""
    matrices = slide_stats[STAIN_COLUMNS[:6]].to_numpy(dtype=np.float32, copy=True).reshape(-1, 3, 2)
    maxc = slide_stats[STAIN_COLUMNS[6:]].to_numpy(dtype=np.float32, copy=True)
    matrices[~np.isfinite(matrices).all(axis=(1, 2))] = HE_REFERENCE
    maxc[~np.isfinite(maxc).all(axis=1)] = MAX_CONCENTRATION_REFERENCE
    return (torch.from_numpy(np.linalg.pinv(matrices)).cuda(),  # B x 2 x 3
            torch.from_numpy(maxc).cuda())


def make_loader(tile_table, slide_index=None):
    return DataLoader(TileDataset(tile_table, slide_index), batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4)


def tile_slide_index(tile_table, slide_stats):
    """Which row of `slide_stats` each tile belongs to."""
    rows = pd.Series(np.arange(len(slide_stats)), index=slide_stats.image_id.values)
    return torch.from_numpy(rows.reindex(tile_table.slide_id.values).to_numpy(dtype=np.int64))


def encode(loader, sink=None, max_batches=None, stains=None):
    """Runs the encoder over a loader, writing rows into `sink` if given. Returns (tiles, tiles/s)."""
    done, start, done_at_start = 0, None, 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for b, batch in enumerate(tqdm(loader, total=max_batches or len(loader), desc="encoding")):
            if b == 1:
                torch.cuda.synchronize()
                start, done_at_start = time.time(), done  # skip the warm-up batch in the timing
            if stains is None:
                out = encoder(batch.cuda(non_blocking=True))
            else:
                images, index = batch
                index = index.cuda(non_blocking=True)
                out = encoder(images.cuda(non_blocking=True), stains[0][index], stains[1][index])
            out = out.float().cpu().numpy()
            if sink is not None:
                sink[done:done + len(out)] = out.astype(np.float16)
            done += len(out)
            if max_batches and b + 1 >= max_batches:
                break
    torch.cuda.synchronize()
    if start is None:
        return done, float("nan")
    return done, (done - done_at_start) / (time.time() - start)

# %% [markdown]
# ## Dry run: check the tiling and estimate the full run
#
# Only runs when `DEBUG_SLIDES` is set. Green boxes are kept tiles; red boxes are mostly
# cancer according to the mask. If boxes miss pale tissue or cover pen marks and background,
# adjust the thresholds in `tissue_mask` and rerun.

# %%
def show_slide(ax, image_id, tiles, slide_stats):
    with openslide.OpenSlide(str(slide_path(image_id))) as slide:
        thumb_level = slide.level_count - 1
        ax.imshow(read_level(slide, thumb_level))
        ds_thumb = slide.level_downsamples[thumb_level]
        side = TILE_SIZE * slide.level_downsamples[TILE_LEVEL] / ds_thumb
    for t in tiles[tiles.slide_id == image_id].itertuples():
        colour = "red" if t.cancer_frac >= 0.5 else "lime"
        ax.add_patch(plt.Rectangle((t.x / ds_thumb, t.y / ds_thumb), side, side, fill=False, lw=0.6, ec=colour))
    row = slide_stats.set_index("image_id").loc[image_id]
    mask_note = "" if row.has_mask else " | no mask"
    ax.set_title(f"{row.data_provider} ISUP {row.isup_grade} | {row.n_tiles} tiles{mask_note}", fontsize=9)
    ax.axis("off")


if DEBUG_SLIDES:
    t0 = time.time()
    tiles, slide_stats, failed = tile_slides(slides)
    tiling_minutes = (time.time() - t0) / 60
    print(f"tiled {len(slide_stats)} slides in {tiling_minutes:.1f} min, {len(failed)} failed")
    print(f"{len(tiles):,} tiles; per slide: {slide_stats.n_tiles.describe().round(1).to_dict()}")
    print("slides at the MAX_TILES cap:", (slide_stats.n_tiles == MAX_TILES).sum())
    for failure in failed[:10]:
        print("  failed:", failure)

    # Sanity check for the mask labels: high-grade slides should have tiles that are mostly cancer.
    high_grade = tiles.merge(slide_stats[["image_id", "data_provider", "isup_grade"]],
                             left_on="slide_id", right_on="image_id")
    high_grade = high_grade[high_grade.cancer_frac.notna() & (high_grade.isup_grade >= 3)]
    high_grade["cancer_share"] = high_grade.cancer_frac / high_grade.epithelium_frac.where(high_grade.epithelium_frac > 0)
    print("tiles in ISUP >= 3 slides that have masks:")
    print(high_grade.groupby("data_provider")[["cancer_frac", "epithelium_frac", "cancer_share"]]
          .median().round(2).rename(columns=lambda c: f"median {c}"))
    print("share of those tiles counted as cancer (>= 10% cancer and >= 50% of epithelium):",
          high_grade.groupby("data_provider").apply(
              lambda g: ((g.cancer_frac >= 0.1) & (g.cancer_share >= 0.5)).mean(), include_groups=False)
          .round(3).to_dict())

    examples = pd.concat([group.sort_values("isup_grade").iloc[[0, -1]]  # lowest and highest grade per centre
                          for _, group in slide_stats.groupby("data_provider")])
    fig, axes = plt.subplots(1, len(examples), figsize=(5 * len(examples), 6))
    for ax, image_id in zip(axes, examples.image_id):
        show_slide(ax, image_id, tiles, slide_stats)
    plt.tight_layout()
    plt.show()

    sample = tiles.head(BATCH_SIZE * 12)
    sample_stains = stain_tables(slide_stats) if STAIN_NORMALISE else None
    sample_index = tile_slide_index(sample, slide_stats) if STAIN_NORMALISE else None
    _, tiles_per_sec = encode(make_loader(sample, sample_index), max_batches=12, stains=sample_stains)

    if STAIN_NORMALISE:  # eyeball the stain transform on one tile per example slide
        picks = [tiles[tiles.slide_id == image_id].iloc[len(tiles[tiles.slide_id == image_id]) // 2]
                 for image_id in examples.image_id]
        raw = torch.stack([TileDataset(pd.DataFrame([t]))[0] for t in picks])
        rows = tile_slide_index(pd.DataFrame(picks), slide_stats)
        pinv, maxc = stain_tables(slide_stats)
        with torch.inference_mode():
            model = encoder.module if isinstance(encoder, nn.DataParallel) else encoder
            fixed = model.restain(raw.cuda(), pinv[rows.cuda()], maxc[rows.cuda()]).cpu()
        fig, axes = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6.4))
        for column, (tile, normalised, image_id) in enumerate(zip(raw, fixed, examples.image_id)):
            axes[0, column].imshow(tile.permute(1, 2, 0).numpy())
            axes[0, column].set_title(f"{image_id[:8]} as scanned", fontsize=8)
            axes[1, column].imshow(normalised.permute(1, 2, 0).numpy().astype(np.uint8))
            axes[1, column].set_title("Macenko-normalised", fontsize=8)
        for ax in axes.ravel():
            ax.axis("off")
        plt.tight_layout()
        plt.show()
    est_tiles = len(tiles) / len(slide_stats) * len(train)
    est_tiling_h = tiling_minutes / 60 / len(slide_stats) * len(train)
    est_encode_h = est_tiles / tiles_per_sec / 3600
    print(f"throughput: {tiles_per_sec:.0f} tiles/s")
    print(f"full run estimate: {len(train):,} slides, {est_tiles:,.0f} tiles")
    print(f"  tiling ~{est_tiling_h:.1f} h + encoding ~{est_encode_h:.1f} h = ~{est_tiling_h + est_encode_h:.1f} h "
          f"(plus a few minutes per upload)")
    print(f"  features ~{est_tiles * FEATURE_DIM * 2 / 1e9:.1f} GB (the /kaggle/working limit is 20 GB)")
    if est_tiling_h + est_encode_h > 11:
        print("  more than one 12-hour run: that's fine, just commit again after the first one stops")
else:
    print("DEBUG_SLIDES is None: skipping the dry run")

# %% [markdown]
# ## Extract all parts, uploading as we go
#
# A part counts as finished once its `slides_partNNN.csv` exists; that file is written last.
# Finished parts from earlier runs are downloaded first and skipped.

# %%
config = {"model": MODEL_NAME, "stain_normalise": STAIN_NORMALISE,
          "feature_dim": FEATURE_DIM, "tile_level": TILE_LEVEL, "tile_size": TILE_SIZE,
          "min_tissue": MIN_TISSUE, "max_tiles": MAX_TILES, "part_slides": PART_SLIDES,
          "n_parts": len(parts), "n_slides": len(slides)}

uploaded = False
if not PERSIST:
    print("Dry run finished. Set DEBUG_SLIDES = None and use Save Version → Save & Run All.")
else:
    restore(SAVE_DIR)
    config_path = SAVE_DIR / "extract_config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        for key, value in config.items():
            assert saved[key] == value, (f"{key} is {value} but the saved parts used {saved[key]}; "
                                         f"restore the old setting or use a new DATASET_SLUG")
    config_path.write_text(json.dumps(config, indent=2))
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    TMP_DIR.mkdir(parents=True)

    run_start = last_upload = time.time()
    for k, part in enumerate(parts):
        name = f"part{k:03d}"
        if (SAVE_DIR / f"slides_{name}.csv").exists():
            print(f"{name}: already saved")
            continue

        part_start = time.time()
        tiles, slide_stats, failed = tile_slides(part)
        features = np.lib.format.open_memmap(TMP_DIR / f"features_{name}.npy", mode="w+",
                                             dtype=np.float16, shape=(len(tiles), FEATURE_DIM))
        stains = stain_tables(slide_stats) if STAIN_NORMALISE else None
        index = tile_slide_index(tiles, slide_stats) if STAIN_NORMALISE else None
        written, tiles_per_sec = encode(make_loader(tiles, index), sink=features, stains=stains)
        features.flush()
        assert written == len(tiles), (written, len(tiles))
        assert np.isfinite(features[:: max(len(tiles) // 1000, 1)]).all(), "NaN/inf in features"
        del features

        shutil.move(TMP_DIR / f"features_{name}.npy", SAVE_DIR / f"features_{name}.npy")
        tiles.to_parquet(SAVE_DIR / f"tiles_{name}.parquet", index=False)
        (SAVE_DIR / f"failed_{name}.json").write_text(json.dumps(failed, indent=2))
        slide_stats.to_csv(SAVE_DIR / f"slides_{name}.csv", index=False)  # last: marks the part finished
        print(f"{name}: saved {len(slide_stats)} slides, {len(tiles):,} tiles, {len(failed)} skipped, "
              f"{tiles_per_sec:.0f} tiles/s, {(time.time() - part_start) / 60:.1f} min")
        for failure in failed[:5]:
            print("   skipped:", failure)

        is_last = k == len(parts) - 1
        # Upload after the first part too, so an early crash costs at most one part.
        if not is_last and (k == 0 or time.time() - last_upload > CHECKPOINT_MINUTES * 60):
            publish(SAVE_DIR, f"parts 0-{k} of {len(parts)}")
            last_upload = time.time()

    uploaded = publish(SAVE_DIR, f"all {len(parts)} parts")
    print(f"total time this run: {(time.time() - run_start) / 3600:.2f} h")

# %% [markdown]
# ## Confirm the upload
#
# Kaggle takes a few minutes to process a new version. This waits for it, then lists the files
# stored in your account.
#
# To download everything to your own machine (after `pip install kaggle` and putting
# `kaggle.json` in `~/.kaggle/`):
#
# ```
# kaggle datasets download kumaarbalbir/panda-phikon-features -p data/pda-ft --unzip
# ```

# %%
if PERSIST and not uploaded:
    print("The upload did not succeed. Everything is still in this version's output: open the version, "
          "go to Output, and download it, or fix the problem and commit again to carry on from here.")
elif PERSIST:
    for _ in range(30):
        try:
            if dataset_status() == "ready":
                break
        except RuntimeError as e:
            print(e)
        time.sleep(30)
    ok, listing = kaggle_cli("datasets", "files", DATASET_ID, "--page-size", "200")
    if not ok:  # older CLI without --page-size
        ok, listing = kaggle_cli("datasets", "files", DATASET_ID)
    print(listing)
    local = sorted(p.name for p in SAVE_DIR.iterdir() if p.name != "dataset-metadata.json")
    missing = [name for name in local if name not in listing]
    if missing:
        print(f"WARNING: not listed yet (may be processing or a truncated listing): {missing}\n"
              f"Check https://www.kaggle.com/datasets/{DATASET_ID}")
    else:
        print(f"all {len(local)} files are in https://www.kaggle.com/datasets/{DATASET_ID}")
