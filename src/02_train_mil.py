# %% [markdown]
# # PANDA 2/2 — Slide-level ISUP grading with attention MIL
#
# Trains on the tile features from notebook 1. Each slide is a "bag" of tile vectors; the
# model learns which tiles matter (attention) and predicts the ISUP grade (0–5).
#
# Experiments:
# 1. **5-fold cross-validation:** gated attention MIL (ABMIL) vs a mean-pooling baseline,
#    scored with quadratic weighted kappa (QWK), the competition metric
# 2. **Cross-centre robustness:** train on Radboud, test on Karolinska, and the reverse
# 3. **Localisation:** does attention land on the tiles the masks call cancer?
# 4. **Quantification:** a tile-level cancer detector, turned into a per-slide cancer-area estimate
# 5. **Attention heatmaps** for a few slides
#
# **Results are saved to a private Kaggle Dataset in your account** (`kumaarbalbir/panda-mil-results`):
# predictions, model weights and training curves after every fold, then the remaining
# experiments and figures. If the run stops, commit again: finished folds are downloaded and skipped.
#
# **Kaggle settings:**
# - Accelerator: GPU (one T4 or P100 is plenty)
# - Internet: **On** (OpenSlide install and uploads)
# - Secrets: tick the same `KAGGLE_KEY` secret as in notebook 1 (Add-ons → Secrets)
# - Inputs: the competition data **plus** your dataset **kumaarbalbir/panda-phikon-features**
#   (Add Input → Datasets → Your Datasets)
#
# Then **Save Version → Save & Run All (Commit)**.

# %%
!pip install -q -U openslide-bin openslide-python kaggle

# %%
import copy
import json
import os
import random
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm.auto import tqdm



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
INPUT_ROOT = Path("/kaggle/input")
OUT_DIR = Path("/kaggle/working")

N_FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 32
MAX_TRAIN_TILES = 512   # random tiles per slide per step; also acts as augmentation
LR = 2e-4
WEIGHT_DECAY = 1e-4
SEED = 0
N_GRADES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

KAGGLE_USERNAME = "kumaarbalbir"
DATASET_SLUG = "panda-mil-results"
DATASET_TITLE = "PANDA MIL grading results"
DATASET_ID = f"{KAGGLE_USERNAME.lower()}/{DATASET_SLUG}"  # replaced below by your real username
RESULTS_DIR = OUT_DIR / DATASET_SLUG  # everything worth keeping goes here and gets uploaded


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

# %% [markdown]
# ## Saving to your Kaggle account
#
# Same helpers as notebook 1. Each upload replaces the previous version of the results dataset.

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


kaggle_login()
ok, output = kaggle_cli("datasets", "list", "--mine")
assert ok, f"Kaggle credentials rejected: {output}"
DATASET_ID = f"{detect_owner()}/{DATASET_SLUG}"
print(f"Kaggle API login OK; results will go to {DATASET_ID}")
restore(RESULTS_DIR)

# %% [markdown]
# ## Load the features

# %%
config_files = sorted(INPUT_ROOT.rglob("extract_config.json"))
assert len(config_files) == 1, f"expected one attached features dataset, found {config_files}"
FEATURES_DIR = config_files[0].parent
config = json.loads(config_files[0].read_text())
part_files = sorted(FEATURES_DIR.glob("slides_part*.csv"))
failed = sum(len(json.loads(p.read_text())) for p in FEATURES_DIR.glob("failed_part*.json"))
if len(part_files) < config["n_parts"]:
    print(f"WARNING: only {len(part_files)} of {config['n_parts']} parts extracted; notebook 1 has not finished")

part_names = [p.stem.removeprefix("slides_") for p in part_files]
part_arrays = [np.load(FEATURES_DIR / f"features_{part}.npy", mmap_mode="r") for part in part_names]
features = torch.empty((sum(len(a) for a in part_arrays), part_arrays[0].shape[1]), dtype=torch.float16)

tile_parts, slide_parts, base = [], [], 0
for part, feats, slides_path in zip(part_names, part_arrays, part_files):
    tiles_k = pd.read_parquet(FEATURES_DIR / f"tiles_{part}.parquet")
    slides_k = pd.read_csv(slides_path)
    assert len(feats) == len(tiles_k), part
    features[base:base + len(feats)] = torch.from_numpy(np.asarray(feats))  # one part in memory at a time
    slides_k["offset"] += base
    base += len(feats)
    tile_parts.append(tiles_k)
    slide_parts.append(slides_k)
print(f"{FEATURES_DIR}: {len(part_files)} parts, {failed} slides failed tiling")

tiles = pd.concat(tile_parts, ignore_index=True)
slides = pd.concat(slide_parts, ignore_index=True)
del part_arrays


def tile_cancer_labels(frame):
    """1 = cancer tile, 0 = no cancer in the mask, NaN = no mask or in between.

    Radboud masks label glands and stroma separately, so a tile counts as cancer when at least half of its
    labelled epithelium is cancer (and cancer covers at least 10% of the tile), not half of the whole tile."""
    share = frame.cancer_frac / frame.epithelium_frac.where(frame.epithelium_frac > 0)
    labels = pd.Series(np.nan, index=frame.index, dtype=np.float32)
    labels[frame.cancer_frac == 0] = 0
    labels[(frame.cancer_frac >= 0.1) & (share >= 0.5)] = 1
    return labels


tiles["cancer_label"] = tile_cancer_labels(tiles)
print("tile cancer labels:", tiles.groupby(["cancer_label"], dropna=False).size().to_dict())
assert slides.image_id.is_unique

FEATURE_DIM = features.shape[1]
# Per-dimension standardisation from a subsample of all tiles (no labels involved).
sample = features[:: max(len(features) // 200_000, 1)].float()
FEATURE_MEAN = sample.mean(dim=0).to(DEVICE)
FEATURE_STD = (sample.std(dim=0) + 1e-6).to(DEVICE)
del sample
TILE_LEVEL, TILE_SIZE = config["tile_level"], config["tile_size"]
print(f"total: {len(slides):,} slides, {features.shape[0]:,} tiles x {FEATURE_DIM} dims, "
      f"{features.numel() * 2 / 1e9:.1f} GB in RAM")
print(pd.crosstab(slides.isup_grade, slides.data_provider, margins=True))

train_config = {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "max_train_tiles": MAX_TRAIN_TILES, "lr": LR,
                "weight_decay": WEIGHT_DECAY, "seed": SEED, "folds": N_FOLDS,
                "n_slides": len(slides), "n_tiles": int(features.shape[0]), "features": config}
config_path = RESULTS_DIR / "train_config.json"
if config_path.exists():
    saved = json.loads(config_path.read_text())
    assert saved == json.loads(json.dumps(train_config)), (
        "settings or features differ from the saved results; restore them or use a new DATASET_SLUG")
config_path.write_text(json.dumps(train_config, indent=2))

# %% [markdown]
# ## Bags, models and the training loop
#
# The grade is ordinal, so the head predicts five "grade > k" probabilities (k = 0..4) with
# binary cross-entropy. The predicted grade is their sum, rounded. This penalises
# predicting 5 for a 1 more than predicting 2, which matches how QWK scores mistakes.

# %%
def standardise(x):
    return (x.float() - FEATURE_MEAN) / FEATURE_STD


def bag_batches(frame, shuffle, max_tiles=None):
    """Yields padded (features, mask, grades, row indices) batches from a slide table."""
    order = np.random.permutation(len(frame)) if shuffle else np.arange(len(frame))
    offsets, counts, grades = frame.offset.values, frame.n_tiles.values, frame.isup_grade.values
    for start in range(0, len(order), BATCH_SIZE):
        rows = order[start:start + BATCH_SIZE]
        bags = []
        for r in rows:
            idx = np.arange(offsets[r], offsets[r] + counts[r])
            if max_tiles and len(idx) > max_tiles:
                idx = np.sort(np.random.choice(idx, max_tiles, replace=False))
            bags.append(features[torch.from_numpy(idx)])
        longest = max(len(b) for b in bags)
        x = torch.zeros(len(bags), longest, FEATURE_DIM, dtype=torch.float16)
        mask = torch.zeros(len(bags), longest, dtype=torch.bool)
        for i, bag in enumerate(bags):
            x[i, :len(bag)] = bag
            mask[i, :len(bag)] = True
        yield (standardise(x.to(DEVICE, non_blocking=True)), mask.to(DEVICE),
               torch.as_tensor(grades[rows], device=DEVICE), rows)


class ABMIL(nn.Module):
    """Gated attention MIL (Ilse et al., 2018)."""

    def __init__(self, in_dim, hidden=256, dropout=0.25):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.attn_tanh = nn.Linear(hidden, hidden // 2)
        self.attn_gate = nn.Linear(hidden, hidden // 2)
        self.attn_out = nn.Linear(hidden // 2, 1)
        self.head = nn.Linear(hidden, N_GRADES - 1)

    def forward(self, x, mask):
        h = self.embed(x)
        scores = self.attn_out(torch.tanh(self.attn_tanh(h)) * torch.sigmoid(self.attn_gate(h))).squeeze(-1)
        attention = scores.masked_fill(~mask, float("-inf")).softmax(dim=1)
        slide_vector = (attention.unsqueeze(-1) * h).sum(dim=1)
        return self.head(slide_vector), attention


class MeanPool(nn.Module):
    """Baseline: every tile counts equally."""

    def __init__(self, in_dim, hidden=256, dropout=0.25):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, N_GRADES - 1)

    def forward(self, x, mask):
        weights = mask.float() / mask.sum(dim=1, keepdim=True)
        slide_vector = (weights.unsqueeze(-1) * self.embed(x)).sum(dim=1)
        return self.head(slide_vector), weights


def ordinal_targets(grades):
    return (grades.unsqueeze(1) > torch.arange(N_GRADES - 1, device=grades.device)).float()


@torch.no_grad()
def predict(model, frame, keep_attention=False):
    """Scores every slide using all of its tiles."""
    model.eval()
    scores = np.zeros(len(frame))
    attention = pd.Series([None] * len(frame), dtype=object)
    for x, mask, _, rows in bag_batches(frame, shuffle=False):
        logits, attn = model(x, mask)
        scores[rows] = logits.sigmoid().sum(dim=1).cpu().numpy()
        if keep_attention:
            for i, r in enumerate(rows):
                attention.iat[r] = attn[i, :mask[i].sum()].cpu().numpy()
    out = frame[["image_id", "data_provider", "isup_grade"]].copy()
    out["score"] = scores
    out["pred"] = np.clip(np.rint(scores), 0, N_GRADES - 1).astype(int)
    if keep_attention:
        out["attention"] = attention.values
    return out


def fit(model_cls, train_frame, val_frame, seed=SEED):
    """Trains with AdamW + cosine decay and returns the epoch with the best validation QWK."""
    seed_everything(seed)
    model = model_cls(FEATURE_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = EPOCHS * int(np.ceil(len(train_frame) / BATCH_SIZE))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=steps, pct_start=0.1)
    best_qwk, best_state, history = -1.0, None, []
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for x, mask, grades, _ in bag_batches(train_frame, shuffle=True, max_tiles=MAX_TRAIN_TILES):
            logits, _ = model(x, mask)
            loss = F.binary_cross_entropy_with_logits(logits, ordinal_targets(grades))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        val_qwk = qwk(val_frame.isup_grade, predict(model, val_frame).pred)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_qwk": val_qwk})
        if val_qwk >= best_qwk:  # ties go to the later epoch
            best_qwk, best_state = val_qwk, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def inner_split(frame, seed=SEED):
    """Holds out 10% of a training set for choosing the epoch, so test folds stay untouched."""
    strata = frame.data_provider + "_" + frame.isup_grade.astype(str)
    return train_test_split(frame, test_size=0.1, stratify=strata, random_state=seed)

# %% [markdown]
# ## Experiment 1 — 5-fold cross-validation
#
# Folds are stratified by centre and grade. Each fold's model picks its epoch on a 10% slice
# of its own training data and is scored once on the untouched fold. The spread across folds
# shows how big a difference between models has to be before it means anything.

# %%
folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
strata = slides.data_provider + "_" + slides.isup_grade.astype(str)
slides["fold"] = -1
for k, (_, test_idx) in enumerate(folds.split(slides, strata)):
    slides.loc[test_idx, "fold"] = k

cv_rows, oof = [], {name: [] for name in ["ABMIL", "MeanPool"]}
for k in range(N_FOLDS):
    train_frame, val_frame = inner_split(slides[slides.fold != k])
    test_frame = slides[slides.fold == k].reset_index(drop=True)
    trained_now = False
    for name, model_cls in [("ABMIL", ABMIL), ("MeanPool", MeanPool)]:
        preds_path = RESULTS_DIR / f"oof_{name}_fold{k}.parquet"
        if preds_path.exists():
            preds = pd.read_parquet(preds_path)
            print(f"{name} fold {k}: loaded saved predictions")
        else:
            model, history = fit(model_cls, train_frame.reset_index(drop=True), val_frame.reset_index(drop=True))
            preds = predict(model, test_frame, keep_attention=(name == "ABMIL"))
            preds["fold"] = k
            torch.save(model.state_dict(), RESULTS_DIR / f"model_{name}_fold{k}.pt")
            history.to_csv(RESULTS_DIR / f"history_{name}_fold{k}.csv", index=False)
            preds.to_parquet(preds_path, index=False)  # last: marks this model and fold finished
            trained_now = True
        oof[name].append(preds)
        row = {"model": name, "fold": k, "qwk": qwk(preds.isup_grade, preds.pred)}
        for provider, part in preds.groupby("data_provider"):
            row[f"qwk_{provider}"] = qwk(part.isup_grade, part.pred)
        cv_rows.append(row)
        print(row)
    if trained_now:
        publish(RESULTS_DIR, f"cross-validation fold {k} of {N_FOLDS}")

cv = pd.DataFrame(cv_rows)
oof = {name: pd.concat(parts, ignore_index=True) for name, parts in oof.items()}
summary = cv.drop(columns="fold").groupby("model").agg(["mean", "std"]).round(4)
print(summary)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, name in zip(axes, oof):
    cm = confusion_matrix(oof[name].isup_grade, oof[name].pred, labels=range(N_GRADES))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(N_GRADES):
        for j in range(N_GRADES):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set(title=f"{name} (out-of-fold QWK {qwk(oof[name].isup_grade, oof[name].pred):.3f})",
           xlabel="predicted ISUP", ylabel="true ISUP")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
plt.show()

# %% [markdown]
# ## Experiment 2 — Cross-centre robustness
#
# The two centres differ in scanner, staining and how grades were assigned. Hold out 20% of
# the source centre as an in-centre test set, train on the rest, then score both. The gap
# between the two numbers is the cost of moving to a new hospital.

# %%
shift_path = RESULTS_DIR / "cross_centre.csv"
if shift_path.exists():
    shift = pd.read_csv(shift_path)
    print("loaded saved cross-centre results")
else:
    shift_rows = []
    for source, target in [("radboud", "karolinska"), ("karolinska", "radboud")]:
        source_frame = slides[slides.data_provider == source]
        rest, in_centre = train_test_split(source_frame, test_size=0.2, stratify=source_frame.isup_grade,
                                           random_state=SEED)
        train_frame, val_frame = inner_split(rest)
        model, history = fit(ABMIL, train_frame.reset_index(drop=True), val_frame.reset_index(drop=True))
        torch.save(model.state_dict(), RESULTS_DIR / f"model_ABMIL_train_{source}.pt")
        history.to_csv(RESULTS_DIR / f"history_ABMIL_train_{source}.csv", index=False)
        for split, frame in [("in-centre", in_centre), ("other centre", slides[slides.data_provider == target])]:
            preds = predict(model, frame.reset_index(drop=True))
            preds.to_csv(RESULTS_DIR / f"preds_train_{source}_{split.replace(' ', '_')}.csv", index=False)
            shift_rows.append({"train": source, "test": split, "slides": len(frame),
                               "qwk": qwk(preds.isup_grade, preds.pred)})
    shift = pd.DataFrame(shift_rows)
    shift.to_csv(shift_path, index=False)
    publish(RESULTS_DIR, "cross-centre experiment")
print(shift.round(4))

# %% [markdown]
# ## Experiment 3 — Does attention find the cancer?
#
# For every out-of-fold slide with a mask and some cancer, compare the attention on cancer
# tiles (see `tile_cancer_labels`) with tiles that have no cancer. AUC 0.5 means attention ignores
# cancer; 1.0 means every cancer tile gets more attention than every benign tile. The model
# never saw the masks, so this is a check, not a training signal.

# %%
abmil_oof = oof["ABMIL"].merge(slides[["image_id", "offset", "n_tiles"]], on="image_id")
localisation = []
for row in abmil_oof.itertuples():
    labels = tiles.cancer_label.values[row.offset:row.offset + row.n_tiles]
    positive, negative = labels == 1, labels == 0
    if row.isup_grade == 0 or positive.sum() < 2 or negative.sum() < 2:
        continue
    keep = positive | negative
    localisation.append({"image_id": row.image_id, "data_provider": row.data_provider,
                         "isup_grade": row.isup_grade,
                         "auc": roc_auc_score(positive[keep], row.attention[keep])})
localisation = pd.DataFrame(localisation)
localisation.to_csv(RESULTS_DIR / "attention_localisation.csv", index=False)
print(localisation.groupby("data_provider").auc.describe().round(3))

# %% [markdown]
# ## Experiment 4 — Tile-level cancer detection and cancer-area quantification
#
# A linear probe on frozen tile features, trained on mask labels from folds 1–4 and tested
# on fold 0. Per slide, the share of tiles it calls cancer is compared with the share the mask calls cancer.

# %%
tile_fold = np.repeat(slides.fold.values, slides.n_tiles.values)
tile_order = np.concatenate([np.arange(o, o + n) for o, n in zip(slides.offset, slides.n_tiles)])
tiles["fold"] = -1
tiles.loc[tile_order, "fold"] = tile_fold

labelled = tiles.cancer_label.notna()
train_idx = np.flatnonzero(labelled & (tiles.fold != 0))
test_idx = np.flatnonzero(labelled & (tiles.fold == 0))

seed_everything(SEED)
probe = nn.Linear(FEATURE_DIM, 1).to(DEVICE)
optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
tile_labels = torch.from_numpy((tiles.cancer_label.values == 1).astype(np.float32))
for epoch in range(3):
    perm = torch.from_numpy(np.random.permutation(train_idx))
    for start in tqdm(range(0, len(perm), 4096), desc=f"probe epoch {epoch}"):
        batch = perm[start:start + 4096]
        logits = probe(standardise(features[batch].to(DEVICE))).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, tile_labels[batch].to(DEVICE))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    probs = torch.cat([probe(standardise(features[torch.from_numpy(test_idx[s:s + 16384])].to(DEVICE))).sigmoid().squeeze(1).cpu()
                       for s in range(0, len(test_idx), 16384)]).numpy()
test_tiles = tiles.iloc[test_idx].assign(prob=probs)
tile_auc = roc_auc_score(test_tiles.cancer_label == 1, test_tiles.prob)

per_slide = test_tiles.groupby("slide_id").agg(true_cancer=("cancer_label", "mean"),
                                               pred_cancer=("prob", lambda p: (p >= 0.5).mean()))
area_r, _ = pearsonr(per_slide.true_cancer, per_slide.pred_cancer)
per_slide.to_csv(RESULTS_DIR / "cancer_area_fold0.csv")
torch.save(probe.state_dict(), RESULTS_DIR / "tile_cancer_probe.pt")
print(f"tile AUC {tile_auc:.3f} on {len(test_tiles):,} fold-0 tiles; "
      f"slide cancer-area Pearson r {area_r:.3f} over {len(per_slide):,} slides")

plt.figure(figsize=(4.5, 4.5))
plt.scatter(per_slide.true_cancer, per_slide.pred_cancer, s=4, alpha=0.4)
plt.plot([0, 1], [0, 1], "k--", lw=0.8)
plt.xlabel("share of labelled tiles that are cancer (mask)")
plt.ylabel("share predicted cancer")
plt.title(f"Fold 0 slides, r = {area_r:.2f}")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "cancer_area.png", dpi=150)
plt.show()

# %% [markdown]
# ## Attention heatmaps
#
# Left: the slide. Middle: ABMIL attention (from the fold model that did not train on this slide; rank-scaled so the colours are
# comparable between slides). Right: cancer according to the mask.

# %%
def slide_heatmaps(image_id):
    row = abmil_oof.set_index("image_id").loc[image_id]
    slide_tiles = tiles.iloc[row.offset:row.offset + row.n_tiles]
    with openslide.OpenSlide(str(DATA_DIR / "train_images" / f"{image_id}.tiff")) as slide:
        thumb_level = slide.level_count - 1
        w, h = slide.level_dimensions[thumb_level]
        thumb = np.asarray(slide.read_region((0, 0), thumb_level, (w, h)).convert("RGB"))
        ds = slide.level_downsamples[thumb_level]
        side = int(np.ceil(TILE_SIZE * slide.level_downsamples[TILE_LEVEL] / ds))
    attention_map = np.full((h, w), np.nan)
    cancer_map = np.full((h, w), np.nan)
    ranks = row.attention.argsort().argsort() / max(len(row.attention) - 1, 1)
    for t, rank in zip(slide_tiles.itertuples(), ranks):
        x, y = int(t.x / ds), int(t.y / ds)
        attention_map[y:y + side, x:x + side] = rank
        cancer_map[y:y + side, x:x + side] = t.cancer_label
    return thumb, attention_map, cancer_map, row


candidates = localisation[localisation.isup_grade >= 3].sort_values("auc", ascending=False)
examples = pd.concat([candidates.head(2), candidates.tail(1)]).image_id
fig, axes = plt.subplots(len(examples), 3, figsize=(15, 5 * len(examples)))
for axes_row, image_id in zip(np.atleast_2d(axes), examples):
    thumb, attention_map, cancer_map, row = slide_heatmaps(image_id)
    axes_row[0].imshow(thumb)
    axes_row[0].set_title(f"{row.data_provider} | ISUP {row.isup_grade} | predicted {row.pred}")
    axes_row[1].imshow(thumb)
    axes_row[1].imshow(attention_map, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    auc = localisation.set_index("image_id").auc[image_id]
    axes_row[1].set_title(f"attention (localisation AUC {auc:.2f})")
    axes_row[2].imshow(thumb)
    axes_row[2].imshow(cancer_map, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes_row[2].set_title("cancer from mask")
    for ax in axes_row:
        ax.axis("off")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "attention_heatmaps.png", dpi=120)
plt.show()

# %% [markdown]
# ## Save and upload the final results
#
# To download everything to your own machine (after `pip install kaggle` and putting
# `kaggle.json` in `~/.kaggle/`):
#
# ```
# kaggle datasets download kumaarbalbir/panda-mil-results -p results/panda-mil-results --unzip
# ```

# %%
results = {
    "cv": cv.to_dict(orient="records"),
    "cv_summary": {m: {"qwk_mean": float(g.qwk.mean()), "qwk_std": float(g.qwk.std())} for m, g in cv.groupby("model")},
    "oof_qwk": {name: float(qwk(p.isup_grade, p.pred)) for name, p in oof.items()},
    "cross_centre": shift.to_dict(orient="records"),
    "attention_localisation_auc_median": localisation.groupby("data_provider").auc.median().to_dict(),
    "tile_cancer_auc": float(tile_auc),
    "slide_cancer_area_pearson_r": float(area_r),
    "train_config": train_config,
}
(RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float))
oof["ABMIL"].drop(columns="attention").to_csv(RESULTS_DIR / "oof_abmil.csv", index=False)
oof["MeanPool"].to_csv(RESULTS_DIR / "oof_meanpool.csv", index=False)
print(json.dumps({k: v for k, v in results.items() if k not in ("cv", "train_config")}, indent=2, default=float))
if publish(RESULTS_DIR, "final results"):
    print(f"https://www.kaggle.com/datasets/{DATASET_ID}")
