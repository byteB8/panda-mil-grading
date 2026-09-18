"""Attention-MIL training on precomputed tile features.

Reads the features produced by notebooks/01_extract_features.ipynb and runs:

  1. 5-fold cross-validation: attention MIL (ABMIL) vs a mean-pooling baseline, scored with
     quadratic weighted kappa (QWK), the metric the PANDA competition used
  2. cross-centre robustness: train on Radboud, test on Karolinska, and the reverse
  3. localisation: does attention land on the tiles the masks call cancer?
  4. quantification: a tile-level cancer detector turned into a per-slide cancer-area estimate
  5. attention heatmaps (needs the slide images; skipped without --slides-dir)

Results go to --out. Finished folds are skipped when rerun, so an interrupted run continues.

  CUDA_VISIBLE_DEVICES=3 python train.py --features data/pda-ft --out results
"""

import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is set)

N_GRADES = 6


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, required=True, help="folder holding features_part*.npy etc.")
    p.add_argument("--quiet", action="store_true", help="no progress bars (for nohup logs)")
    p.add_argument("--out", type=Path, default=Path("results"), help="where results are written")
    p.add_argument("--slides-dir", type=Path, help="folder of source slide images; enables attention heatmaps")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32, help="slides per step")
    p.add_argument("--max-train-tiles", type=int, default=512, help="tiles sampled per slide per step")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


# --------------------------------------------------------------------------------------
# Loading


def load_features(features_dir):
    """Reads every extracted part into one feature tensor plus the tile and slide tables."""
    config = json.loads((features_dir / "extract_config.json").read_text())
    part_files = sorted(features_dir.glob("slides_part*.csv"))
    assert part_files, f"no slides_part*.csv in {features_dir}"
    skipped = sum(len(json.loads(p.read_text())) for p in features_dir.glob("failed_part*.json"))
    if len(part_files) < config["n_parts"]:
        print(f"WARNING: only {len(part_files)} of {config['n_parts']} parts present; extraction unfinished")

    part_names = [p.stem.removeprefix("slides_") for p in part_files]
    arrays = [np.load(features_dir / f"features_{part}.npy", mmap_mode="r") for part in part_names]
    features = torch.empty((sum(len(a) for a in arrays), arrays[0].shape[1]), dtype=torch.float16)

    tile_parts, slide_parts, base = [], [], 0
    for part, feats, slides_path in zip(part_names, arrays, part_files):
        tiles_k = pd.read_parquet(features_dir / f"tiles_{part}.parquet")
        slides_k = pd.read_csv(slides_path)
        assert len(feats) == len(tiles_k), part
        features[base:base + len(feats)] = torch.from_numpy(np.asarray(feats))  # one part at a time
        slides_k["offset"] += base
        base += len(feats)
        tile_parts.append(tiles_k)
        slide_parts.append(slides_k)

    tiles = pd.concat(tile_parts, ignore_index=True)
    slides = pd.concat(slide_parts, ignore_index=True)
    assert slides.image_id.is_unique
    print(f"{features_dir}: {len(part_files)} parts, {skipped} slides skipped during tiling")
    print(f"total: {len(slides):,} slides, {len(features):,} tiles x {features.shape[1]} dims, "
          f"{features.numel() * 2 / 1e9:.1f} GB in memory")
    print(pd.crosstab(slides.isup_grade, slides.data_provider, margins=True))
    return features, tiles, slides, config


def tile_cancer_labels(frame):
    """1 = cancer tile, 0 = no cancer in the mask, NaN = no mask or in between.

    Radboud masks label glands and stroma separately, so a tile counts as cancer when at least half of its
    labelled epithelium is cancer (and cancer covers at least 10% of the tile), not half of the whole tile."""
    share = frame.cancer_frac / frame.epithelium_frac.where(frame.epithelium_frac > 0)
    labels = pd.Series(np.nan, index=frame.index, dtype=np.float32)
    labels[frame.cancer_frac == 0] = 0
    labels[(frame.cancer_frac >= 0.1) & (share >= 0.5)] = 1
    return labels


# --------------------------------------------------------------------------------------
# Models and training
#
# The grade is ordinal, so the head predicts five "grade > k" probabilities (k = 0..4) with binary
# cross-entropy. The predicted grade is their sum, rounded, which penalises predicting 5 for a 1 more
# than predicting 2 — the way QWK scores mistakes.


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


class Trainer:
    """Holds the features and settings that every experiment needs."""

    def __init__(self, features, slides, args):
        self.features = features
        self.slides = slides
        self.args = args
        self.dim = features.shape[1]
        # Per-dimension standardisation from a subsample of all tiles (no labels involved).
        sample = features[:: max(len(features) // 200_000, 1)].float()
        self.mean = sample.mean(dim=0).to(args.device)
        self.std = (sample.std(dim=0) + 1e-6).to(args.device)

    def standardise(self, x):
        return (x.float() - self.mean) / self.std

    def bag_batches(self, frame, shuffle, max_tiles=None):
        """Yields padded (features, mask, grades, row indices) batches from a slide table."""
        order = np.random.permutation(len(frame)) if shuffle else np.arange(len(frame))
        offsets, counts, grades = frame.offset.values, frame.n_tiles.values, frame.isup_grade.values
        for start in range(0, len(order), self.args.batch_size):
            rows = order[start:start + self.args.batch_size]
            bags = []
            for r in rows:
                idx = np.arange(offsets[r], offsets[r] + counts[r])
                if max_tiles and len(idx) > max_tiles:
                    idx = np.sort(np.random.choice(idx, max_tiles, replace=False))
                bags.append(self.features[torch.from_numpy(idx)])
            longest = max(len(b) for b in bags)
            x = torch.zeros(len(bags), longest, self.dim, dtype=torch.float16)
            mask = torch.zeros(len(bags), longest, dtype=torch.bool)
            for i, bag in enumerate(bags):
                x[i, :len(bag)] = bag
                mask[i, :len(bag)] = True
            yield (self.standardise(x.to(self.args.device, non_blocking=True)), mask.to(self.args.device),
                   torch.as_tensor(grades[rows], device=self.args.device), rows)

    @torch.no_grad()
    def predict(self, model, frame, keep_attention=False):
        """Scores every slide using all of its tiles."""
        model.eval()
        scores = np.zeros(len(frame))
        attention = pd.Series([None] * len(frame), dtype=object)
        for x, mask, _, rows in self.bag_batches(frame, shuffle=False):
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

    def fit(self, model_cls, train_frame, val_frame):
        """Trains with AdamW + one-cycle schedule and keeps the epoch with the best validation QWK."""
        args = self.args
        seed_everything(args.seed)
        model = model_cls(self.dim).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        steps = args.epochs * int(np.ceil(len(train_frame) / args.batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.1)
        best_qwk, best_state, history = -1.0, None, []
        for epoch in tqdm(range(args.epochs), desc=model_cls.__name__, disable=args.quiet):
            model.train()
            losses = []
            for x, mask, grades, _ in self.bag_batches(train_frame, shuffle=True, max_tiles=args.max_train_tiles):
                logits, _ = model(x, mask)
                targets = (grades.unsqueeze(1) > torch.arange(N_GRADES - 1, device=args.device)).float()
                loss = F.binary_cross_entropy_with_logits(logits, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
            val_qwk = qwk(val_frame.isup_grade, self.predict(model, val_frame).pred)
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_qwk": val_qwk})
            if val_qwk >= best_qwk:  # ties go to the later epoch
                best_qwk, best_state = val_qwk, copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        return model, pd.DataFrame(history)

    def inner_split(self, frame):
        """Holds out 10% of a training set for choosing the epoch, so test folds stay untouched."""
        strata = frame.data_provider + "_" + frame.isup_grade.astype(str)
        return train_test_split(frame, test_size=0.1, stratify=strata, random_state=self.args.seed)


# --------------------------------------------------------------------------------------
# Experiments


def cross_validation(trainer, out_dir):
    """5-fold CV of ABMIL against mean pooling. Folds are stratified by centre and grade, and each
    fold's model picks its epoch on 10% of its own training data, so test folds stay untouched."""
    args, slides = trainer.args, trainer.slides
    folds = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    strata = slides.data_provider + "_" + slides.isup_grade.astype(str)
    slides["fold"] = -1
    for k, (_, test_idx) in enumerate(folds.split(slides, strata)):
        slides.loc[test_idx, "fold"] = k

    cv_rows, oof = [], {"ABMIL": [], "MeanPool": []}
    for k in range(args.folds):
        train_frame, val_frame = trainer.inner_split(slides[slides.fold != k])
        test_frame = slides[slides.fold == k].reset_index(drop=True)
        for name, model_cls in [("ABMIL", ABMIL), ("MeanPool", MeanPool)]:
            preds_path = out_dir / f"oof_{name}_fold{k}.parquet"
            if preds_path.exists():
                preds = pd.read_parquet(preds_path)
                print(f"{name} fold {k}: loaded saved predictions")
            else:
                model, history = trainer.fit(model_cls, train_frame.reset_index(drop=True),
                                             val_frame.reset_index(drop=True))
                preds = trainer.predict(model, test_frame, keep_attention=(name == "ABMIL"))
                preds["fold"] = k
                torch.save(model.state_dict(), out_dir / f"model_{name}_fold{k}.pt")
                history.to_csv(out_dir / f"history_{name}_fold{k}.csv", index=False)
                preds.to_parquet(preds_path, index=False)  # last: marks this model and fold finished
            oof[name].append(preds)
            row = {"model": name, "fold": k, "qwk": qwk(preds.isup_grade, preds.pred)}
            for provider, part in preds.groupby("data_provider"):
                row[f"qwk_{provider}"] = qwk(part.isup_grade, part.pred)
            cv_rows.append(row)
            print(row)

    cv = pd.DataFrame(cv_rows)
    oof = {name: pd.concat(parts, ignore_index=True) for name, parts in oof.items()}
    print(cv.drop(columns="fold").groupby("model").agg(["mean", "std"]).round(4))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name in zip(axes, oof):
        cm = confusion_matrix(oof[name].isup_grade, oof[name].pred, labels=range(N_GRADES))
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(N_GRADES):
            for j in range(N_GRADES):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
        ax.set(title=f"{name} (out-of-fold QWK {qwk(oof[name].isup_grade, oof[name].pred):.3f})",
               xlabel="predicted ISUP", ylabel="true ISUP")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    return cv, oof


def cross_centre(trainer, out_dir):
    """Train on one hospital, test on both. The gap is the cost of moving to a new hospital."""
    shift_path = out_dir / "cross_centre.csv"
    if shift_path.exists():
        print("loaded saved cross-centre results")
        return pd.read_csv(shift_path)

    slides, rows = trainer.slides, []
    for source, target in [("radboud", "karolinska"), ("karolinska", "radboud")]:
        source_frame = slides[slides.data_provider == source]
        rest, in_centre = train_test_split(source_frame, test_size=0.2, stratify=source_frame.isup_grade,
                                           random_state=trainer.args.seed)
        train_frame, val_frame = trainer.inner_split(rest)
        model, history = trainer.fit(ABMIL, train_frame.reset_index(drop=True), val_frame.reset_index(drop=True))
        torch.save(model.state_dict(), out_dir / f"model_ABMIL_train_{source}.pt")
        history.to_csv(out_dir / f"history_ABMIL_train_{source}.csv", index=False)
        for split, frame in [("in-centre", in_centre), ("other centre", slides[slides.data_provider == target])]:
            preds = trainer.predict(model, frame.reset_index(drop=True))
            preds.to_csv(out_dir / f"preds_train_{source}_{split.replace(' ', '_')}.csv", index=False)
            rows.append({"train": source, "test": split, "slides": len(frame),
                         "qwk": qwk(preds.isup_grade, preds.pred)})
    shift = pd.DataFrame(rows)
    shift.to_csv(shift_path, index=False)
    print(shift.round(4))
    return shift


def attention_localisation(abmil_oof, tiles, out_dir):
    """Per slide, how well attention separates cancer tiles from cancer-free ones. The model never
    saw the masks, so this is a check rather than a training signal."""
    rows = []
    for row in abmil_oof.itertuples():
        labels = tiles.cancer_label.values[row.offset:row.offset + row.n_tiles]
        positive, negative = labels == 1, labels == 0
        if row.isup_grade == 0 or positive.sum() < 2 or negative.sum() < 2:
            continue
        keep = positive | negative
        rows.append({"image_id": row.image_id, "data_provider": row.data_provider,
                     "isup_grade": row.isup_grade, "auc": roc_auc_score(positive[keep], row.attention[keep])})
    localisation = pd.DataFrame(rows)
    localisation.to_csv(out_dir / "attention_localisation.csv", index=False)
    print(localisation.groupby("data_provider").auc.describe().round(3))
    return localisation


def cancer_quantification(trainer, tiles, out_dir):
    """A linear probe on frozen tile features, trained on mask labels from folds 1+ and tested on fold 0.
    Per slide, the share of tiles it calls cancer is compared with the share the mask calls cancer."""
    args, slides = trainer.args, trainer.slides
    tile_fold = np.repeat(slides.fold.values, slides.n_tiles.values)
    tile_order = np.concatenate([np.arange(o, o + n) for o, n in zip(slides.offset, slides.n_tiles)])
    tiles["fold"] = -1
    tiles.loc[tile_order, "fold"] = tile_fold

    labelled = tiles.cancer_label.notna()
    train_idx = np.flatnonzero(labelled & (tiles.fold != 0))
    test_idx = np.flatnonzero(labelled & (tiles.fold == 0))

    seed_everything(args.seed)
    probe = nn.Linear(trainer.dim, 1).to(args.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    tile_labels = torch.from_numpy((tiles.cancer_label.values == 1).astype(np.float32))
    for epoch in range(3):
        perm = torch.from_numpy(np.random.permutation(train_idx))
        for start in tqdm(range(0, len(perm), 4096), desc=f"probe epoch {epoch}", disable=args.quiet):
            batch = perm[start:start + 4096]
            logits = probe(trainer.standardise(trainer.features[batch].to(args.device))).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, tile_labels[batch].to(args.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        probs = torch.cat([
            probe(trainer.standardise(trainer.features[torch.from_numpy(test_idx[s:s + 16384])].to(args.device)))
            .sigmoid().squeeze(1).cpu() for s in range(0, len(test_idx), 16384)]).numpy()
    test_tiles = tiles.iloc[test_idx].assign(prob=probs)
    tile_auc = roc_auc_score(test_tiles.cancer_label == 1, test_tiles.prob)

    per_slide = test_tiles.groupby("slide_id").agg(true_cancer=("cancer_label", "mean"),
                                                   pred_cancer=("prob", lambda p: (p >= 0.5).mean()))
    area_r, _ = pearsonr(per_slide.true_cancer, per_slide.pred_cancer)
    per_slide.to_csv(out_dir / "cancer_area_fold0.csv")
    torch.save(probe.state_dict(), out_dir / "tile_cancer_probe.pt")
    print(f"tile AUC {tile_auc:.3f} on {len(test_tiles):,} fold-0 tiles; "
          f"slide cancer-area Pearson r {area_r:.3f} over {len(per_slide):,} slides")

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(per_slide.true_cancer, per_slide.pred_cancer, s=4, alpha=0.4)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="share of labelled tiles that are cancer (mask)", ylabel="share predicted cancer",
           title=f"Fold 0 slides, r = {area_r:.2f}")
    fig.tight_layout()
    fig.savefig(out_dir / "cancer_area.png", dpi=150)
    plt.close(fig)
    return float(tile_auc), float(area_r)


def attention_heatmaps(abmil_oof, tiles, localisation, config, slides_dir, out_dir):
    """Slide, attention (from the fold model that did not train on it), and the mask's cancer tiles."""
    import openslide

    candidates = localisation[localisation.isup_grade >= 3].sort_values("auc", ascending=False)
    if candidates.empty:
        print("no slides to draw heatmaps for")
        return
    examples = pd.concat([candidates.head(2), candidates.tail(1)]).image_id
    indexed = abmil_oof.set_index("image_id")
    fig, axes = plt.subplots(len(examples), 3, figsize=(15, 5 * len(examples)))
    for axes_row, image_id in zip(np.atleast_2d(axes), examples):
        row = indexed.loc[image_id]
        slide_tiles = tiles.iloc[row.offset:row.offset + row.n_tiles]
        with openslide.OpenSlide(str(slides_dir / f"{image_id}.tiff")) as slide:
            thumb_level = slide.level_count - 1
            w, h = slide.level_dimensions[thumb_level]
            thumb = np.asarray(slide.read_region((0, 0), thumb_level, (w, h)).convert("RGB"))
            ds = slide.level_downsamples[thumb_level]
            side = int(np.ceil(config["tile_size"] * slide.level_downsamples[config["tile_level"]] / ds))
        attention_map = np.full((h, w), np.nan)
        cancer_map = np.full((h, w), np.nan)
        ranks = row.attention.argsort().argsort() / max(len(row.attention) - 1, 1)
        for t, rank in zip(slide_tiles.itertuples(), ranks):
            x, y = int(t.x / ds), int(t.y / ds)
            attention_map[y:y + side, x:x + side] = rank
            cancer_map[y:y + side, x:x + side] = t.cancer_label
        axes_row[0].imshow(thumb)
        axes_row[0].set_title(f"{row.data_provider} | ISUP {row.isup_grade} | predicted {row.pred}")
        axes_row[1].imshow(thumb)
        axes_row[1].imshow(attention_map, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes_row[1].set_title(f"attention (localisation AUC {localisation.set_index('image_id').auc[image_id]:.2f})")
        axes_row[2].imshow(thumb)
        axes_row[2].imshow(cancer_map, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
        axes_row[2].set_title("cancer from mask")
        for ax in axes_row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "attention_heatmaps.png", dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device: {args.device}")

    features, tiles, slides, config = load_features(args.features)
    tiles["cancer_label"] = tile_cancer_labels(tiles)
    print("tile cancer labels:", tiles.groupby("cancer_label", dropna=False).size().to_dict())

    settings = {k: v for k, v in vars(args).items() if k not in ("features", "out", "slides_dir")}
    train_config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in settings.items()},
                    "n_slides": len(slides), "n_tiles": int(len(features)), "features_config": config}
    config_path = args.out / "train_config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        assert saved == json.loads(json.dumps(train_config)), (
            f"settings or features differ from the results already in {args.out}; use a different --out")
    config_path.write_text(json.dumps(train_config, indent=2))

    trainer = Trainer(features, slides, args)
    cv, oof = cross_validation(trainer, args.out)
    shift = cross_centre(trainer, args.out)

    abmil_oof = oof["ABMIL"].merge(slides[["image_id", "offset", "n_tiles"]], on="image_id")
    localisation = attention_localisation(abmil_oof, tiles, args.out)
    tile_auc, area_r = cancer_quantification(trainer, tiles, args.out)
    if args.slides_dir:
        attention_heatmaps(abmil_oof, tiles, localisation, config, args.slides_dir, args.out)
    else:
        print("no --slides-dir: skipping attention heatmaps")

    results = {
        "cv": cv.to_dict(orient="records"),
        "cv_summary": {m: {"qwk_mean": float(g.qwk.mean()), "qwk_std": float(g.qwk.std())}
                       for m, g in cv.groupby("model")},
        "oof_qwk": {name: float(qwk(p.isup_grade, p.pred)) for name, p in oof.items()},
        "cross_centre": shift.to_dict(orient="records"),
        "attention_localisation_auc_median": localisation.groupby("data_provider").auc.median().to_dict(),
        "tile_cancer_auc": tile_auc,
        "slide_cancer_area_pearson_r": area_r,
        "train_config": train_config,
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    oof["ABMIL"].drop(columns="attention").to_csv(args.out / "oof_abmil.csv", index=False)
    oof["MeanPool"].to_csv(args.out / "oof_meanpool.csv", index=False)
    print(json.dumps({k: v for k, v in results.items() if k not in ("cv", "train_config")}, indent=2, default=float))
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
