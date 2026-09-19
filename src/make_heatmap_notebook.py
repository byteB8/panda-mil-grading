"""Writes src/03_heatmaps.py with a few slides' attention baked in.

The heatmaps need the slide images, which only exist on Kaggle, while the attention lives in the
training results here. Rather than upload 40 MB of results, this picks a handful of example slides
and embeds their tile coordinates, attention and mask labels in the notebook source, so the
notebook needs nothing but the competition data.

    python src/make_heatmap_notebook.py && python src/build_notebooks.py
"""

import base64
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FEATURES = ROOT / "data" / "pda-ft"
N_TYPICAL_SPREAD = 0.01  # how close to the median AUC a "typical" example must be
MIN_TILES = 250          # skip slides too small to show much


def load_oof():
    frames = [pd.read_parquet(p) for p in sorted(RESULTS.glob("oof_ABMIL_fold*.parquet"))]
    return pd.concat(frames, ignore_index=True)


def load_tiles():
    parts = [pd.read_parquet(p) for p in sorted(FEATURES.glob("tiles_part*.parquet"))]
    tiles = pd.concat(parts, ignore_index=True)
    share = tiles.cancer_frac / tiles.epithelium_frac.where(tiles.epithelium_frac > 0)
    tiles["cancer_label"] = np.where(tiles.cancer_frac == 0, 0,
                                     np.where((tiles.cancer_frac >= 0.1) & (share >= 0.5), 1, -1))
    tiles.loc[tiles.cancer_frac.isna(), "cancer_label"] = -1  # no mask
    return tiles


def pick_slides(oof, localisation, tile_counts):
    """A few slides that show what the attention does, including one failure."""
    table = localisation.merge(oof[["image_id", "pred", "score"]], on="image_id")
    table = table[table.image_id.map(tile_counts) >= MIN_TILES]  # thin slides make thin pictures
    correct = table[table.isup_grade == table.pred]
    chosen = []

    for provider in ["radboud", "karolinska"]:  # one strong example per centre
        part = correct[(correct.data_provider == provider) & (correct.isup_grade >= 4)]
        chosen.append((part.sort_values("auc", ascending=False).iloc[0], "high grade, attention agrees with the mask"))

    median = table.auc.median()
    typical = table[(table.auc - median).abs() < N_TYPICAL_SPREAD].sort_values("isup_grade", ascending=False)
    chosen.append((typical.iloc[0], f"typical slide (median localisation AUC {median:.2f})"))

    worst = table[table.isup_grade >= 3].sort_values("auc").iloc[0]
    chosen.append((worst, "failure case: attention avoids the mask's cancer"))

    benign = oof[(oof.isup_grade == 0) & (oof.pred == 0) & (oof.image_id.map(tile_counts) >= MIN_TILES)].iloc[0]
    benign = benign.copy()
    benign["auc"] = np.nan
    chosen.append((benign, "benign slide (ISUP 0), for contrast"))
    return chosen


def main():
    oof = load_oof()
    localisation = pd.read_csv(RESULTS / "attention_localisation.csv")
    tiles = load_tiles()
    config = json.loads((FEATURES / "extract_config.json").read_text())
    attention_by_slide = dict(zip(oof.image_id, oof.attention))

    payload = {"tile_size": config["tile_size"], "tile_level": config["tile_level"], "slides": []}
    tile_counts = tiles.groupby("slide_id").size()
    for row, note in pick_slides(oof, localisation, tile_counts):
        slide_tiles = tiles[tiles.slide_id == row.image_id]
        attention = np.asarray(attention_by_slide[row.image_id], dtype=np.float32)
        assert len(attention) == len(slide_tiles), (row.image_id, len(attention), len(slide_tiles))
        payload["slides"].append({
            "image_id": row.image_id,
            "provider": row.data_provider,
            "isup": int(row.isup_grade),
            "pred": int(row.pred),
            "auc": None if pd.isna(row.auc) else round(float(row.auc), 3),
            "note": note,
            "x": slide_tiles.x.astype(int).tolist(),
            "y": slide_tiles.y.astype(int).tolist(),
            "attention": [round(float(a), 7) for a in attention],
            "label": slide_tiles.cancer_label.astype(int).tolist(),
        })
        print(f"{row.image_id}  {row.data_provider:11s} ISUP {row.isup_grade}->{row.pred}  "
              f"tiles {len(slide_tiles):5d}  auc {row.auc if not pd.isna(row.auc) else float('nan'):.3f}  {note}")

    blob = base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode()
    print(f"\npayload: {len(blob) / 1000:.0f} kB for {len(payload['slides'])} slides")

    source = (ROOT / "src" / "03_heatmaps_template.py").read_text()
    assert 'PAYLOAD = ""' in source
    wrapped = "\n".join(blob[i:i + 120] for i in range(0, len(blob), 120))
    out = source.replace('PAYLOAD = ""', f'PAYLOAD = """\\\n{wrapped}"""')
    (ROOT / "src" / "03_heatmaps.py").write_text(out)
    print("wrote src/03_heatmaps.py")


if __name__ == "__main__":
    main()
