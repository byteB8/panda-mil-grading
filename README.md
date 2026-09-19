# PANDA prostate cancer grading

Slide-level ISUP grading of prostate biopsies (Kaggle PANDA, 10,615 whole-slide images) with
attention-based multiple-instance learning on pathology foundation-model features.

**Headline numbers** (5-fold cross-validation, stratified by hospital and grade):

| | quadratic weighted kappa |
|---|---|
| Attention MIL (ABMIL) | **0.897 ± 0.004** |
| Mean-pooling baseline | 0.842 ± 0.005 |

The attention model beats the baseline by 0.055, more than ten times the 0.004 spread across
folds. Trained on one hospital and tested on the other, kappa falls from 0.86–0.90 to 0.21–0.31:
the model learns site-specific appearance, not just cancer. Attention separates cancer tiles from
benign ones at 0.88 median AUC (Radboud) without ever seeing a mask during training, and predicted
tumour area tracks the mask at r = 0.94.

See [RESULTS.md](RESULTS.md) for the full tables and figures.

Two stages, because the slides are 411 GB and the features are ~9 GB:

| Stage | Where | What |
|---|---|---|
| 1. Tile + encode | Kaggle (data is mounted there) | Tissue detection, 224px tiles at 20x, Phikon (ViT-B/16) features |
| 2. Train + evaluate | GPU server | ABMIL grading, cross-hospital test, attention localisation, cancer-area quantification |

## Stage 1 — features (Kaggle)

`notebooks/01_extract_features.ipynb`, run with GPU T4 x2, Internet on, the
`prostate-cancer-grade-assessment` competition attached, and a `KAGGLE_KEY` notebook secret.

- Dry run first (`DEBUG_SLIDES = 50`): checks the tiling pictures and estimates the full run.
- Real run (`DEBUG_SLIDES = None`): Save Version -> Save & Run All, about 4.5 hours.
- Output goes to the private Kaggle dataset `kumaarbalbir/panda-phikon-features`, uploaded part by part.
  Rerunning the notebook downloads finished parts and continues, so a crash or the 12-hour
  session limit costs at most one part (~1,000 slides).

Saved per tile: the feature vector, the slide and level-0 position, the tissue fraction, and from
the label masks the cancer, epithelium and Gleason 3/4/5 fractions.

## Stage 2 — training (GPU server)

Get the features (either the Kaggle notebook's output zip, or the dataset) into `data/`:

```bash
kaggle datasets download kumaarbalbir/panda-phikon-features -p data --unzip
./sync.sh                                    # copies the code, not the data, to the GPU box
CUDA_VISIBLE_DEVICES=3 python train.py --features data/pda-ft --out results --quiet
```

Needs `torch numpy pandas pyarrow scikit-learn scipy matplotlib tqdm` in the environment, plus
`openslide-python` and `openslide-bin` only for `--slides-dir` (attention heatmaps over the slide
images). `--device` defaults to CUDA when available; `python train.py --help` lists the training
settings.

Results land in `--out`: `results.json`, per-fold predictions and weights, training curves,
cross-hospital scores, per-slide cancer area, and figures. Reruns skip folds that already finished.

## Layout

    train.py                              stage 2 (the one to run)
    sync.sh                               rsync the code to the GPU box
    notebooks/01_extract_features.ipynb   stage 1, generated from src/
    notebooks/02_train_mil.ipynb          stage 2 as a Kaggle notebook (fallback if no server)
    notebooks/03_heatmaps.ipynb           attention heatmaps, run on Kaggle where the slides are
    src/01_extract_features.py            notebook sources in `# %%` percent format
    src/02_train_mil.py
    src/03_heatmaps_template.py           heatmap notebook without the baked-in attention
    src/make_heatmap_notebook.py          picks example slides and writes src/03_heatmaps.py
    src/build_notebooks.py                rebuilds notebooks/*.ipynb from src/*.py
    figures/, RESULTS.md                  what the run produced
    data/, results/                       not in git

Edit `src/*.py` and run `python src/build_notebooks.py` rather than editing the notebooks.
`train.py` holds the same training logic as notebook 2; keep changes in step.

## Stage 3 — heatmaps (Kaggle)

`notebooks/03_heatmaps.ipynb` draws five example slides with their attention and the mask's cancer
tiles. The attention values are baked into the notebook by `src/make_heatmap_notebook.py`, so it
needs no GPU and no results upload: just the competition images. Regenerate it after a new run with

```bash
python src/make_heatmap_notebook.py && python src/build_notebooks.py
```

## Evaluation

- Quadratic weighted kappa, 5-fold cross-validation stratified by hospital and grade. Each fold
  picks its epoch on 10% of its own training data, so test folds stay untouched, and the spread
  across folds shows how large a difference has to be to mean anything.
- Mean pooling over tiles as a baseline, to show what attention is worth.
- Train on one hospital, test on the other: the cost of moving to a new site.
- Attention is checked against the masks (which the model never sees), and a linear probe on the
  frozen features measures cancer area per slide.
