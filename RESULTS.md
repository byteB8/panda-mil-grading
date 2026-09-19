# Results

One run of `train.py` on 10,615 slides and 5,686,231 tiles: 20 epochs, 5 folds, seed 0, about
50 minutes on one A100. Raw numbers in `results/results.json` (not in git).

## Grading

Quadratic weighted kappa, the metric the PANDA competition used. Folds are stratified by hospital
and grade; each fold's model picks its epoch on 10% of its own training slides, so the fold it is
scored on stays untouched.

| Model | QWK (mean ± sd) | Karolinska | Radboud |
|---|---|---|---|
| **ABMIL** (gated attention) | **0.897 ± 0.004** | 0.917 | 0.857 |
| Mean pooling | 0.842 ± 0.005 | 0.814 | 0.819 |

![grading accuracy and cross-hospital transfer](figures/results_summary.png)

Per fold, ABMIL scores 0.891–0.901 and mean pooling 0.837–0.850, so the 0.055 gap never comes
close to overlapping. Attention is doing real work: a biopsy is mostly benign tissue, and
averaging over every tile dilutes the few that carry the grade.

Both models are better on Karolinska slides than Radboud ones, which matches Karolinska being the
larger and more class-balanced half of the dataset.

![confusion matrices](figures/confusion_matrices.png)

Mistakes sit next to the diagonal: the model confuses neighbouring grades, which is what the
quadratic weighting forgives and what pathologists disagree about too.

## Cross-hospital robustness

Train on one hospital, hold out 20% of it as an in-hospital test set, then test on the other
hospital as well.

| Trained on | Same hospital | Other hospital |
|---|---|---|
| Radboud | 0.860 | **0.205** |
| Karolinska | 0.899 | **0.305** |

Roughly three quarters of the performance disappears at a new site. Since both halves are the same
disease and the same grading scale, what fails to transfer is appearance: different scanners,
stains and grading habits. This is the case for stain normalisation and colour augmentation, and
it is the number to quote when someone asks what a deployed model would do at their lab.

## What the model looks at

The masks are never used in training, so comparing attention to them is a genuine check.

- Attention separates cancer tiles from cancer-free ones at **0.88** median AUC (Radboud) and
  **0.77** (Karolinska), over 7,455 out-of-fold slides that have a mask and some cancer.
- A linear probe on the frozen tile features detects cancer tiles at **AUC 0.94**.
- The share of tiles it calls cancer tracks the mask's share at **r = 0.94** per slide, so the
  pipeline also quantifies tumour area rather than only grading.

![cancer area](figures/cancer_area.png)

![attention heatmaps](figures/attention_heatmaps.png)

Five slides from `notebooks/03_heatmaps.ipynb`: two where attention lands squarely on the mask's
cancer (one per hospital), a typical slide at the median localisation AUC, a failure where
attention avoids the cancer (ISUP 3 predicted as 1, AUC 0.03), and a benign slide for contrast.
Attention is rank-scaled per slide, since the raw weights sum to one and shrink as a slide has
more tiles.

## Honest limits

- Cross-validation on the public training set, not the competition's private test set, so these
  numbers are not directly comparable to the leaderboard (winners were around 0.93–0.94 there).
- PANDA's labels are known to be noisy, especially the Karolinska half, which caps what any model
  can score.
- Tiles are capped at 2,048 per slide and sampled to 512 per training step; a handful of large
  slides are therefore only partly seen during any one step.
- The cancer-tile definition treats a tile as cancer when at least half its labelled epithelium is
  cancer. Radboud masks label glands and stroma separately, Karolinska's do not, so the two
  hospitals' tile labels are not exactly the same thing.
- One slide of 10,616 was dropped: no tile passed the tissue threshold.
