"""Can the hospital-to-hospital gap be closed without touching the images?

Training on one hospital and testing on the other costs about two thirds of the kappa. That gap
could come from the slides (stain, scanner) or from the features Phikon produces for them. This
script tries three fixes that need no new feature extraction, and scores each one both at home and
at the other hospital:

  none        the baseline from train.py: features standardised over the whole dataset
  per-centre  each hospital's features standardised with its own mean and sd
  coral       source features whitened and recoloured to the target's covariance (Sun & Saenko)
  dann        a hospital classifier behind a gradient-reversal layer (Ganin et al.)

All three use only unlabelled target features, never target grades.

    CUDA_VISIBLE_DEVICES=3 python adapt.py --features data/pda-ft --out results-adapt --quiet
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import train as base

CENTRES = ["radboud", "karolinska"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results-adapt"))
    p.add_argument("--methods", nargs="+", default=["none", "per-centre", "coral", "dann"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-train-tiles", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--domain-weight", type=float, default=0.3, help="dann: weight on the hospital loss")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)  # unused here; kept so configs compare cleanly
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Feature-level fixes


def centre_rows(tiles, slides, centre):
    """Row indices of every tile belonging to one hospital."""
    keep = slides[slides.data_provider == centre]
    return np.concatenate([np.arange(o, o + n) for o, n in zip(keep.offset, keep.n_tiles)])


CHUNK = 50_000  # rows per pass; keeps the temporaries small enough for a laptop too


def moments(features, rows, device, chunk=CHUNK):
    """Mean and covariance of a set of feature rows: float32 blocks, float64 accumulators."""
    dim = features.shape[1]
    total = torch.zeros(dim, dtype=torch.float64, device=device)
    gram = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    for start in range(0, len(rows), chunk):
        block = features[torch.from_numpy(rows[start:start + chunk])].to(device).float()
        total += block.sum(0).double()
        gram += (block.T @ block).double()
        del block
    n = len(rows)
    mean = total / n
    cov = gram / n - torch.outer(mean, mean)
    return mean, cov


def matrix_power(cov, power, eps=1e-3):
    """cov ** power for a symmetric positive-definite matrix, with a ridge for stability."""
    cov = cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device) * cov.diagonal().mean()
    values, vectors = torch.linalg.eigh(cov)
    values = values.clamp_min(1e-8)
    return (vectors * values.pow(power)) @ vectors.T


def standardise_per_centre(features, tiles, slides, device):
    """Each hospital's features centred and scaled by its own statistics."""
    out = features.clone()
    for centre in CENTRES:
        rows = centre_rows(tiles, slides, centre)
        mean, cov = moments(out, rows, device)
        sd = cov.diagonal().clamp_min(1e-12).sqrt().float()
        mean = mean.float()
        for start in range(0, len(rows), CHUNK):
            index = torch.from_numpy(rows[start:start + CHUNK])
            block = out[index].to(device).float()
            out[index] = ((block - mean) / sd).half().cpu()
            del block
    return out


def coral(features, tiles, slides, source, target, device):
    """Recolour the source hospital's features to the target hospital's covariance."""
    out = features.clone()
    source_rows = centre_rows(tiles, slides, source)
    target_rows = centre_rows(tiles, slides, target)
    mean_s, cov_s = moments(out, source_rows, device)
    mean_t, cov_t = moments(out, target_rows, device)
    transform = (matrix_power(cov_s, -0.5) @ matrix_power(cov_t, 0.5)).float()
    mean_s, mean_t = mean_s.float(), mean_t.float()
    for start in range(0, len(source_rows), CHUNK):
        index = torch.from_numpy(source_rows[start:start + CHUNK])
        block = out[index].to(device).float()
        out[index] = (((block - mean_s) @ transform) + mean_t).half().cpu()
        del block
    return out


# --------------------------------------------------------------------------------------
# Adversarial training


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.weight * grad, None


class DANN(nn.Module):
    """ABMIL with a hospital classifier that the encoder is trained to defeat."""

    def __init__(self, in_dim, hidden=256, dropout=0.25):
        super().__init__()
        self.body = base.ABMIL(in_dim, hidden=hidden, dropout=dropout)
        self.domain = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def embed(self, x, mask):
        h = self.body.embed(x)
        scores = self.body.attn_out(
            torch.tanh(self.body.attn_tanh(h)) * torch.sigmoid(self.body.attn_gate(h))).squeeze(-1)
        attention = scores.masked_fill(~mask, float("-inf")).softmax(dim=1)
        return (attention.unsqueeze(-1) * h).sum(dim=1), attention

    def forward(self, x, mask):
        slide_vector, attention = self.embed(x, mask)
        return self.body.head(slide_vector), attention

    def domain_logits(self, slide_vector, weight):
        return self.domain(GradientReversal.apply(slide_vector, weight)).squeeze(-1)


def fit_dann(trainer, train_frame, val_frame, target_frame):
    """Trains ABMIL on the source hospital while making the slide vector hospital-agnostic."""
    args = trainer.args
    base.seed_everything(args.seed)
    model = DANN(trainer.dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = int(np.ceil(len(train_frame) / args.batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    best_qwk, best_state, history = -1.0, None, []
    for epoch in tqdm(range(args.epochs), desc="DANN", disable=args.quiet):
        model.train()
        # The adversary is eased in: early on the encoder has nothing worth aligning.
        weight = args.domain_weight * (2 / (1 + np.exp(-5 * epoch / max(args.epochs - 1, 1))) - 1)
        target_batches = trainer.bag_batches(target_frame, shuffle=True, max_tiles=args.max_train_tiles)
        task_losses, domain_losses = [], []
        for x, mask, grades, _ in trainer.bag_batches(train_frame, shuffle=True, max_tiles=args.max_train_tiles):
            try:
                tx, tmask, _, _ = next(target_batches)
            except StopIteration:  # the target hospital has fewer slides; start over
                target_batches = trainer.bag_batches(target_frame, shuffle=True, max_tiles=args.max_train_tiles)
                tx, tmask, _, _ = next(target_batches)

            source_vector, _ = model.embed(x, mask)
            target_vector, _ = model.embed(tx, tmask)
            targets = (grades.unsqueeze(1) > torch.arange(base.N_GRADES - 1, device=args.device)).float()
            task_loss = F.binary_cross_entropy_with_logits(model.body.head(source_vector), targets)

            vectors = torch.cat([source_vector, target_vector])
            labels = torch.cat([torch.zeros(len(source_vector)), torch.ones(len(target_vector))]).to(args.device)
            domain_loss = F.binary_cross_entropy_with_logits(model.domain_logits(vectors, weight), labels)

            optimizer.zero_grad(set_to_none=True)
            (task_loss + domain_loss).backward()
            optimizer.step()
            scheduler.step()
            task_losses.append(task_loss.item())
            domain_losses.append(domain_loss.item())

        val_qwk = base.qwk(val_frame.isup_grade, trainer.predict(model, val_frame).pred)
        history.append({"epoch": epoch, "task_loss": float(np.mean(task_losses)),
                        "domain_loss": float(np.mean(domain_losses)), "domain_weight": float(weight),
                        "val_qwk": val_qwk})
        if val_qwk >= best_qwk:
            best_qwk, best_state = val_qwk, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


# --------------------------------------------------------------------------------------


def run(method, source, target, features, tiles, slides, args):
    """Trains on `source` with one adaptation method and scores it at home and at `target`."""
    if method == "per-centre":
        adapted = standardise_per_centre(features, tiles, slides, args.device)
    elif method == "coral":
        adapted = coral(features, tiles, slides, source, target, args.device)
    else:
        adapted = features

    trainer = base.Trainer(adapted, slides, args)
    if method in ("per-centre", "coral"):  # already on a common scale; do not undo it
        trainer.mean = torch.zeros_like(trainer.mean)
        trainer.std = torch.ones_like(trainer.std)

    source_frame = slides[slides.data_provider == source]
    rest, in_centre = train_test_split(source_frame, test_size=0.2, stratify=source_frame.isup_grade,
                                       random_state=args.seed)
    train_frame, val_frame = trainer.inner_split(rest)
    train_frame = train_frame.reset_index(drop=True)
    val_frame = val_frame.reset_index(drop=True)
    target_frame = slides[slides.data_provider == target].reset_index(drop=True)

    if method == "dann":
        model, history = fit_dann(trainer, train_frame, val_frame, target_frame)
    else:
        model, history = trainer.fit(base.ABMIL, train_frame, val_frame)

    rows = []
    for split, frame in [("in-centre", in_centre), ("other centre", target_frame)]:
        preds = trainer.predict(model, frame.reset_index(drop=True))
        preds.to_csv(args.out / f"preds_{method}_{source}_{split.replace(' ', '_')}.csv", index=False)
        rows.append({"method": method, "train": source, "test": split, "slides": len(frame),
                     "qwk": base.qwk(preds.isup_grade, preds.pred)})
        print(rows[-1])
    history.to_csv(args.out / f"history_{method}_{source}.csv", index=False)
    torch.save(model.state_dict(), args.out / f"model_{method}_{source}.pt")
    del adapted
    return rows


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device: {args.device}")

    features, tiles, slides, config = base.load_features(args.features)
    results_path = args.out / "adaptation.csv"
    done = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()

    rows = done.to_dict("records")
    for method in args.methods:
        for source, target in [(CENTRES[0], CENTRES[1]), (CENTRES[1], CENTRES[0])]:
            if len(done) and ((done.method == method) & (done.train == source)).any():
                print(f"{method} / {source}: already done")
                continue
            rows += run(method, source, target, features, tiles, slides, args)
            pd.DataFrame(rows).to_csv(results_path, index=False)  # checkpoint after each fit

    table = pd.DataFrame(rows)
    wide = table.pivot_table(index=["method", "train"], columns="test", values="qwk").round(4)
    print("\n", wide)
    (args.out / "adaptation.json").write_text(json.dumps({
        "rows": rows,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "features_config": config,
    }, indent=2, default=float))
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
