"""
make_grid_splits.py

Build a train/validation pair for hyperparameter grid search, sized to match the
temporal-window regime so the tuned hyperparameters transfer to the per-window
models.

What it does
------------
1. Read the window manifest produced by prepare_window_splits.py to find N =
   the number of samples in the smallest window (all windows are subsampled to
   this common size, so every window has exactly N rows; N is read from the
   manifest's sampled_n).
2. Pool the same raw corpus the windows were drawn from (via spec.load_raw),
   restricted to the same contiguous date span the windows cover (so the grid
   search sees the same distribution as the window experiment, not data from
   outside the qualifying run).
3. Draw a training set of N rows and a validation set of N rows, disjoint, both
   stratified on the dataset's stratify_col so class proportions are conserved.
4. Write them as TSV to data/splits/grid_search/:
       <dataset>_grid_train.tsv
       <dataset>_grid_val.tsv
   plus a small JSON describing sizes / proportions / provenance.

Why sized to the window, not the whole corpus
---------------------------------------------
The per-window models each train on N samples. Tuning lr / hidden size on far
more (or far fewer) data would pick hyperparameters that don't match the regime
the windows actually run in. Matching N keeps the optimisation problem
comparable.

Requires 2N rows available in the pooled span. If the span is too small, the
script reports how many rows are available and exits.

Usage
-----
    python -m src.training.make_grid_splits --dataset yelp \
        --data_dir data \
        --manifest data/splits/hmm_windows/yelp/yelp_windows_manifest.csv

    python -m src.training.make_grid_splits --dataset fakeddit \
        --data_dir assets/raw/multimodal_only_samples \
        --manifest data/splits/hmm_windows/fakeddit/fakeddit_windows_manifest.csv
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets.registry import get_spec

# Reuse the exact stratified sampler used for windowing so proportions are
# handled identically (floor allocation + largest-remainder).
from src.preprocess.prepare_window_splits import stratified_subsample


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, help="fakeddit | yelp")
    p.add_argument("--data_dir", required=True,
                   help="Directory with the raw source file(s) (same as windowing).")
    p.add_argument("--manifest", required=True,
                   help="Window manifest CSV (to read N = smallest window size "
                        "and the date span the windows cover).")
    p.add_argument("--output_dir", default="data/splits/grid_search")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n_override", type=int, default=None,
                   help="Override N (per-set size). Default: min sampled_n from "
                        "the manifest.")
    return p.parse_args()


def main():
    """Build and write the stratified grid-search train/val splits."""
    args = parse_args()
    spec = get_spec(args.dataset)
    if spec.window is None:
        sys.exit(f"Dataset '{spec.name}' has no WindowConfig.")
    stratify_col = spec.window.stratify_col
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Read N and the covered date span from the manifest.
    manifest = pd.read_csv(args.manifest, parse_dates=["start_date", "end_date"])
    if manifest.empty:
        sys.exit(f"Manifest {args.manifest} is empty.")

    N = args.n_override if args.n_override is not None else int(manifest["sampled_n"].min())
    span_start = manifest["start_date"].min()
    span_end = manifest["end_date"].max()

    print("=" * 64)
    print(f"  Grid-search splits for dataset: {spec.name}")
    print(f"  N (per set)      : {N:,}   (smallest window size)")
    print(f"  Window date span : {span_start.date()} → {span_end.date()}")
    print(f"  Stratify on      : {stratify_col}")
    print(f"  Seed             : {args.seed}")
    print("=" * 64)

    # Pool the raw corpus over the same span the windows cover.
    data = spec.load_raw(args.data_dir)
    # created_dt is tz-aware UTC; make the span bounds tz-aware to match.
    lo = pd.Timestamp(span_start).tz_localize("UTC")
    hi = pd.Timestamp(span_end).tz_localize("UTC") + pd.Timedelta(days=1)
    pool = data[(data["created_dt"] >= lo) & (data["created_dt"] < hi)].copy()
    print(f"Rows available in span: {len(pool):,}  (need 2N = {2*N:,})")

    if len(pool) < 2 * N:
        sys.exit(
            f"\nNot enough rows: span has {len(pool):,} but need 2N = {2*N:,}.\n"
            f"Options: lower N with --n_override, or widen the manifest span."
        )

    if stratify_col not in pool.columns:
        sys.exit(f"Stratify column '{stratify_col}' not in data columns.")

    # Draw disjoint train / val, each N rows, with proportions conserved.
    # First draw a stratified pool of 2N rows, then split that pool into two
    # stratified halves. This guarantees (a) both sets are size N, (b) they are
    # disjoint, and (c) both preserve the pool's class proportions.
    combined = stratified_subsample(pool, 2 * N, stratify_col, rng)

    # Split the combined 2N into two disjoint N-row halves, stratified.
    train_parts, val_parts = [], []
    for cls, grp in combined.groupby(stratify_col, sort=True):
        grp = grp.sample(frac=1, random_state=rng.integers(2**31)).reset_index(drop=True)
        half = len(grp) // 2
        train_parts.append(grp.iloc[:half])
        val_parts.append(grp.iloc[half:2 * half])   # Drop 1 if odd, keeps sizes equal.

    train_df = (pd.concat(train_parts)
                  .sample(frac=1, random_state=rng.integers(2**31))
                  .reset_index(drop=True))
    val_df = (pd.concat(val_parts)
                .sample(frac=1, random_state=rng.integers(2**31))
                .reset_index(drop=True))

    def top_up(df, target, exclude_ids):
        """Top up df to target rows from the unused pool remainder, stratified."""
        # Odd class counts can leave a set a few rows under N; refill from rows
        # not already used so the sets stay disjoint.
        deficit = target - len(df)
        if deficit <= 0:
            return df.iloc[:target].reset_index(drop=True)
        spare = pool[~pool[spec.id_col].isin(exclude_ids)]
        extra = stratified_subsample(spare, deficit, stratify_col, rng)
        return pd.concat([df, extra]).reset_index(drop=True)

    used_ids = set(train_df[spec.id_col]) | set(val_df[spec.id_col])
    train_df = top_up(train_df, N, used_ids)
    used_ids |= set(train_df[spec.id_col])
    val_df = top_up(val_df, N, used_ids)

    # Verify the sets are disjoint and exactly N rows each.
    overlap = set(train_df[spec.id_col]) & set(val_df[spec.id_col])
    assert not overlap, f"Train/val overlap by {len(overlap)} ids!"
    assert len(train_df) == N, (len(train_df), N)
    assert len(val_df) == N, (len(val_df), N)

    # Write the splits and a provenance JSON.
    train_path = os.path.join(args.output_dir, f"{spec.name}_grid_train.tsv")
    val_path = os.path.join(args.output_dir, f"{spec.name}_grid_val.tsv")
    train_df.to_csv(train_path, sep="\t", index=False)
    val_df.to_csv(val_path, sep="\t", index=False)

    def props(df):
        """Return normalized class proportions as a rounded {label: frac} dict."""
        c = df[stratify_col].value_counts(normalize=True).sort_index()
        return {str(k): round(float(v), 4) for k, v in c.items()}

    info = {
        "dataset": spec.name,
        "N_per_set": N,
        "stratify_col": stratify_col,
        "span_start": str(span_start.date()),
        "span_end": str(span_end.date()),
        "pool_size": int(len(pool)),
        "seed": args.seed,
        "train_file": os.path.basename(train_path),
        "val_file": os.path.basename(val_path),
        "train_props": props(train_df),
        "val_props": props(val_df),
        "pool_props": props(pool),
    }
    info_path = os.path.join(args.output_dir, f"{spec.name}_grid_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nTrain → {train_path}  ({len(train_df):,} rows)")
    print(f"Val   → {val_path}  ({len(val_df):,} rows)")
    print(f"Info  → {info_path}")
    print(f"\nClass proportions (train / val / pool):")
    tp, vp, pp = props(train_df), props(val_df), props(pool)
    for k in sorted(pp):
        print(f"  {stratify_col}={k}: "
              f"{tp.get(k, 0):.4f} / {vp.get(k, 0):.4f} / {pp[k]:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()