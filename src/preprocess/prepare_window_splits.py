"""
src/prepare_window_splits.py

Generalized temporal windowing for the HMM experiment. Works for any dataset
registered with a WindowConfig (Fakeddit, Yelp, ...). Dataset-specific behavior
— source merging, row filtering, timestamp parsing, and stratification column —
comes from the dataset's WindowConfig (see datasets/registry.py).

Procedure
---------
1. Load the raw corpus via spec.load_raw(): concatenate sources, apply the
   optional row filter, parse timestamps, sort by time. Produces a "created_dt"
   column regardless of the raw timestamp format.
2. Divide the full date range into fixed-width windows of `window_days`.
3. Find the largest contiguous run of windows each having >= min_samples raw rows.
4. N = min raw count across that run.
5. Subsample each qualifying window to exactly N rows, stratified by the
   dataset's stratify_col (class proportions preserved, not equalized).
6. Save each window as <name>_window_000.tsv, _001.tsv, ... plus a manifest CSV.

The output filename stem and manifest columns are dataset-neutral. Class-count
columns in the manifest are cls_<label> for whatever integer labels the
stratify column takes, so the manifest is self-describing.

Note on stratify labels: subsampling stratifies on the RAW stratify_col values
(e.g. Yelp's 1..5 stars, Fakeddit's 6_way_label 0..5). The manifest records
cls_<raw_value> counts. Downstream class-distribution analysis should read these
columns rather than assuming a fixed 0..5 range.

Usage
-----
    # Fakeddit (60-day windows, min 9k, defaults come from the spec)
    python src.prepare_window_splits --dataset fakeddit \
        --data_dir assets/raw/multimodal_only_samples \
        --output_dir data/splits/hmm_windows/fakeddit

    # Yelp (90-day windows, min 10k)
    python -m src.preprocess.prepare_window_splits --dataset yelp \
        --data_dir data \
        --output_dir data/splits/hmm_windows/yelp

    # Override any spec default from the CLI:
    python src.prepare_window_splits --dataset yelp --data_dir data \
        --window_days 45 --min_samples 20000 --seed 7
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets.registry import get_spec


def stratified_subsample(df, n, label_col, rng):
    """
    Draw exactly `n` rows from `df`, preserving class proportions of `label_col`.
    Floor allocation with largest-remainder tie-breaking so the total is exactly n.
    """
    counts = df[label_col].value_counts()
    proportions = counts / counts.sum()

    alloc = (proportions * n).apply(np.floor).astype(int)
    remainder = n - alloc.sum()

    # Distribute the leftover rows to the classes with the largest fractional
    # parts so the allocation sums to exactly n.
    frac = (proportions * n) - alloc
    top_classes = frac.nlargest(remainder).index
    alloc[top_classes] += 1

    assert alloc.sum() == n, f"Allocation sum {alloc.sum()} != {n}"

    parts = []
    for cls, k in alloc.items():
        pool = df[df[label_col] == cls]
        sampled = pool.sample(n=int(k), replace=False,
                              random_state=rng.integers(2**31))
        parts.append(sampled)

    # Shuffle the concatenated parts so classes are not grouped in output order.
    return (pd.concat(parts)
              .sample(frac=1, random_state=rng.integers(2**31))
              .reset_index(drop=True))


def largest_contiguous_run(mask):
    """
    (start_idx, end_idx) inclusive of the longest contiguous True run in `mask`.
    Ties broken by earliest start.
    """
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0
    for i, val in enumerate(mask):
        if val:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    if best_len == 0:
        raise ValueError("No window meets the minimum sample threshold.")
    return best_start, best_start + best_len - 1


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True,
                        help="Registered dataset name (fakeddit | yelp).")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing the raw source file(s).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--window_days", type=int, default=None,
                        help="Override the dataset's default window width.")
    parser.add_argument("--min_samples", type=int, default=None,
                        help="Override the dataset's default min samples/window.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spec = get_spec(args.dataset)
    if spec.window is None:
        parser.error(f"Dataset '{spec.name}' has no WindowConfig; cannot window.")
    wc = spec.window

    window_days = args.window_days if args.window_days is not None else wc.window_days
    min_samples = args.min_samples if args.min_samples is not None else wc.min_samples
    stratify_col = wc.stratify_col

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 64)
    print(f"  Windowing dataset : {spec.name}")
    print(f"  Window width      : {window_days} days")
    print(f"  Min samples/window: {min_samples:,}")
    print(f"  Stratify on       : {stratify_col}")
    print(f"  Seed              : {args.seed}")
    print("=" * 64)

    # 1. Load raw corpus (spec-driven).
    data = spec.load_raw(args.data_dir)
    print(f"Rows after load/filter: {len(data):,}")
    print(f"Date range: {data['created_dt'].min().date()} → "
          f"{data['created_dt'].max().date()}\n")

    if stratify_col not in data.columns:
        raise KeyError(
            f"Stratify column '{stratify_col}' not in raw data columns: "
            f"{list(data.columns)}"
        )

    # 2. Assign each row to a fixed-width window over the full date range.
    origin = data["created_dt"].min().normalize()   # midnight of first day
    data["window_idx"] = (
        (data["created_dt"] - origin) // pd.Timedelta(days=window_days)
    ).astype(int)

    all_window_idxs = sorted(data["window_idx"].unique())
    print(f"Total windows in date range: {len(all_window_idxs)}\n")

    # 3. Count per window and find the qualifying contiguous run. Build a dense
    # index range so empty windows count as gaps that break the run.
    window_counts = data.groupby("window_idx").size()
    idx_min, idx_max = all_window_idxs[0], all_window_idxs[-1]
    all_idxs = list(range(idx_min, idx_max + 1))
    counts_full = [int(window_counts.get(i, 0)) for i in all_idxs]
    meets_threshold = [c >= min_samples for c in counts_full]

    run_start_local, run_end_local = largest_contiguous_run(meets_threshold)
    qualifying_global_idxs = all_idxs[run_start_local: run_end_local + 1]
    qualifying_counts = counts_full[run_start_local: run_end_local + 1]

    n_windows = len(qualifying_global_idxs)
    n_subsample = min(qualifying_counts)

    run_start_dt = origin + pd.Timedelta(days=qualifying_global_idxs[0] * window_days)
    run_end_dt = origin + pd.Timedelta(days=(qualifying_global_idxs[-1] + 1) * window_days)

    print(f"Qualifying contiguous run: {n_windows} windows")
    print(f"  Global window indices : {qualifying_global_idxs[0]} → {qualifying_global_idxs[-1]}")
    print(f"  Date range            : {run_start_dt.date()} → "
          f"{(run_end_dt - pd.Timedelta(days=1)).date()}")
    print(f"  Raw counts range      : min={min(qualifying_counts):,}, "
          f"max={max(qualifying_counts):,}")
    print(f"  Subsampling N         : {n_subsample:,}  (= min raw count)\n")

    # 4 & 5. Subsample each qualifying window and save it.
    stem = spec.name
    manifest_rows = []

    for local_i, global_idx in enumerate(qualifying_global_idxs):
        window_data = data[data["window_idx"] == global_idx].copy()

        w_start = origin + pd.Timedelta(days=global_idx * window_days)
        w_end = (origin + pd.Timedelta(days=(global_idx + 1) * window_days)
                 - pd.Timedelta(seconds=1))

        raw_n = len(window_data)
        sampled = stratified_subsample(window_data, n_subsample, stratify_col, rng)

        fname = f"{stem}_window_{local_i:03d}.tsv"
        sampled.to_csv(os.path.join(args.output_dir, fname), sep="\t", index=False)

        class_dist = sampled[stratify_col].value_counts().sort_index().to_dict()
        manifest_rows.append({
            "window_local_idx":  local_i,
            "window_global_idx": global_idx,
            "start_date":        w_start.date().isoformat(),
            "end_date":          w_end.date().isoformat(),
            "raw_n":             raw_n,
            "sampled_n":         len(sampled),
            "filename":          fname,
            **{f"cls_{k}": v for k, v in class_dist.items()},
        })

        print(f"  [{local_i:03d}] {w_start.date()} → {w_end.date()}"
              f"  raw={raw_n:,}  sampled={len(sampled):,}  → {fname}")

    # 6. Save the manifest describing all emitted windows.
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(args.output_dir, f"{stem}_windows_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print(f"\nManifest saved → {manifest_path}")
    print(f"Done. {n_windows} windows × {n_subsample:,} samples each → {args.output_dir}")

    # Sanity check: compare the stratify distribution of the first and last window.
    print(f"\n── {stratify_col} distribution (first vs last window) ──")
    for row in (manifest_rows[0], manifest_rows[-1]):
        cls_cols = {k: v for k, v in row.items() if k.startswith("cls_")}
        total = sum(cls_cols.values())
        dist = {k: f"{v/total:.3f}" for k, v in cls_cols.items()}
        print(f"  Window {row['window_local_idx']:03d} ({row['start_date']}): {dist}")


if __name__ == "__main__":
    main()