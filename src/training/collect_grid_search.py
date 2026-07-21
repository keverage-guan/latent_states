"""
collect_grid_search.py

Scan the grid-search run directory, gather each cell's best-validation metrics,
build a results table, and report the optimal (lr, hidden_size).

Reads every <output_dir>/*/metrics.json written by train_grid.py. Selects the
best cell by a chosen validation metric (default: val macro-F1, the metric the
paper cares about for the imbalanced label distributions; val accuracy is also
available).

Outputs
-------
  <output_dir>/grid_results.csv    one row per cell, sorted best-first
  <output_dir>/grid_best.json      the winning config + its metrics
  a printed table + a printed lr x hidden pivot of the selection metric

Usage
-----
    python src/training/collect_grid_search.py --output_dir runs/grid_search/yelp
    python src/training/collect_grid_search.py --output_dir runs/grid_search/fakeddit \
        --metric val_acc
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import pandas as pd


METRIC_KEYS = {
    "val_macro_f1": "best_val_macro_f1",
    "val_micro_f1": "best_val_micro_f1",
    "val_acc": "best_val_acc",
}


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", required=True,
                   help="Grid-search run dir containing <cell>/metrics.json.")
    p.add_argument("--metric", default="val_macro_f1",
                   choices=list(METRIC_KEYS),
                   help="Selection metric (default: val_macro_f1).")
    p.add_argument("--csv_name", default="grid_results.csv")
    p.add_argument("--best_name", default="grid_best.json")
    return p.parse_args()


def load_cells(output_dir):
    """Read every cell's metrics.json under output_dir into a DataFrame."""
    rows = []
    for mpath in sorted(glob.glob(os.path.join(output_dir, "*", "metrics.json"))):
        try:
            with open(mpath) as f:
                m = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # A cell may still be running or have written a truncated file;
            # skip it rather than aborting the whole collection.
            print(f"  [warn] could not read {mpath}: {e}")
            continue
        rows.append({
            "run_name": m.get("run_name", os.path.basename(os.path.dirname(mpath))),
            "dataset": m.get("dataset"),
            "lr": m.get("lr"),
            "hidden_size": m.get("hidden_size"),
            "num_classes": m.get("num_classes"),
            "best_epoch": m.get("best_epoch"),
            "epochs_run": m.get("epochs_run"),
            "best_val_acc": m.get("best_val_acc"),
            "best_val_micro_f1": m.get("best_val_micro_f1"),
            "best_val_macro_f1": m.get("best_val_macro_f1"),
            "best_val_loss": m.get("best_val_loss"),
            "path": mpath,
        })
    return pd.DataFrame(rows)


def main():
    """Collect grid-search results, pick the winner, and print reports."""
    args = parse_args()
    if not os.path.isdir(args.output_dir):
        sys.exit(f"No such directory: {args.output_dir}")

    df = load_cells(args.output_dir)
    if df.empty:
        sys.exit(f"No metrics.json files found under {args.output_dir}. "
                 f"Did the grid-search array run?")

    sel_col = METRIC_KEYS[args.metric]
    df = df.sort_values(sel_col, ascending=False).reset_index(drop=True)

    # Save the full table, dropping the internal path column.
    csv_path = os.path.join(args.output_dir, args.csv_name)
    df.drop(columns="path").to_csv(csv_path, index=False)

    # The winner is the first row after sorting by the selection metric.
    best = df.iloc[0]
    best_info = {
        "metric": args.metric,
        "lr": float(best["lr"]),
        "hidden_size": int(best["hidden_size"]),
        "num_classes": int(best["num_classes"]) if pd.notna(best["num_classes"]) else None,
        "best_val_acc": float(best["best_val_acc"]),
        "best_val_micro_f1": float(best["best_val_micro_f1"]),
        "best_val_macro_f1": float(best["best_val_macro_f1"]),
        "best_epoch": int(best["best_epoch"]) if pd.notna(best["best_epoch"]) else None,
        "run_name": best["run_name"],
        "n_cells": int(len(df)),
    }
    best_path = os.path.join(args.output_dir, args.best_name)
    with open(best_path, "w") as f:
        json.dump(best_info, f, indent=2)

    # Print the per-cell table.
    show = df[["lr", "hidden_size", "best_val_acc",
               "best_val_macro_f1", "best_val_micro_f1",
               "best_epoch", "epochs_run"]].copy()
    pd.set_option("display.width", 120)
    pd.set_option("display.max_rows", 200)
    print(f"\nGrid-search results (sorted by {args.metric}, best first):\n")
    print(show.to_string(index=False))

    # Print an lr x hidden_size pivot of the selection metric.
    try:
        pivot = df.pivot_table(index="lr", columns="hidden_size",
                               values=sel_col, aggfunc="first")
        pivot = pivot.sort_index(ascending=False)  # High lr at top.
        print(f"\n{args.metric} by lr (rows) × hidden_size (cols):\n")
        print(pivot.round(4).to_string())
    except Exception as e:
        print(f"[warn] could not build pivot: {e}")

    # Report the optimal configuration.
    print("\n" + "=" * 60)
    print(f"  OPTIMAL: lr={best_info['lr']}  hidden_size={best_info['hidden_size']}")
    print(f"  {args.metric} = {best[sel_col]:.4f}  "
          f"(acc={best_info['best_val_acc']:.4f}, "
          f"macro_f1={best_info['best_val_macro_f1']:.4f}) "
          f"@ epoch {best_info['best_epoch']}")
    print("=" * 60)
    print(f"\nFull table → {csv_path}")
    print(f"Winner     → {best_path}")

    # Warn if the winner sits on a grid edge, since that suggests the true
    # optimum may lie outside the searched range.
    lrs = sorted(df["lr"].unique())
    hids = sorted(df["hidden_size"].unique())
    edge = []
    if best_info["lr"] in (lrs[0], lrs[-1]) and len(lrs) > 1:
        edge.append("lr")
    if best_info["hidden_size"] in (hids[0], hids[-1]) and len(hids) > 1:
        edge.append("hidden_size")
    if edge:
        print(f"\n[note] Best config is on the grid edge for: {', '.join(edge)}. "
              f"Consider extending the grid in that direction.")


if __name__ == "__main__":
    main()