"""
merge_cross_window.py

Assemble the per-row .npz files written by cross_window_eval.py into the full
cross-window F1 matrix, generalized across datasets.

HMM-agnostic: no decode file is read here. State labels are joined later in
within_across_states.py.

Usage
-----
    # Fakeddit:
    python -m src.eval.merge_cross_window --dataset fakeddit --num_classes 6

    # Yelp, explicit dirs:
    python -m src.eval.merge_cross_window --dataset yelp --num_classes 5 \
        --rows_dir   data/hmm_perf/yelp/rows \
        --output_dir data/hmm_perf/yelp

Requirements: numpy, pandas, matplotlib
"""

import os
import sys
import glob
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.eval.cross_window_eval import plot_heatmap, elapsed
from src.datasets.registry import get_spec


def _save_and_plot(
    f1_matrix: np.ndarray,
    f1_per_class: np.ndarray,
    f1_per_seed_cube: np.ndarray,
    valid_ids: list[int],
    args,           # needs: output_dir, num_classes, dataset
    t0: float,
) -> None:
    """
    Persist the assembled F1 matrix and generate plots.

    HMM-agnostic: no state_seq / window_ids / k parameters. HMM state labels are
    joined downstream in within_across_states.py.
    """
    import pandas as pd
    os.makedirs(args.output_dir, exist_ok=True)

    # NPZ.
    npz_path = os.path.join(args.output_dir, "cross_window_f1.npz")
    np.savez(
        npz_path,
        f1_matrix        = f1_matrix,
        f1_per_class     = f1_per_class,
        f1_per_seed_cube = f1_per_seed_cube,
        valid_ids        = np.array(valid_ids, dtype=np.int32),
        num_classes      = np.int32(args.num_classes),
    )
    print(f"  Saved: {npz_path}")

    # CSV.
    csv_path = os.path.join(args.output_dir, "cross_window_f1.csv")
    pd.DataFrame(
        f1_matrix,
        index   = [f"train_{i:03d}" for i in valid_ids],
        columns = [f"test_{j:03d}"  for j in valid_ids],
    ).to_csv(csv_path)
    print(f"  Saved: {csv_path}")

    # Column-centered matrix. Control for the inherent difficulty of each
    # testing window: subtract each test column's mean F1 (averaged over train
    # windows) so that intrinsically easy/hard test windows don't dominate the
    # picture. A positive entry then means "this train window does better than
    # the average train window does on this particular test window."
    # nanmean over axis=0 ignores missing (NaN) rows.
    col_means = np.nanmean(f1_matrix, axis=0, keepdims=True)   # (1, n_valid)
    f1_matrix_colcentered = f1_matrix - col_means

    npz_cc_path = os.path.join(args.output_dir, "cross_window_f1_colcentered.npz")
    np.savez(
        npz_cc_path,
        f1_matrix_colcentered = f1_matrix_colcentered,
        col_means             = col_means.ravel(),
        valid_ids             = np.array(valid_ids, dtype=np.int32),
        num_classes           = np.int32(args.num_classes),
    )
    print(f"  Saved: {npz_cc_path}")

    csv_cc_path = os.path.join(args.output_dir, "cross_window_f1_colcentered.csv")
    pd.DataFrame(
        f1_matrix_colcentered,
        index   = [f"train_{i:03d}" for i in valid_ids],
        columns = [f"test_{j:03d}"  for j in valid_ids],
    ).to_csv(csv_cc_path)
    print(f"  Saved: {csv_cc_path}")

    # Heatmaps.
    plot_heatmap(f1_matrix, valid_ids,
                 os.path.join(args.output_dir, "heatmap_f1.png"),
                 dataset=getattr(args, "dataset", None))
    plot_heatmap(f1_matrix_colcentered, valid_ids,
                 os.path.join(args.output_dir, "heatmap_f1_colcentered.png"),
                 dataset=getattr(args, "dataset", None))

    print(f"  Total wall time: {elapsed(t0)}")


def main(args: argparse.Namespace) -> None:
    """Merge all row files into the cross-window F1 matrix and write outputs."""
    t0 = time.time()
    spec = get_spec(args.dataset)
    stem = spec.name

    print("=" * 60)
    print(f"  merge_cross_window.py  —  dataset={stem}")
    print("=" * 60)

    row_files = sorted(glob.glob(os.path.join(args.rows_dir, "row_*.npz")))
    if not row_files:
        print(f"  ERROR: no row_*.npz found in {args.rows_dir}. "
              f"Did the array job finish?")
        sys.exit(1)

    # Peek at one file to learn the shape.
    probe = np.load(row_files[0])
    valid_ids = probe["valid_ids"].tolist()
    n_valid = len(valid_ids)

    # Infer per-class width from the row files themselves, so this stays correct
    # for any class count.
    n_way = int(probe["row_per_class"].shape[1])
    if n_way != args.num_classes:
        print(f"  [warn] row files carry {n_way}-way per-class F1 but "
              f"--num_classes={args.num_classes}; using {n_way} from the data.")

    # Infer max seed count from the first row file that carries per-seed F1.
    n_seeds_max = 10
    for fpath in row_files:
        d = np.load(fpath)
        if "row_f1_per_seed" in d:
            n_seeds_max = d["row_f1_per_seed"].shape[0]
            break
    print(f"  n_valid={n_valid}, n_way={n_way}, n_seeds_max={n_seeds_max}")

    # Assemble matrices.
    f1_matrix = np.full((n_valid, n_valid), np.nan)
    f1_per_class = np.full((n_valid, n_valid, n_way), np.nan)
    f1_per_seed_cube = np.full((n_valid, n_seeds_max, n_valid), np.nan)
    rows_filled: set[int] = set()

    for fpath in row_files:
        d = np.load(fpath)
        idx = int(d["row_idx"])

        if idx >= n_valid:
            print(f"  [warn] row_idx={idx} >= n_valid={n_valid} — skipping")
            continue
        if list(d["valid_ids"]) != valid_ids:
            print(f"  [warn] {fpath}: valid_ids mismatch — skipping")
            continue

        f1_matrix[idx] = d["row_f1"]
        f1_per_class[idx] = d["row_per_class"]

        if "row_f1_per_seed" in d:
            fps = d["row_f1_per_seed"]          # (n_seeds, n_valid)
            # Guard against a row carrying more seeds than the cube was sized for.
            s = min(fps.shape[0], n_seeds_max)
            f1_per_seed_cube[idx, :s, :] = fps[:s]

        rows_filled.add(idx)

    missing = sorted(set(range(n_valid)) - rows_filled)
    if missing:
        print(f"\n  [warn] {len(missing)} rows missing: {missing}")
        print(f"  These will appear as NaN in the matrix.")
        print(f"  To rerun missing tasks only:")
        print(f"    sbatch --array={','.join(str(m) for m in missing)} "
              f"cross_window_eval.slurm")
    else:
        print(f"\n  All {n_valid} rows present — matrix is complete.")

    # Save + plot.
    class _Args:
        pass
    save_args = _Args()
    save_args.output_dir = args.output_dir
    save_args.num_classes = n_way
    save_args.dataset = stem

    _save_and_plot(f1_matrix, f1_per_class, f1_per_seed_cube,
                   valid_ids, save_args, t0)

    perf_dir = args.output_dir
    hmm_dir = f"data/hmm_hmm/{stem}"
    wa_dir = f"data/hmm_within_across/{stem}"
    decode_npz = os.path.join(hmm_dir, f"{stem}_decode_k<K>.npz")
    print(
        "\n  Next steps:\n"
        "    1. (if not done) Run the HMM pipeline for this dataset:\n"
        f"         python -m src.hmm.extract_weights_pca --dataset {stem}\n"
        f"         python -m src.hmm.select_hmm_states  --dataset {stem} --n_pcs <P>\n"
        f"         python -m src.hmm.decode_hmm         --dataset {stem} --k <K> --n_pcs <P>\n"
        "    2. Join the F1 matrix with the HMM states:\n"
        "         python -m src.eval.within_across_states \\\n"
        f"             --f1_npz     {os.path.join(perf_dir, 'cross_window_f1.npz')} \\\n"
        f"             --decode_npz {decode_npz} \\\n"
        f"             --output_dir {os.path.join(wa_dir, 'k<K>')}"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and fill in per-dataset default paths."""
    p = argparse.ArgumentParser(
        description=(
            "Merge per-row .npz files into the full cross-window F1 matrix, "
            "dataset-generalized. HMM-agnostic: no decode_npz required. "
            "State labels are joined later in within_across_states.py."
        )
    )
    p.add_argument("--dataset", required=True, choices=["fakeddit", "yelp"],
                   help="Which dataset spec to use (drives default paths + stem).")
    p.add_argument("--num_classes", type=int, required=True,
                   help="Class count (6 fakeddit, 5 yelp). The per-class width "
                        "is read from the row files; this is only a sanity check.")
    p.add_argument("--rows_dir", default=None,
                   help="Directory containing row_NNN.npz files from the array "
                        "job. Defaults to data/hmm_perf/<dataset>/rows.")
    p.add_argument("--output_dir", default=None,
                   help="Where to write cross_window_f1.npz/.csv and the heatmap. "
                        "Defaults to data/hmm_perf/<dataset>.")
    args = p.parse_args()

    ds = args.dataset
    if args.output_dir is None:
        args.output_dir = f"data/hmm_perf/{ds}"
    if args.rows_dir is None:
        args.rows_dir = os.path.join(args.output_dir, "rows")
    return args


if __name__ == "__main__":
    main(parse_args())