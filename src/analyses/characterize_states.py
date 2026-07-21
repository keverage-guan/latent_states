"""
characterize_states.py

Rich per-state characterisation of the HMM decode, generalized across datasets.

For each state (and overall):
  - Date range of member windows, window count, total samples
  - Class distribution (counts + %) over the manifest's cls_* columns
  - Optional top-N category breakdown from raw window TSVs (only for datasets
    whose splits carry a category column, e.g. Fakeddit's "subreddit")
  - Per-state HMM emission mean in z-scaled PCA space (the space the HMM is fit
    and decoded in), plus pairwise state-centroid distances
  - Per-window centroid matrix C (summary), for reference
  - Cross-window generalisation: mean within-state vs across-state F1 per state

Class columns are discovered from the manifest's cls_* columns. Human-readable
class names are taken from --class_names if given, else the cls_* column names
are used verbatim. The category breakdown is gated on a --category_col that must
exist in the split TSVs. Emission means are reported in z-scaled PCA space only,
since the HMM is fit on the already z-scored centroid sequence.

Note on `C`: weights_pca.npz stores C as the per-window centroid matrix
(n_valid_windows x n_pca), not the PCA loadings.

Outputs  (--output_dir)
-----------------------
  state_report.txt       human-readable terminal-width report
  state_summary.csv      one row per state
  state_class_dist.csv   class % per state (wide)
  state_categories.csv   top-N categories per state (only if --category_col found)
  window_centroids.csv   C matrix  (n_valid_windows x n_pca)

Usage
-----
  python -m src.analyses.characterize_states --dataset fakeddit --k 11
  python -m src.analyses.characterize_states --dataset yelp --k 16 --no_tsv

Requirements: numpy, pandas
"""

import os
import sys
import argparse
import textwrap

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.datasets.registry import get_spec


# Default per-dataset category column for the raw-TSV breakdown. Datasets absent
# from this map (or with a None value) get no category breakdown.
DEFAULT_CATEGORY_COL = {
    "fakeddit": "subreddit",
    "yelp": None,
}


def parse_args():
    """Parse CLI arguments and fill in dataset-derived default paths."""
    p = argparse.ArgumentParser(
        description="Per-state characterisation of the HMM decode, "
                    "dataset-generalized.")
    p.add_argument("--dataset", required=True, choices=["fakeddit", "yelp"],
                   help="Which dataset spec to use (drives stem + default paths).")
    p.add_argument("--k", type=int, required=True,
                   help="HMM state count from the decode step (used for default "
                        "paths; authoritative k is read from the decode npz).")
    p.add_argument("--decode_npz", default=None,
                   help="Defaults to data/hmm_hmm/<dataset>/<dataset>_decode_k<k>.npz")
    p.add_argument("--weights_npz", default=None,
                   help="Defaults to data/hmm_weights/<dataset>/weights_pca.npz")
    p.add_argument("--manifest", default=None,
                   help="Defaults to data/splits/hmm_windows/<dataset>/"
                        "<dataset>_windows_manifest.csv")
    p.add_argument("--splits_dir", default=None,
                   help="Dir with the per-window split TSVs (for the category "
                        "breakdown). Defaults to data/splits/hmm_windows/<dataset>")
    p.add_argument("--f1_npz", default=None,
                   help="Col-centered cross-window F1. Defaults to "
                        "data/hmm_perf/<dataset>/cross_window_f1_colcentered.npz")
    p.add_argument("--output_dir", default=None,
                   help="Defaults to data/state_characterisation/<dataset>/k<k>")
    p.add_argument("--class_names", default=None,
                   help="Optional comma-separated human-readable class names, in "
                        "cls_* column order. If omitted, the cls_* column names "
                        "are used.")
    p.add_argument("--category_col", default=None,
                   help="Column in the split TSVs to tabulate per state "
                        "(e.g. 'subreddit'). Defaults to the dataset's usual "
                        "column; pass '' or use --no_tsv to disable.")
    p.add_argument("--top_categories", type=int, default=10)
    p.add_argument("--no_tsv", action="store_true",
                   help="Skip loading raw window TSVs (category breakdown off).")
    args = p.parse_args()

    ds = args.dataset
    if args.decode_npz is None:
        args.decode_npz = f"data/hmm_hmm/{ds}/{ds}_decode_k{args.k}.npz"
    if args.weights_npz is None:
        args.weights_npz = f"data/hmm_weights/{ds}/weights_pca.npz"
    if args.manifest is None:
        args.manifest = f"data/splits/hmm_windows/{ds}/{ds}_windows_manifest.csv"
    if args.splits_dir is None:
        args.splits_dir = f"data/splits/hmm_windows/{ds}"
    if args.f1_npz is None:
        args.f1_npz = f"data/hmm_perf/{ds}/cross_window_f1_colcentered.npz"
    if args.output_dir is None:
        args.output_dir = f"data/state_characterisation/{ds}/k{args.k}"
    if args.category_col is None:
        args.category_col = DEFAULT_CATEGORY_COL.get(ds)
    return args


def discover_cls_cols(manifest: pd.DataFrame) -> list[str]:
    """Return the manifest's class-count columns, sorted by integer label."""
    cols = [c for c in manifest.columns if c.startswith("cls_")]
    if not cols:
        raise ValueError(
            "Manifest has no cls_* columns; cannot build class distributions. "
            f"Columns present: {list(manifest.columns)}")

    def _key(c: str):
        tail = c[len("cls_"):]
        return (0, int(tail)) if tail.isdigit() else (1, tail)

    return sorted(cols, key=_key)


def hline(char="─", width=80):
    """Return a horizontal rule string of the given character and width."""
    return char * width


def fmt_pct(v):
    """Format a percentage value to one decimal place with a percent sign."""
    return f"{v:6.1f}%"


def load_categories(splits_dir, manifest, category_col):
    """Load raw window TSVs and return a dict of per-window value_counts.

    Returns None if the column or a TSV is missing.
    """
    results = {}
    for _, row in manifest.iterrows():
        idx = int(row["window_local_idx"])
        fname = row.get("filename")
        if not isinstance(fname, str):
            print("  [warn] manifest has no 'filename' column — category breakdown skipped")
            return None
        path = os.path.join(splits_dir, fname)
        if not os.path.exists(path):
            print(f"  [warn] TSV not found: {path} — category breakdown skipped")
            return None
        try:
            df = pd.read_csv(path, sep="\t", usecols=[category_col], low_memory=False)
        except (ValueError, KeyError):
            print(f"  [warn] column '{category_col}' not in {path} — "
                  f"category breakdown skipped")
            return None
        results[idx] = df[category_col].value_counts()
    return results


def main():
    """Run the per-state characterisation and write the report and CSV outputs."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Validate the dataset name against the registry (fail fast on typos).
    spec = get_spec(args.dataset)

    WIDTH = 88

    # Load the decode. The authoritative k is read from the decode file; under a
    # left-to-right topology a state may be unvisited, so max()+1 is only a
    # fallback.
    dec = np.load(args.decode_npz, allow_pickle=True)
    state_seq = dec["state_seq"].astype(int)
    window_ids = dec["window_ids"].astype(int)
    k = int(dec["k"]) if "k" in dec else int(state_seq.max() + 1)
    hmm_means = dec["means"].astype(float)
    trans_mat = dec.get("transition_matrix")
    if trans_mat is not None:
        trans_mat = np.array(trans_mat, dtype=float)

    n_pca = hmm_means.shape[1]
    N = len(state_seq)
    print(f"[{spec.name}] Loaded decode: k={k}, N={N} windows, n_pca={n_pca}")

    # Load PCA-space centroids. The HMM was fit and decoded on the z-scored
    # centroid sequence, so hmm_means are already in z-scaled PCA space.
    wpca = np.load(args.weights_npz, allow_pickle=True)
    C = wpca["C"].astype(float)
    Z_scaled = wpca["Z_scaled"].astype(float)
    centroid_wins = (wpca["centroid_wins"].astype(int)
                     if "centroid_wins" in wpca else np.arange(C.shape[0]))
    print(f"Centroid matrix C shape: {C.shape}   Z_scaled shape: {Z_scaled.shape}")

    # Load manifest, align to the decode window order, and discover class columns.
    manifest = pd.read_csv(args.manifest, parse_dates=["start_date", "end_date"])
    manifest = manifest.set_index("window_local_idx").loc[window_ids].reset_index()
    cls_cols = discover_cls_cols(manifest)
    for c in cls_cols:
        manifest[c] = manifest[c].fillna(0).astype(int)
    n_classes = len(cls_cols)

    # Human-readable class names, defaulting to the raw cls_* column names.
    if args.class_names:
        class_names = [s.strip() for s in args.class_names.split(",")]
        if len(class_names) != n_classes:
            raise ValueError(
                f"--class_names has {len(class_names)} entries but the manifest "
                f"has {n_classes} class columns {cls_cols}")
    else:
        class_names = list(cls_cols)

    # Prefer an explicit sampled_n column; otherwise fall back to the row sum of
    # the class-count columns so the script works for any manifest layout.
    if "sampled_n" in manifest.columns:
        manifest["_total_n"] = manifest["sampled_n"].fillna(0).astype(int)
    else:
        manifest["_total_n"] = manifest[cls_cols].sum(axis=1).astype(int)

    # Optional category breakdown from the raw TSVs.
    cat_col = None if (args.no_tsv or not args.category_col) else args.category_col
    cat_data = load_categories(args.splits_dir, manifest, cat_col) if cat_col else None

    # Load the col-centered F1 matrix. If absent, F1 stats are skipped but the
    # report is still produced.
    f1_available = False
    f1_matrix = None
    if os.path.exists(args.f1_npz):
        f1_data = np.load(args.f1_npz, allow_pickle=True)
        key = ("f1_matrix_colcentered" if "f1_matrix_colcentered" in f1_data
               else "f1_matrix")
        f1_matrix = f1_data[key].astype(float)
        f1_available = True
        print(f"Loaded F1 matrix ({key}): {f1_matrix.shape}")
    else:
        print(f"  [warn] F1 matrix not found at {args.f1_npz} — F1 stats skipped")

    # Build the per-state summary.
    lines, rows, cls_rows, cat_rows = [], [], [], []

    def pr(*a):
        lines.append(" ".join(str(x) for x in a))

    pr(hline("═", WIDTH))
    pr(f"  HMM STATE CHARACTERISATION  —  dataset={spec.name}  k={k}")
    pr(hline("═", WIDTH))
    pr()
    pr(f"  Windows : {N}   (window IDs {window_ids[0]}–{window_ids[-1]})")
    pr(f"  Date span: {manifest['start_date'].min().date()} → "
       f"{manifest['end_date'].max().date()}")
    pr(f"  Classes : {n_classes}   {cls_cols}")
    pr(f"  PCA dims : {n_pca}   |   HMM emission means shape: {hmm_means.shape}")
    pr()

    pr("  State sequence (window index → state):")
    seq_str = "  " + "  ".join(f"W{i:02d}→S{s}" for i, s in zip(window_ids, state_seq))
    for chunk in textwrap.wrap(seq_str, width=WIDTH - 2, subsequent_indent="    "):
        pr(chunk)
    pr()

    if trans_mat is not None:
        pr("  Transition matrix  (row = from-state, col = to-state):")
        pr("        " + "  ".join(f"S{j:1d}" for j in range(k)))
        for i in range(k):
            pr(f"  S{i:1d}  [ " + "  ".join(f"{trans_mat[i, j]:.3f}"
                                            for j in range(k)) + " ]")
        pr()

    for s in range(k):
        mask = state_seq == s
        win_idxs = window_ids[mask]
        local_idxs = np.where(mask)[0]
        state_manifest = manifest[mask]

        n_windows = int(mask.sum())
        if n_windows == 0:
            pr(hline("─", WIDTH))
            pr(f"  STATE {s}  │  (unvisited under the decoded path)")
            pr(hline("─", WIDTH))
            pr()
            cls_rows.append({"state": s,
                             **{class_names[c]: 0.0 for c in range(n_classes)}})
            row = {"state": s, "n_windows": 0, "window_ids": "",
                   "start_date": "", "end_date": "", "total_samples": 0,
                   "within_f1_mean": np.nan, "across_f1_mean": np.nan,
                   "f1_gap": np.nan}
            for i in range(n_pca):
                row[f"pca_mean_scaled_PC{i+1}"] = round(hmm_means[s][i], 6)
            rows.append(row)
            continue

        start_date = state_manifest["start_date"].min().date()
        end_date = state_manifest["end_date"].max().date()
        total_samp = int(state_manifest["_total_n"].sum())

        cls_counts = {c: int(state_manifest[cls_cols[c]].sum())
                      for c in range(n_classes)}
        cls_total = sum(cls_counts.values())
        cls_pcts = {c: (cls_counts[c] / cls_total * 100 if cls_total > 0 else 0.0)
                    for c in range(n_classes)}

        # Split every off-diagonal F1 pair for windows in this state into
        # within-state and across-state, so the two means can be compared.
        if f1_available:
            within_f1, across_f1 = [], []
            for i in local_idxs:
                for j in range(N):
                    if i == j:
                        continue
                    v = f1_matrix[i, j]
                    if np.isnan(v):
                        continue
                    (within_f1 if state_seq[j] == s else across_f1).append(v)
            wf = float(np.mean(within_f1)) if within_f1 else float("nan")
            af = float(np.mean(across_f1)) if across_f1 else float("nan")
            gap = wf - af
        else:
            wf = af = gap = float("nan")
            within_f1 = across_f1 = []

        pr(hline("─", WIDTH))
        pr(f"  STATE {s}  │  Windows: {n_windows}  │  "
           f"{start_date}  →  {end_date}  │  Samples: {total_samp:,}")
        pr(hline("─", WIDTH))
        pr()
        pr("  Window IDs in this state:",
           ", ".join(str(w) for w in sorted(win_idxs)))
        pr()

        pr(f"  {n_classes}-way class distribution:")
        pr(f"    {'Class':<25} {'Count':>8}  {'%':>7}")
        pr(f"    {'─'*25} {'─'*8}  {'─'*7}")
        for c in range(n_classes):
            bar = "█" * int(cls_pcts[c] / 2)
            pr(f"    {class_names[c]:<25} {cls_counts[c]:>8,}  "
               f"{fmt_pct(cls_pcts[c])}  {bar}")
        pr()

        pca_dim_labels = [f"PC{i+1}" for i in range(n_pca)]
        pr("  HMM emission mean  (Z-scaled PCA space):")
        for lbl, v in zip(pca_dim_labels, hmm_means[s]):
            pr(f"    {lbl}: {v:+.4f}")
        pr()

        if f1_available:
            pr("  Cross-window F1 generalisation (col-centered macro F1):")
            pr(f"    Within-state mean F1  : {wf:.4f}  (n={len(within_f1):,} pairs)")
            pr(f"    Across-state mean F1  : {af:.4f}  (n={len(across_f1):,} pairs)")
            pr(f"    Gap (within − across) : {gap:+.4f}")
        pr()

        if cat_data is not None:
            agg = pd.Series(dtype=float)
            for local_i in local_idxs:
                agg = agg.add(cat_data.get(int(local_i), pd.Series(dtype=float)),
                              fill_value=0)
            agg = agg.sort_values(ascending=False)
            total_posts = agg.sum()
            pr(f"  Top-{args.top_categories} {cat_col} (of {len(agg):,} unique):")
            pr(f"    {cat_col:<35} {'Count':>7}  {'%':>7}")
            pr(f"    {'─'*35} {'─'*7}  {'─'*7}")
            for name, cnt in agg.head(args.top_categories).items():
                pct = cnt / total_posts * 100 if total_posts else 0.0
                pr(f"    {str(name):<35} {int(cnt):>7,}  {fmt_pct(pct)}")
            pr()
            for rank, (name, cnt) in enumerate(
                    agg.head(args.top_categories).items(), 1):
                cat_rows.append({
                    "state": s, "rank": rank, cat_col: name,
                    "count": int(cnt),
                    "pct": cnt / total_posts * 100 if total_posts else 0.0,
                })

        row = {
            "state": s,
            "n_windows": n_windows,
            "window_ids": ";".join(str(w) for w in sorted(win_idxs)),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "total_samples": total_samp,
            "within_f1_mean": round(wf, 4) if not np.isnan(wf) else np.nan,
            "across_f1_mean": round(af, 4) if not np.isnan(af) else np.nan,
            "f1_gap": round(gap, 4) if not np.isnan(gap) else np.nan,
        }
        for i in range(n_pca):
            row[f"pca_mean_scaled_PC{i+1}"] = round(hmm_means[s][i], 6)
        rows.append(row)

        cls_rows.append({"state": s,
                         **{class_names[c]: round(cls_pcts[c], 2)
                            for c in range(n_classes)}})

    # Summarise each PC column of C across windows (spread of the centroid
    # cloud), since C rows are per-window centroids, not PCA loadings.
    pr(hline("═", WIDTH))
    pr("  PER-WINDOW CENTROIDS  (C matrix, z-scaled PCA space)")
    pr(f"  Shape: {C.shape}  —  rows = windows, cols = PCA components")
    pr()
    pr(f"  {'PC':<6}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    pr(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
    for j in range(C.shape[1]):
        col = C[:, j]
        pr(f"  PC{j+1:<4}  {col.mean():>10.4f}  {col.std():>10.4f}  "
           f"{col.min():>10.4f}  {col.max():>10.4f}")
    pr()

    if k > 1:
        pr("  Pairwise Euclidean distances between state centroids (Z-scaled):")
        pr("        " + "  ".join(f"  S{j}" for j in range(k)))
        for i in range(k):
            vals = [f"{np.linalg.norm(hmm_means[i] - hmm_means[j]):5.3f}"
                    for j in range(k)]
            pr(f"  S{i}  [ " + "  ".join(vals) + " ]")
        pr()

    pr(hline("═", WIDTH))

    # Write outputs.
    report_path = os.path.join(args.output_dir, "state_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n── Saved report → {report_path}")

    pd.DataFrame(rows).to_csv(
        os.path.join(args.output_dir, "state_summary.csv"), index=False)
    print(f"── Saved summary → {os.path.join(args.output_dir, 'state_summary.csv')}")

    pd.DataFrame(cls_rows).to_csv(
        os.path.join(args.output_dir, "state_class_dist.csv"), index=False)
    print(f"── Saved class dist → "
          f"{os.path.join(args.output_dir, 'state_class_dist.csv')}")

    if cat_rows:
        pd.DataFrame(cat_rows).to_csv(
            os.path.join(args.output_dir, "state_categories.csv"), index=False)
        print(f"── Saved categories → "
              f"{os.path.join(args.output_dir, 'state_categories.csv')}")

    centroid_df = pd.DataFrame(
        C,
        index=pd.Index(centroid_wins, name="window_id"),
        columns=[f"PC{j+1}" for j in range(C.shape[1])],
    )
    centroid_df.to_csv(os.path.join(args.output_dir, "window_centroids.csv"))
    print(f"── Saved window centroids → "
          f"{os.path.join(args.output_dir, 'window_centroids.csv')}")

    print("\nDone.")


if __name__ == "__main__":
    main()