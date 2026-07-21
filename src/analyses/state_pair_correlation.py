"""
state_pair_correlation.py

State-pair correlation analysis, generalized across datasets.

Correlates cross-window transfer F1 with two complementary measures of
state-pair dissimilarity, aggregating over the k*(k-1) ordered off-diagonal
state pairs (i->j and j->i kept separate, since transfer F1 is asymmetric):

  * data-side  : Jensen-Shannon divergence between the two states' class
                 distributions (built from the manifest cls_* columns).
  * model-side : Euclidean distance between the two states' HMM emission means
                 (centroids) in z-scored PCA weight space.

Both are expected to correlate negatively with mean transfer F1: a model
trained on one state should transfer worse to a more dissimilar state.

The manifest's cls_* columns are discovered dynamically, so the state class
distributions have whatever width the dataset actually has.

Outputs  (--output_dir)
-----------------------
  statepair_f1.csv          mean (col-centered) F1 + both dissimilarities per pair
  correlation_results.csv   Spearman r, p (parametric + permutation) per measure
  plot_f1_vs_jsd.png        scatter: F1 vs JS divergence
  plot_f1_vs_pca_dist.png   scatter: F1 vs PCA centroid distance
  plot_correlation_null.png permutation null distributions for both measures

Usage
-----
  python -m src.analyses.state_pair_correlation --dataset fakeddit --k 7

  python -m src.analyses.state_pair_correlation --dataset yelp --k 5 \
      --decode_npz data/hmm_hmm/yelp/yelp_decode_k5.npz \
      --f1_npz     data/hmm_perf/yelp/cross_window_f1_colcentered.npz \
      --manifest   data/splits/hmm_windows/yelp/yelp_windows_manifest.csv \
      --output_dir data/state_pair_correlation/yelp/k5

Requirements: numpy, pandas, scipy, matplotlib, tqdm
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.datasets.registry import get_spec


STATE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#17becf", "#e377c2", "#7f7f7f", "#bcbd22",
]


def parse_args():
    """Parse CLI arguments and fill in dataset-derived default paths."""
    p = argparse.ArgumentParser(
        description="State-pair correlation of transfer F1 with class-distribution "
                    "(JSD) and weight-space (PCA-centroid) dissimilarity, "
                    "dataset-generalized.")
    p.add_argument("--dataset", required=True, choices=["fakeddit", "yelp"],
                   help="Which dataset spec to use (drives stem + default paths).")
    p.add_argument("--k", type=int, required=True,
                   help="Number of HMM states chosen in the decode step (used "
                        "for default paths; authoritative k is read from the "
                        "decode npz).")
    p.add_argument("--decode_npz", default=None,
                   help="HMM decode file from decode_hmm.py. Defaults to "
                        "data/hmm_hmm/<dataset>/<dataset>_decode_k<k>.npz")
    p.add_argument("--f1_npz", default=None,
                   help="Col-centered cross-window F1 from the merge step. "
                        "Defaults to "
                        "data/hmm_perf/<dataset>/cross_window_f1_colcentered.npz")
    p.add_argument("--manifest", default=None,
                   help="Window manifest with cls_* columns. Defaults to "
                        "data/splits/hmm_windows/<dataset>/"
                        "<dataset>_windows_manifest.csv")
    p.add_argument("--output_dir", default=None,
                   help="Defaults to data/state_pair_correlation/<dataset>/k<k>")
    p.add_argument("--n_permutations", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ds = args.dataset
    if args.decode_npz is None:
        args.decode_npz = f"data/hmm_hmm/{ds}/{ds}_decode_k{args.k}.npz"
    if args.f1_npz is None:
        args.f1_npz = f"data/hmm_perf/{ds}/cross_window_f1_colcentered.npz"
    if args.manifest is None:
        args.manifest = f"data/splits/hmm_windows/{ds}/{ds}_windows_manifest.csv"
    if args.output_dir is None:
        args.output_dir = f"data/state_pair_correlation/{ds}/k{args.k}"
    return args


def discover_cls_cols(manifest: pd.DataFrame) -> list[str]:
    """Return the manifest's class-count columns, sorted by integer label.

    Sorting by the trailing integer keeps a stable, readable class order in the
    output distributions.
    """
    cols = [c for c in manifest.columns if c.startswith("cls_")]
    if not cols:
        raise ValueError(
            "Manifest has no cls_* columns; cannot build class distributions. "
            f"Columns present: {list(manifest.columns)}")

    def _key(c: str):
        tail = c[len("cls_"):]
        return (0, int(tail)) if tail.isdigit() else (1, tail)

    return sorted(cols, key=_key)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Return the Jensen-Shannon divergence (base-2, range [0, 1])."""
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def pca_euclidean(means_i: np.ndarray, means_j: np.ndarray) -> float:
    """Return the Euclidean distance between two states' emission means."""
    return float(np.linalg.norm(means_i - means_j))


def state_class_props(manifest: pd.DataFrame,
                      state_seq: np.ndarray,
                      k: int,
                      cls_cols: list[str]) -> np.ndarray:
    """Return (k, n_classes) class proportions per state.

    States never visited under a Bakis topology get an all-zero row.
    """
    n_classes = len(cls_cols)
    props = np.zeros((k, n_classes), dtype=float)
    for s in range(k):
        mask = state_seq == s
        sub = manifest[mask]
        counts = np.array([sub[c].sum() for c in cls_cols], dtype=float)
        total = counts.sum()
        props[s] = counts / total if total > 0 else counts
    return props


def build_pair_table(state_seq, f1_matrix, class_props, hmm_means, k):
    """Build one row per ordered state pair (i, j), i != j.

    mean_f1 is the mean col-centered transfer F1 over every window pair (ti in
    state i, tj in state j).
    """
    rows = []
    for si in range(k):
        for sj in range(k):
            if si == sj:
                continue

            idx_i = np.where(state_seq == si)[0]
            idx_j = np.where(state_seq == sj)[0]

            vals = []
            for ti in idx_i:
                for tj in idx_j:
                    v = f1_matrix[ti, tj]
                    if not np.isnan(v):
                        vals.append(v)

            mean_f1 = float(np.mean(vals)) if vals else np.nan
            n_pairs = len(vals)

            jsd = js_divergence(class_props[si], class_props[sj])
            pdist = pca_euclidean(hmm_means[si], hmm_means[sj])

            rows.append({
                "state_i": si,
                "state_j": sj,
                "mean_f1": round(mean_f1, 4) if not np.isnan(mean_f1) else np.nan,
                "n_pairs": n_pairs,
                "jsd": round(jsd, 6),
                "pca_dist": round(pdist, 6),
            })

    return pd.DataFrame(rows)


def spearman_perm_test(x: np.ndarray, y: np.ndarray,
                       n_perm: int, rng: np.random.Generator,
                       desc: str = "") -> dict:
    """Return Spearman r plus a one-tailed permutation p-value testing r <= observed."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    r_obs, p_param = stats.spearmanr(x, y)

    null_r = np.empty(n_perm)
    for b in tqdm(range(n_perm), desc=desc, leave=False):
        null_r[b] = stats.spearmanr(rng.permutation(x), y)[0]

    p_perm = float((null_r <= r_obs).mean())
    return {"r": float(r_obs), "p_param": float(p_param),
            "p_perm": p_perm, "n": int(mask.sum()), "null_r": null_r}


def _scatter(x, y, xlabel, r, p_perm, output_path, color):
    """Scatter x vs y with a fitted line, annotating the Spearman r and perm p."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(x, y, s=40, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
    if len(x) >= 2:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(coef, xs), color="black", lw=1.5, ls="--", alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Mean col-centered transfer F1", fontsize=12)
    ax.set_title(f"Spearman r = {r:+.3f}   (perm p = {p_perm:.4f})", fontsize=12)
    ax.grid(True, alpha=0.3, ls="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_correlation_nulls(res_jsd, res_pca, output_path):
    """Plot the permutation null distributions for both correlations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, res, label, color in [
        (axes[0], res_jsd, "JS divergence", "#2ca02c"),
        (axes[1], res_pca, "PCA centroid distance", "#d62728"),
    ]:
        ax.hist(res["null_r"], bins=60, color="#94A3B8",
                edgecolor="white", linewidth=0.3, alpha=0.85, label="Null r")
        ax.axvline(res["r"], color=color, lw=2.5,
                   label=f"Observed r = {res['r']:+.3f}  (p = {res['p_perm']:.4f})")
        ax.set_xlabel(f"Spearman r (F1 vs {label})", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, ls="--")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Permutation nulls for state-pair correlations", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    """Run the state-pair correlation analysis and write outputs and plots."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Validate the dataset name against the registry (fail fast on typos).
    spec = get_spec(args.dataset)

    print("=" * 60)
    print(f"State-pair correlation analysis  [{spec.name}]")
    print("=" * 60)

    dec = np.load(args.decode_npz, allow_pickle=True)
    state_seq = dec["state_seq"].astype(int)
    window_ids = dec["window_ids"].astype(int)
    k = int(dec["k"]) if "k" in dec else int(state_seq.max() + 1)
    hmm_means = dec["means"].astype(float)   # (k, n_pca), z-scaled

    f1_data = np.load(args.f1_npz, allow_pickle=True)
    f1_matrix = f1_data["f1_matrix_colcentered"].astype(float)

    N = len(state_seq)
    assert f1_matrix.shape == (N, N), (
        f"state_seq length {N} does not match f1_matrix shape {f1_matrix.shape}")

    manifest = pd.read_csv(args.manifest)
    manifest = manifest.set_index("window_local_idx").loc[window_ids].reset_index()
    cls_cols = discover_cls_cols(manifest)
    for c in cls_cols:
        manifest[c] = manifest[c].fillna(0).astype(int)

    print(f"Loaded: k={k}, N={N} windows, {len(cls_cols)} classes {cls_cols}")

    class_props = state_class_props(manifest, state_seq, k, cls_cols)

    print("\nPer-state class proportions:")
    header = f"  {'State':<8}" + "".join(f"{c:>12}" for c in cls_cols)
    print(header)
    for s in range(k):
        row = f"  {s:<8}" + "".join(f"{class_props[s, c]:>11.1%}"
                                    for c in range(len(cls_cols)))
        print(row)

    print("\nBuilding state-pair table ...")
    df = build_pair_table(state_seq, f1_matrix, class_props, hmm_means, k)

    pair_path = os.path.join(args.output_dir, "statepair_f1.csv")
    df.to_csv(pair_path, index=False)
    print(f"  Saved: {pair_path}")
    print(df.to_string(index=False))

    f1_vals = df["mean_f1"].values
    jsd_vals = df["jsd"].values
    pca_vals = df["pca_dist"].values

    print(f"\nRunning Spearman + permutation tests (n={args.n_permutations:,}) ...")
    res_jsd = spearman_perm_test(jsd_vals, f1_vals, args.n_permutations, rng,
                                 desc="JSD permutation")
    res_pca = spearman_perm_test(pca_vals, f1_vals, args.n_permutations, rng,
                                 desc="PCA dist permutation")

    print(f"\n── Correlation results ──────────────────────────────────────────")
    print(f"  F1 vs JS divergence  :  r = {res_jsd['r']:+.4f}  "
          f"p_param = {res_jsd['p_param']:.4f}  "
          f"p_perm = {res_jsd['p_perm']:.4f}  (n={res_jsd['n']})")
    print(f"  F1 vs PCA distance   :  r = {res_pca['r']:+.4f}  "
          f"p_param = {res_pca['p_param']:.4f}  "
          f"p_perm = {res_pca['p_perm']:.4f}  (n={res_pca['n']})")

    corr_df = pd.DataFrame([
        {"measure": "JSD", "spearman_r": res_jsd["r"],
         "p_param": res_jsd["p_param"], "p_perm": res_jsd["p_perm"],
         "n_pairs": res_jsd["n"]},
        {"measure": "PCA_dist", "spearman_r": res_pca["r"],
         "p_param": res_pca["p_param"], "p_perm": res_pca["p_perm"],
         "n_pairs": res_pca["n"]},
    ])
    corr_path = os.path.join(args.output_dir, "correlation_results.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"\n  Saved: {corr_path}")

    null_path = os.path.join(args.output_dir, "correlation_null.npz")
    np.savez(
        null_path,
        null_jsd=res_jsd["null_r"],
        obs_jsd=res_jsd["r"],
        p_jsd=res_jsd["p_perm"],
        null_pca=res_pca["null_r"],
        obs_pca=res_pca["r"],
        p_pca=res_pca["p_perm"],
    )
    print(f"\n  Saved: {null_path}")

    print("\nGenerating plots ...")
    _scatter(jsd_vals, f1_vals, "JS divergence between state class dists",
             res_jsd["r"], res_jsd["p_perm"],
             os.path.join(args.output_dir, "plot_f1_vs_jsd.png"), "#2ca02c")
    _scatter(pca_vals, f1_vals, "PCA centroid distance between states",
             res_pca["r"], res_pca["p_perm"],
             os.path.join(args.output_dir, "plot_f1_vs_pca_dist.png"), "#d62728")
    plot_correlation_nulls(res_jsd, res_pca,
                           os.path.join(args.output_dir, "plot_correlation_null.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()