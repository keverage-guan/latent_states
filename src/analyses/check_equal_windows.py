"""
check_equal_windows.py

HMM-vs-equal-duration segmentation comparison, generalized across datasets.

Does the HMM segmentation predict cross-window generalisation better than a
naive equal-duration (calendar-time) segmentation? Both split the N windows into
exactly k contiguous groups:

  HMM segmentation   — the Viterbi-decoded state labels from decode_hmm.py.
  Equal-duration seg — N windows split into k contiguous, near-equal-size groups
                       (remainder distributed round-robin over the first groups).

For each segmentation we compute the pooled within/across mean F1 and the gap.
We then ask whether the HMM gap is significantly larger than the equal-duration
gap via a direct permutation test on the difference of gaps.

Outputs  (--output_dir)
-----------------------
  comparison_summary.csv           one row per method + one diff row
  distance_stratified_both.csv     per-lag within/across means for both methods
  plot_f1_vs_distance_both.png     within/across curves for both methods
  plot_direct_perm_test.png        null distribution for (HMM_gap − equal_gap)
  plot_segmentation_comparison.png colour strip: group per window, both methods
  null_diffs.npy, decode_snapshot.npz  (for figures)

Usage
-----
  python -m src.analyses.check_equal_windows --dataset fakeddit --k 11
  python -m src.analyses.check_equal_windows --dataset yelp --k 16 \
      --output_dir data/check_equal_windows/yelp/k16 --n_permutations 10000

Requirements: numpy, pandas, matplotlib, tqdm
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.datasets.registry import get_spec


def parse_args():
    """Parse CLI arguments and fill in dataset-derived default paths."""
    p = argparse.ArgumentParser(
        description="Compare HMM segmentation against an equal-duration baseline "
                    "for predicting cross-window transfer, dataset-generalized.")
    p.add_argument("--dataset", required=True, choices=["fakeddit", "yelp"],
                   help="Which dataset spec to use (drives stem + default paths).")
    p.add_argument("--k", type=int, required=True,
                   help="HMM state count from the decode step (also the number of "
                        "equal-duration groups; used for default paths. The "
                        "authoritative k is read from the decode npz).")
    p.add_argument("--decode_npz", default=None,
                   help="HMM decode file. Defaults to "
                        "data/hmm_hmm/<dataset>/<dataset>_decode_k<k>.npz")
    p.add_argument("--f1_npz", default=None,
                   help="Col-centered cross-window F1. Defaults to "
                        "data/hmm_perf/<dataset>/cross_window_f1_colcentered.npz")
    p.add_argument("--output_dir", default=None,
                   help="Defaults to data/check_equal_windows/<dataset>/k<k>")
    p.add_argument("--n_permutations", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ds = args.dataset
    if args.decode_npz is None:
        args.decode_npz = f"data/hmm_hmm/{ds}/{ds}_decode_k{args.k}.npz"
    if args.f1_npz is None:
        args.f1_npz = f"data/hmm_perf/{ds}/cross_window_f1_colcentered.npz"
    if args.output_dir is None:
        args.output_dir = f"data/check_equal_windows/{ds}/k{args.k}"
    return args


def equal_duration_labels(N: int, k: int) -> np.ndarray:
    """Assign N consecutive windows to k contiguous, near-equal-size groups.

    The remainder r = N % k is distributed round-robin: the first r groups get
    one extra window each. Example: N=35, k=6 -> sizes [6, 6, 6, 6, 6, 5].
    """
    base = N // k
    remainder = N % k
    sizes = [base + (1 if g < remainder else 0) for g in range(k)]
    labels = np.empty(N, dtype=int)
    start = 0
    for g, sz in enumerate(sizes):
        labels[start:start + sz] = g
        start += sz
    assert start == N
    return labels


def build_pair_df(label_seq: np.ndarray, f1_matrix: np.ndarray) -> pd.DataFrame:
    """Build one row per off-diagonal (i, j) cell with transfer F1 and grouping.

    Columns: train_win, test_win, distance, same_group, label_i, label_j, f1.
    """
    N = len(label_seq)
    rows = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            v = f1_matrix[i, j]
            if np.isnan(v):
                continue
            rows.append(dict(
                train_win=i, test_win=j,
                distance=abs(i - j),
                same_group=int(label_seq[i] == label_seq[j]),
                label_i=int(label_seq[i]),
                label_j=int(label_seq[j]),
                f1=float(v),
            ))
    return pd.DataFrame(rows)


def pooled_gap(df: pd.DataFrame):
    """Return (gap, within_mean, across_mean) of F1 pooled over all pairs."""
    within = df.loc[df["same_group"] == 1, "f1"].mean()
    across = df.loc[df["same_group"] == 0, "f1"].mean()
    return float(within - across), float(within), float(across)


def distance_stratified(df: pd.DataFrame):
    """Return per-distance within/across means, gap, and pair counts."""
    records = []
    for d, grp in df.groupby("distance"):
        w = grp.loc[grp["same_group"] == 1, "f1"]
        a = grp.loc[grp["same_group"] == 0, "f1"]
        records.append({
            "distance": d,
            "within_mean": w.mean() if len(w) else np.nan,
            "across_mean": a.mean() if len(a) else np.nan,
            "gap": w.mean() - a.mean() if (len(w) and len(a)) else np.nan,
            "n_within": len(w),
            "n_across": len(a),
        })
    return pd.DataFrame(records).sort_values("distance")


COLORS = {"hmm": "#1f77b4", "equal": "#ff7f0e"}


def plot_f1_vs_distance(strat_hmm, strat_eq, output_path):
    """Plot within/across F1 by temporal lag for both segmentations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, strat, label, color in [
        (axes[0], strat_hmm, "HMM segmentation", COLORS["hmm"]),
        (axes[1], strat_eq, "Equal-duration segmentation", COLORS["equal"]),
    ]:
        # Limit the x-axis to distances that have both within and across pairs,
        # so the two curves are directly comparable at every plotted lag.
        has_within = set(strat.loc[strat["n_within"] > 0, "distance"])
        has_across = set(strat.loc[strat["n_across"] > 0, "distance"])
        common_d = sorted(has_within & has_across)

        s = strat[strat["distance"].isin(common_d)].sort_values("distance")
        d = s["distance"].values

        ax.plot(d, s["within_mean"].values, color=color, lw=2,
                marker="o", ms=4, label="Within-group")
        ax.plot(d, s["across_mean"].values, color=color, lw=2,
                marker="s", ms=4, linestyle="--", label="Across-group", alpha=0.7)
        ax.fill_between(d, s["within_mean"].values, s["across_mean"].values,
                        alpha=0.10, color=color)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Temporal distance |i − j| (windows)", fontsize=11)
        if common_d:
            ax.set_xlim(min(common_d) - 0.5, max(common_d) + 0.5)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

    axes[0].set_ylabel("Macro F1 (col-centered)", fontsize=11)
    fig.suptitle("Within vs. across-group F1 by temporal lag\n"
                 "(x-axis limited to distances present in both within and "
                 "across pairs)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_direct_perm(null_diffs, obs_diff, p_val, output_path):
    """Plot the permutation null for (HMM gap − equal-duration gap)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(null_diffs, bins=60, color="steelblue", alpha=0.75,
            label="Null distribution")
    ax.axvline(obs_diff, color="red", lw=2,
               label=f"Observed diff = {obs_diff:+.4f}\np = {p_val:.4f}")
    ax.set_xlabel("HMM gap − equal-duration gap", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Direct permutation test: does HMM segment better than "
                 "equal-duration?", fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_state_comparison(hmm_labels, equal_labels, N, output_path):
    """Plot a two-row colour strip of group membership per window per method."""
    import matplotlib.cm as cm
    k = int(max(hmm_labels.max(), equal_labels.max()) + 1)
    cmap = cm.tab10
    colors = [cmap(i / 10) for i in range(k)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 3), gridspec_kw={"hspace": 0.6})
    for ax, labels, title in [
        (axes[0], hmm_labels, "HMM segmentation"),
        (axes[1], equal_labels, "Equal-duration segmentation"),
    ]:
        for i, lbl in enumerate(labels):
            ax.bar(i, 1, color=colors[lbl], edgecolor="white", linewidth=0.3)
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Window index →", fontsize=9)
        ax.set_title(title, fontsize=10)
        patches = [mpatches.Patch(color=colors[g], label=f"G{g}") for g in range(k)]
        ax.legend(handles=patches, loc="upper right", fontsize=7,
                  ncol=min(k, 9), framealpha=0.7)

    fig.suptitle("Segmentation comparison (each bar = one window)", fontsize=11)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    """Run the HMM-vs-equal-duration comparison and write outputs and plots."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Validate the dataset name against the registry (fail fast on typos).
    spec = get_spec(args.dataset)

    print("=" * 60)
    print(f"HMM vs. equal-duration segmentation  [{spec.name}]")
    print("=" * 60)

    dec = np.load(args.decode_npz, allow_pickle=True)
    hmm_labels = dec["state_seq"].astype(int)
    window_ids = dec["window_ids"].astype(int)
    N = len(hmm_labels)
    k = int(dec["k"]) if "k" in dec else int(hmm_labels.max() + 1)

    f1_data = np.load(args.f1_npz, allow_pickle=True)
    f1_matrix = f1_data["f1_matrix_colcentered"].astype(float)

    assert f1_matrix.shape == (N, N), (
        f"state_seq length {N} does not match f1_matrix shape {f1_matrix.shape}")

    print(f"N={N} windows, k={k}")
    print(f"F1 matrix shape: {f1_matrix.shape}  (col-centered, per-window mean)")

    # k drives the equal-duration group count, so both segmentations have the
    # same number of groups.
    equal_labels = equal_duration_labels(N, k)
    print(f"\nHMM    labels : {hmm_labels}")
    print(f"Equal  labels : {equal_labels}")

    print("\nBuilding pair tables ...")
    df_hmm = build_pair_df(hmm_labels, f1_matrix)
    df_equal = build_pair_df(equal_labels, f1_matrix)
    print(f"  HMM   pairs : {len(df_hmm):,}  (per-window mean)")
    print(f"  Equal pairs : {len(df_equal):,}")

    hmm_gap, hmm_w, hmm_a = pooled_gap(df_hmm)
    equal_gap, equal_w, equal_a = pooled_gap(df_equal)

    print(f"\n── Pooled results ───────────────────────────────")
    print(f"  HMM   : within={hmm_w:.4f}  across={hmm_a:.4f}  gap={hmm_gap:+.4f}")
    print(f"  Equal : within={equal_w:.4f}  across={equal_a:.4f}  gap={equal_gap:+.4f}")
    print(f"  HMM gap − Equal gap = {hmm_gap - equal_gap:+.4f}")

    strat_hmm = distance_stratified(df_hmm)
    strat_equal = distance_stratified(df_equal)

    # Permute the HMM labels; the equal-duration labels are fixed, so the null is
    # "the HMM grouping is no better than a random contiguity-free grouping."
    print(f"\nRunning direct permutation test (HMM gap > equal gap) "
          f"(n={args.n_permutations:,}) ...")
    obs_diff = hmm_gap - equal_gap
    null_diffs = np.empty(args.n_permutations)
    for b in tqdm(range(args.n_permutations), desc="Direct perm"):
        perm_labels = rng.permutation(hmm_labels)
        df_perm = build_pair_df(perm_labels, f1_matrix)
        g_perm, _, _ = pooled_gap(df_perm)
        null_diffs[b] = g_perm - equal_gap

    p_direct = float((null_diffs >= obs_diff).mean())
    print(f"  Observed HMM − equal gap = {obs_diff:+.4f}")
    print(f"  p-value (one-tailed)     = {p_direct:.4f}"
          f"  {'significant' if p_direct < 0.05 else 'not significant'} at α=0.05")

    summary_rows = [
        {"method": "HMM",
         "within_mean_f1": round(hmm_w, 4), "across_mean_f1": round(hmm_a, 4),
         "gap": round(hmm_gap, 4), "direct_perm_p": ""},
        {"method": "equal_duration",
         "within_mean_f1": round(equal_w, 4), "across_mean_f1": round(equal_a, 4),
         "gap": round(equal_gap, 4), "direct_perm_p": ""},
        {"method": "HMM_minus_equal (direct perm)",
         "within_mean_f1": "", "across_mean_f1": "",
         "gap": round(obs_diff, 4), "direct_perm_p": round(p_direct, 4)},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n── Saved summary → {summary_path}")

    strat_hmm["method"] = "HMM"
    strat_equal["method"] = "equal_duration"
    strat_both = pd.concat([strat_hmm, strat_equal], ignore_index=True)
    strat_path = os.path.join(args.output_dir, "distance_stratified_both.csv")
    strat_both.to_csv(strat_path, index=False)
    print(f"── Saved distance-stratified → {strat_path}")

    plot_f1_vs_distance(strat_hmm, strat_equal,
                        os.path.join(args.output_dir, "plot_f1_vs_distance_both.png"))
    plot_direct_perm(null_diffs, obs_diff, p_direct,
                     os.path.join(args.output_dir, "plot_direct_perm_test.png"))
    plot_state_comparison(hmm_labels, equal_labels, N,
                          os.path.join(args.output_dir, "plot_segmentation_comparison.png"))

    np.save(os.path.join(args.output_dir, "null_diffs.npy"), null_diffs)
    np.savez(os.path.join(args.output_dir, "decode_snapshot.npz"),
             hmm_labels=hmm_labels, equal_labels=equal_labels, N=N)

    print("\nDone.")


if __name__ == "__main__":
    main()