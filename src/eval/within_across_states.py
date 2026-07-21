"""
within_across_states.py

Within- vs. across-state generalisation test, generalized across datasets.

Inspired by Baldassano et al. (2017, Neuron): tests whether windows that share
the same HMM latent state generalise better to each other than windows in
different states, after controlling for temporal distance.

State count k is read from the decode file's saved `k` field (authoritative
under a left-to-right / Bakis topology where a state may never be visited),
falling back to the number of distinct decoded states only if `k` is absent.

Core analyses
-------------
  1. Effect size        mean(within-state F1) − mean(across-state F1),
                        pooled over all off-diagonal pairs.
  2. Distance-stratified the same gap computed for each lag d = |i−j|.
  3. Distance-conditioned label shuffle
                        within each lag stratum independently, shuffle the
                        within/across labels n times. Statistic is the
                        harmonic-mean-weighted average per-lag gap. Fully
                        partials out the temporal-proximity confound.

Outputs (--output_dir)
----------------------
  within_across_summary.csv              one row per distance d + pooled row
  distance_conditioned_null.npz          label-shuffle null_gaps + observed gap
  plot_f1_heatmap_with_states.png        full F1 matrix with state boundaries
  plot_f1_vs_distance.png                within/across F1 ± 95 % CI vs lag
  plot_distance_conditioned_null.png     label-shuffle null histogram
  plot_statepair_heatmap.png             mean F1 for each (state_i, state_j) pair
  plot_state_timeline.png                HMM state sequence with F1 summary

Usage
-----
    # Fakeddit, k=5:
    python -m src.eval.within_across_states --dataset fakeddit --k 11

    # Yelp, k=18, explicit paths:
    python -m src.eval.within_across_states --dataset yelp --k 16 \
        --decode_npz data/hmm_hmm/yelp/yelp_decode_k16.npz \
        --f1_npz     data/hmm_perf/yelp/cross_window_f1_colcentered.npz \
        --output_dir data/hmm_within_across/yelp/k16 \
        --n_permutations 10000 --seed 42

Requirements: numpy, pandas, matplotlib, scipy, tqdm
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.datasets.registry import get_spec

warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_args():
    """Parse CLI arguments and fill in per-dataset default paths."""
    p = argparse.ArgumentParser(
        description="Within- vs. across-state generalisation test, "
                    "dataset-generalized and left-to-right-aware."
    )
    p.add_argument("--dataset", required=True, choices=["fakeddit", "yelp"],
                   help="Which dataset spec to use (drives stem + default paths).")
    p.add_argument("--k", type=int, required=True,
                   help="Number of HMM states chosen in the decode step "
                        "(used only to build default paths / filenames; the "
                        "authoritative k is read from the decode npz).")
    p.add_argument("--decode_npz", default=None,
                   help="HMM decode file from decode_hmm.py. Defaults to "
                        "data/hmm_hmm/<dataset>/<dataset>_decode_k<k>.npz")
    p.add_argument("--f1_npz", default=None,
                   help="Cross-window F1 matrix from the merge step. Defaults to "
                        "data/hmm_perf/<dataset>/cross_window_f1_colcentered.npz")
    p.add_argument("--output_dir", default=None,
                   help="Output directory. Defaults to "
                        "data/hmm_within_across/<dataset>/k<k>")
    p.add_argument("--n_permutations", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Derive per-dataset default paths (mirrors decode_hmm.py).
    ds = args.dataset
    if args.decode_npz is None:
        args.decode_npz = f"data/hmm_hmm/{ds}/{ds}_decode_k{args.k}.npz"
    if args.f1_npz is None:
        args.f1_npz = f"data/hmm_perf/{ds}/cross_window_f1_colcentered.npz"
    if args.output_dir is None:
        args.output_dir = f"data/hmm_within_across/{ds}/k{args.k}"
    return args


def load_inputs(decode_npz: str, f1_npz: str):
    """
    Load the HMM decode and the cross-window F1 matrix.

    Returns (state_seq, window_ids, f1_matrix, f1_per_seed_cube, k). k is taken
    from the decode file's saved `k` field when present (correct under a
    left-to-right topology where a state may be unvisited), falling back to the
    number of distinct decoded states otherwise.
    """
    dec = np.load(decode_npz, allow_pickle=True)
    state_seq = dec["state_seq"].astype(int)   # (N,)
    window_ids = dec["window_ids"].astype(int)  # (N,)

    # Prefer the saved k (handles unvisited states under Bakis topology); fall
    # back to the number of decoded states.
    if "k" in dec:
        k = int(dec["k"])
    else:
        k = int(len(np.unique(state_seq)))

    f1_data = np.load(f1_npz, allow_pickle=True)
    f1_matrix = f1_data["f1_matrix_colcentered"].astype(float)  # (N, N)

    # Per-seed cube (N, n_seeds, N) when present; fall back to None so the
    # script also runs on files that predate the per-seed output.
    if "f1_per_seed_cube" in f1_data:
        f1_per_seed_cube = f1_data["f1_per_seed_cube"].astype(float)
    else:
        f1_per_seed_cube = None

    N = len(state_seq)
    assert f1_matrix.shape == (N, N), (
        f"state_seq length {N} does not match f1_matrix shape {f1_matrix.shape}")

    n_seeds = (f1_per_seed_cube.shape[1]
               if f1_per_seed_cube is not None else "n/a")
    print(f"Loaded {N} windows, k={k} states, n_seeds={n_seeds}")
    print(f"State sequence: {state_seq}")
    return state_seq, window_ids, f1_matrix, f1_per_seed_cube, k


def build_pair_table(state_seq: np.ndarray,
                     f1_matrix: np.ndarray,
                     f1_per_seed_cube: np.ndarray | None = None) -> pd.DataFrame:
    """
    Return a DataFrame with one row per off-diagonal (i, j [, seed]) observation.

    When f1_per_seed_cube (N, n_seeds, N) is supplied, each (i, j) cell is
    expanded into up to n_seeds rows — one per seed — giving n_seeds× more
    observations. NaN seed slots (failed evals) are dropped. When it is None,
    falls back to window-mean F1.

    Columns: train_win, test_win, seed (int or -1), distance, same_state,
             state_i, state_j, f1
    """
    N = len(state_seq)
    rows = []

    if f1_per_seed_cube is not None:
        n_seeds = f1_per_seed_cube.shape[1]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                base = dict(
                    train_win  = i,
                    test_win   = j,
                    distance   = abs(i - j),
                    same_state = int(state_seq[i] == state_seq[j]),
                    state_i    = state_seq[i],
                    state_j    = state_seq[j],
                )
                for s in range(n_seeds):
                    v = f1_per_seed_cube[i, s, j]
                    # Skip failed-eval slots rather than polluting the table.
                    if np.isnan(v):
                        continue
                    rows.append({**base, "seed": s, "f1": v})
    else:
        # Fallback: window-mean F1.
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                rows.append(dict(
                    train_win  = i,
                    test_win   = j,
                    seed       = -1,
                    distance   = abs(i - j),
                    same_state = int(state_seq[i] == state_seq[j]),
                    state_i    = state_seq[i],
                    state_j    = state_seq[j],
                    f1         = f1_matrix[i, j],
                ))

    return pd.DataFrame(rows)


def within_across_gap(df: pd.DataFrame) -> float:
    """Mean(within-state F1) − mean(across-state F1)."""
    within = df.loc[df["same_state"] == 1, "f1"].mean()
    across = df.loc[df["same_state"] == 0, "f1"].mean()
    return float(within - across)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95,
                 rng: np.random.Generator = None) -> tuple:
    """Return (lower, upper) bootstrap CI for the mean."""
    if rng is None:
        rng = np.random.default_rng()
    if len(values) == 0:
        return (np.nan, np.nan)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return (np.percentile(boot_means, 100 * alpha),
            np.percentile(boot_means, 100 * (1 - alpha)))


def distance_stratified_summary(df: pd.DataFrame,
                                rng: np.random.Generator) -> pd.DataFrame:
    """
    For each lag d (and a pooled row), compute:
      within_mean, within_lo, within_hi,
      across_mean, across_lo, across_hi,
      gap, n_within, n_across
    """
    results = []
    distances = sorted(df["distance"].unique())

    for d in distances:
        sub = df[df["distance"] == d]
        w = sub.loc[sub["same_state"] == 1, "f1"].values
        a = sub.loc[sub["same_state"] == 0, "f1"].values

        # No same-state pairs at this lag — skip (common for large d).
        if len(w) == 0 or len(a) == 0:
            continue

        wlo, whi = bootstrap_ci(w, rng=rng)
        alo, ahi = bootstrap_ci(a, rng=rng)
        results.append({
            "distance":    d,
            "within_mean": w.mean(),
            "within_lo":   wlo,
            "within_hi":   whi,
            "across_mean": a.mean(),
            "across_lo":   alo,
            "across_hi":   ahi,
            "gap":         w.mean() - a.mean(),
            "n_within":    len(w),
            "n_across":    len(a),
        })

    # Pooled row.
    w_all = df.loc[df["same_state"] == 1, "f1"].values
    a_all = df.loc[df["same_state"] == 0, "f1"].values
    wlo, whi = bootstrap_ci(w_all, rng=rng)
    alo, ahi = bootstrap_ci(a_all, rng=rng)
    results.append({
        "distance":    "pooled",
        "within_mean": w_all.mean(),
        "within_lo":   wlo,
        "within_hi":   whi,
        "across_mean": a_all.mean(),
        "across_lo":   alo,
        "across_hi":   ahi,
        "gap":         w_all.mean() - a_all.mean(),
        "n_within":    len(w_all),
        "n_across":    len(a_all),
    })

    return pd.DataFrame(results)


def distance_conditioned_gap(df: pd.DataFrame) -> float:
    """
    Compute the weighted mean within-vs-across gap, stratified by lag.

    At each lag d, gap_d = mean(within F1 at d) − mean(across F1 at d). The
    pooled statistic is the weighted mean of gap_d, weighted by the harmonic
    mean of n_within and n_across at each lag (gives lower weight to lags where
    one class is very sparse). This is the observed statistic for the
    distance-conditioned test.
    """
    gaps = []
    weights = []
    for d, sub in df.groupby("distance"):
        w = sub.loc[sub["same_state"] == 1, "f1"].values
        a = sub.loc[sub["same_state"] == 0, "f1"].values
        if len(w) == 0 or len(a) == 0:
            continue
        gaps.append(w.mean() - a.mean())
        # Harmonic mean of counts — down-weights heavily imbalanced lags.
        weights.append(2 * len(w) * len(a) / (len(w) + len(a)))
    if not gaps:
        return 0.0
    weights = np.array(weights)
    return float(np.average(gaps, weights=weights))


def distance_conditioned_shuffle_test(
    df: pd.DataFrame,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple:
    """
    Distance-conditioned label shuffle test.

    Null: within each lag stratum independently, randomly reassign the
    same_state labels while preserving their counts at that lag. This destroys
    any association between state membership and F1 within each lag bin, while
    leaving the marginal distribution of F1 values at every lag intact. Because
    the shuffle is done within-lag, the null can never be inflated by the
    temporal proximity confound.

    The test statistic is distance_conditioned_gap(df): the harmonic-mean-
    weighted average of per-lag within-vs-across gaps.

    Returns (null_gaps, observed_gap, p_value). p_value is one-sided: fraction
    of null gaps >= observed gap.
    """
    observed_gap = distance_conditioned_gap(df)

    # Pre-group by distance so we don't re-group inside the loop, keeping only
    # lags that have at least one within pair and one across pair.
    lag_groups = {
        d: sub.copy()
        for d, sub in df.groupby("distance")
        if sub["same_state"].sum() > 0
        and (sub["same_state"] == 0).sum() > 0
    }

    # Pre-compute harmonic-mean weights (fixed across permutations).
    lag_weights = {}
    for d, sub in lag_groups.items():
        nw = sub["same_state"].sum()
        na = (sub["same_state"] == 0).sum()
        lag_weights[d] = 2 * nw * na / (nw + na)
    total_weight = sum(lag_weights.values())

    null_gaps = np.empty(n_permutations)

    for perm_idx in tqdm(range(n_permutations),
                         desc="Distance-conditioned shuffle"):
        perm_gap_num = 0.0
        for d, sub in lag_groups.items():
            labels = sub["same_state"].values.copy()
            rng.shuffle(labels)           # within-lag shuffle only
            f1 = sub["f1"].values
            w_mean = f1[labels == 1].mean()
            a_mean = f1[labels == 0].mean()
            perm_gap_num += lag_weights[d] * (w_mean - a_mean)
        null_gaps[perm_idx] = perm_gap_num / total_weight

    p_value = float((null_gaps >= observed_gap).mean())
    return null_gaps, observed_gap, p_value


def plot_distance_conditioned_null(null_gaps: np.ndarray, observed_gap: float,
                                   p_value: float, output_path: str):
    """Histogram of distance-conditioned null with observed statistic."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(null_gaps, bins=60, color=COLORS["null"], alpha=0.7,
            edgecolor="white", linewidth=0.4, label="Null distribution")
    ax.axvline(observed_gap, color=COLORS["obs"], lw=2.5,
               label=f"Observed gap = {observed_gap:.4f}")
    ax.axvline(np.percentile(null_gaps, 95), color="black", lw=1,
               linestyle="--", alpha=0.6, label="Null 95th pct")

    ax.set_xlabel("Distance-conditioned within − across F1 gap", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Distance-conditioned label-shuffle null  (n={len(null_gaps):,})\n"
        f"p = {p_value:.4f}  "
        f"[labels shuffled within each lag stratum independently]",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def statepair_mean_f1(df: pd.DataFrame, k: int) -> np.ndarray:
    """
    Return (k, k) matrix where entry [s_i, s_j] is the mean F1 when training on
    windows in state s_i and testing on windows in state s_j. Diagonal =
    within-state mean.
    """
    mat = np.full((k, k), np.nan)
    for si in range(k):
        for sj in range(k):
            sub = df[(df["state_i"] == si) & (df["state_j"] == sj)]
            if len(sub) > 0:
                mat[si, sj] = sub["f1"].mean()
    return mat


COLORS = {
    "within":  "#2196F3",   # blue
    "across":  "#F44336",   # red
    "gap":     "#4CAF50",   # green
    "null":    "#9E9E9E",   # grey
    "obs":     "#E91E63",   # pink/magenta
}

# Base palette; indexed modulo len() so it works for any k.
STATE_PALETTE = [
    "#E57373", "#81C784", "#64B5F6",
    "#FFD54F", "#BA68C8", "#4DB6AC",
    "#F06292", "#AED581", "#4FC3F7",
    "#FFB74D", "#9575CD", "#4DD0E1",
    "#A1887F", "#DCE775", "#7986CB",
    "#FF8A65", "#90A4AE", "#BA68C8",
]


def plot_f1_heatmap_with_states(
    f1_matrix: np.ndarray,
    state_seq: np.ndarray,
    window_ids: np.ndarray,
    k: int,
    output_path: str,
) -> None:
    """
    Full N×N cross-window F1 heatmap with HMM state bands overlaid.

    Adds coloured rectangles along both axes marking each window's state, and
    dashed grid lines at state boundaries so within-state blocks are visually
    obvious.
    """
    N = len(window_ids)
    band_width = 0.6          # thickness of the state-colour strip (axis units)

    # Mask the diagonal (within-distribution) so it doesn't bias the colormap.
    plot_mat = f1_matrix.copy().astype(float)
    np.fill_diagonal(plot_mat, np.nan)

    fig, ax = plt.subplots(figsize=(max(7, N * 0.35 + 2),
                                    max(6, N * 0.35 + 2)))

    # F1 heatmap.
    im = ax.imshow(
        plot_mat, aspect="auto", cmap="RdYlGn",
        vmin=np.nanmin(plot_mat), vmax=np.nanmax(plot_mat),
        extent=[-0.5, N - 0.5, N - 0.5, -0.5],   # keeps ticks aligned
    )
    plt.colorbar(im, ax=ax, label="Macro F1", fraction=0.035, pad=0.02)

    # State boundary lines. Draw twice: a thick white halo first, then a thinner
    # dark line on top, so the boundary reads clearly against any cell colour in
    # the RdYlGn map.
    boundaries = [b + 0.5 for b in range(N - 1) if state_seq[b] != state_seq[b + 1]]
    for b in boundaries:
        for lw, color, alpha in [(3.0, "white", 0.9), (1.2, "black", 0.7)]:
            ax.axvline(b, color=color, lw=lw, alpha=alpha, linestyle="--")
            ax.axhline(b, color=color, lw=lw, alpha=alpha, linestyle="--")

    # State colour strips along each axis.
    for i, s in enumerate(state_seq):
        c = STATE_PALETTE[s % len(STATE_PALETTE)]
        # Top strip (x-axis).
        ax.add_patch(mpatches.Rectangle(
            (i - 0.5, -0.5 - band_width), 1, band_width,
            color=c, clip_on=False, edgecolor="white", linewidth=0.3))
        # Left strip (y-axis).
        ax.add_patch(mpatches.Rectangle(
            (-0.5 - band_width, i - 0.5), band_width, 1,
            color=c, clip_on=False, edgecolor="white", linewidth=0.3))

    ax.set_xlim(-0.5 - band_width, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5 - band_width)
    ax.set_xlabel("Test window", fontsize=12)
    ax.set_ylabel("Train window", fontsize=12)
    ax.set_title("Cross-window macro F1 with HMM state boundaries", fontsize=13)

    legend_patches = [
        mpatches.Patch(color=STATE_PALETTE[s % len(STATE_PALETTE)],
                       label=f"State {s}")
        for s in range(k)
    ]
    ax.legend(handles=legend_patches, loc="upper left",
              bbox_to_anchor=(1.12, 1.0), fontsize=8,
              ncol=1 + (k // 10), framealpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_f1_vs_distance(summary_df: pd.DataFrame, output_path: str):
    """Within- and across-state mean F1 (± 95 % CI) as a function of lag."""
    df = summary_df[summary_df["distance"] != "pooled"].copy()
    df["distance"] = df["distance"].astype(int)
    df = df.sort_values("distance")

    fig, ax = plt.subplots(figsize=(9, 5))

    for kind, color, label in [
        ("within", COLORS["within"], "Within-state"),
        ("across", COLORS["across"], "Across-state"),
    ]:
        mean = df[f"{kind}_mean"].values
        lo = df[f"{kind}_lo"].values
        hi = df[f"{kind}_hi"].values
        d = df["distance"].values
        ax.plot(d, mean, color=color, lw=2, label=label, marker="o", ms=4)
        ax.fill_between(d, lo, hi, color=color, alpha=0.15)

    ax.set_xlabel("Temporal distance  |i − j|  (windows)", fontsize=12)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title("Cross-window F1 vs. temporal distance\n"
                 "split by HMM state membership", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_statepair_heatmap(mat: np.ndarray, k: int, output_path: str):
    """Heatmap of mean F1 for each (train_state, test_state) combination."""
    fig, ax = plt.subplots(figsize=(max(6, k * 0.7), max(5, k * 0.6)))

    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                   vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    plt.colorbar(im, ax=ax, label="Mean macro F1")

    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([f"State {s}" for s in range(k)], fontsize=9, rotation=90)
    ax.set_yticklabels([f"State {s}" for s in range(k)], fontsize=9)
    ax.set_xlabel("Test-window state", fontsize=11)
    ax.set_ylabel("Train-window state", fontsize=11)
    ax.set_title("Mean cross-window F1 by state pair", fontsize=12)

    # Annotate cells (skip when k is large enough that text would overlap).
    if k <= 10:
        for si in range(k):
            for sj in range(k):
                v = mat[si, sj]
                if not np.isnan(v):
                    ax.text(sj, si, f"{v:.3f}", ha="center", va="center",
                            fontsize=9,
                            color="black" if 0.3 < v < 0.85 else "white")

    # Diagonal outline (within-state cells).
    for s in range(k):
        rect = mpatches.FancyBboxPatch(
            (s - 0.5, s - 0.5), 1, 1,
            boxstyle="square,pad=0", linewidth=2.5,
            edgecolor="navy", facecolor="none"
        )
        ax.add_patch(rect)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_state_timeline(state_seq: np.ndarray, summary_df: pd.DataFrame,
                        k: int, output_path: str):
    """
    Two-panel figure:
      Top:    HMM state timeline (colour-coded bar per window).
      Bottom: pooled within/across F1 with 95 % CI (horizontal reference).
    """
    N = len(state_seq)
    pool = summary_df[summary_df["distance"] == "pooled"].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 5),
                             gridspec_kw={"height_ratios": [1, 2]})

    # Top: state timeline.
    ax0 = axes[0]
    for i, s in enumerate(state_seq):
        ax0.bar(i, 1, color=STATE_PALETTE[s % len(STATE_PALETTE)],
                edgecolor="white", linewidth=0.4)
    ax0.set_xlim(-0.5, N - 0.5)
    ax0.set_ylim(0, 1)
    ax0.set_yticks([])
    ax0.set_xlabel("")
    ax0.set_title("HMM state sequence (window index →)", fontsize=11)
    legend_patches = [
        mpatches.Patch(color=STATE_PALETTE[s % len(STATE_PALETTE)],
                       label=f"State {s}")
        for s in range(k)
    ]
    ax0.legend(handles=legend_patches, loc="upper right",
               fontsize=8, ncol=min(k, 9), framealpha=0.7)

    # Bottom: within vs across reference lines.
    ax1 = axes[1]
    w_mean, w_lo, w_hi = pool["within_mean"], pool["within_lo"], pool["within_hi"]
    a_mean, a_lo, a_hi = pool["across_mean"], pool["across_lo"], pool["across_hi"]

    ax1.axhline(w_mean, color=COLORS["within"], lw=2,
                label=f"Within-state mean F1 = {w_mean:.3f}")
    ax1.axhspan(w_lo, w_hi, alpha=0.15, color=COLORS["within"])

    ax1.axhline(a_mean, color=COLORS["across"], lw=2,
                label=f"Across-state mean F1 = {a_mean:.3f}")
    ax1.axhspan(a_lo, a_hi, alpha=0.15, color=COLORS["across"])

    ax1.set_ylabel("Macro F1", fontsize=11)
    ax1.set_xlabel("Window index", fontsize=11)
    ax1.set_title(f"Pooled within vs. across-state F1  "
                  f"(gap = {pool['gap']:+.4f})", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xlim(-0.5, N - 0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    """Run the full within- vs. across-state analysis and write all outputs."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Validate the dataset name against the registry (fail fast on typos, and
    # keep the script honest about being spec-driven even though the analysis
    # math is dataset-agnostic).
    spec = get_spec(args.dataset)

    # 1. Load data.
    print("=" * 60)
    print(f"Within- vs. across-state analysis  [{spec.name}]")
    print("=" * 60)
    state_seq, window_ids, f1_matrix, f1_per_seed_cube, k = load_inputs(
        args.decode_npz, args.f1_npz)
    N = len(state_seq)

    # 2. Build pair table.
    print("\nBuilding pair table ...")
    df = build_pair_table(state_seq, f1_matrix, f1_per_seed_cube)
    using_seeds = (f1_per_seed_cube is not None)
    obs_label = "per-seed" if using_seeds else "per-window-mean"
    print(f"  {len(df):,} off-diagonal observations ({obs_label})  "
          f"({df['same_state'].sum():,} within-state, "
          f"{(df['same_state']==0).sum():,} across-state)")

    # 3. Distance-stratified summary + pooled effect size.
    print("\nComputing distance-stratified summary ...")
    summary_df = distance_stratified_summary(df, rng)
    pooled_row = summary_df[summary_df["distance"] == "pooled"].iloc[0]
    print(f"  Pooled within-state  mean F1 = {pooled_row['within_mean']:.4f} "
          f"[{pooled_row['within_lo']:.4f}, {pooled_row['within_hi']:.4f}]")
    print(f"  Pooled across-state  mean F1 = {pooled_row['across_mean']:.4f} "
          f"[{pooled_row['across_lo']:.4f}, {pooled_row['across_hi']:.4f}]")
    print(f"  Gap (within − across)        = {pooled_row['gap']:+.4f}")

    csv_path = os.path.join(args.output_dir, "within_across_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # 4. Distance-conditioned label shuffle test.
    print(f"\nRunning distance-conditioned label shuffle "
          f"(n={args.n_permutations:,}) ...")
    dc_null_gaps, dc_observed_gap, dc_p_value = distance_conditioned_shuffle_test(
        df, args.n_permutations, rng
    )
    print(f"  Observed distance-conditioned gap = {dc_observed_gap:+.4f}")
    print(f"  p-value                           = {dc_p_value:.4f}  "
          f"({'significant' if dc_p_value < 0.05 else 'not significant'} "
          f"at α=0.05)")

    dc_npz_path = os.path.join(args.output_dir, "distance_conditioned_null.npz")
    np.savez(dc_npz_path, null_gaps=dc_null_gaps, observed_gap=dc_observed_gap,
             p_value=dc_p_value)
    print(f"  Saved: {dc_npz_path}")

    # 5. State-pair heatmap.
    sp_mat = statepair_mean_f1(df, k)

    # 6. Plots.
    print("\nGenerating plots ...")
    plot_f1_heatmap_with_states(
        f1_matrix, state_seq, window_ids, k,
        os.path.join(args.output_dir, "plot_f1_heatmap_with_states.png"))

    plot_f1_vs_distance(
        summary_df,
        os.path.join(args.output_dir, "plot_f1_vs_distance.png"))

    plot_distance_conditioned_null(
        dc_null_gaps, dc_observed_gap, dc_p_value,
        os.path.join(args.output_dir, "plot_distance_conditioned_null.png"))

    plot_statepair_heatmap(
        sp_mat, k,
        os.path.join(args.output_dir, "plot_statepair_heatmap.png"))

    plot_state_timeline(
        state_seq, summary_df, k,
        os.path.join(args.output_dir, "plot_state_timeline.png"))

    # 7. Print per-distance table.
    print("\nDistance-stratified summary:")
    print(summary_df.to_string(index=False, float_format="{:.4f}".format))

    print("\nDone.")


if __name__ == "__main__":
    main()