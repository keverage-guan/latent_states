"""
jsd_stability_analysis.py

JSD stability analysis, generalized across datasets.

For each pair of windows in the same HMM state, compute the Jensen-Shannon
divergence between their class distributions (from the manifest cls_* columns).
Do the same for every pair from different states. Report mean/variance of each
distribution, run a permutation test, and produce the standard plots consumed by
generate_all_figures.py.

The question: are within-state windows more class-compositionally similar to each
other than across-state pairs? If so, the HMM states partially reflect shifts in
class distribution — context for interpreting the within/across F1 advantage.

Outputs  (--output_dir)
-----------------------
  jsd_stability_summary.csv    mean/var per group + per state
  jsd_stability_null.npy       permutation null distribution
  jsd_within.npy               raw within-state JSD values
  jsd_across.npy               raw across-state JSD values
  plot_jsd_stability.png       violin + per-state bar (side by side)
  plot_jsd_stability_null.png  permutation null + observed Δμ

Usage
-----
  python -m src.analyses.jsd_stability_analysis --dataset fakeddit --k 11
  python -m src.analyses.jsd_stability_analysis --dataset yelp --k 16 \
      --output_dir data/jsd_stability/yelp/k16 --n_perm 10000

Requirements: numpy, pandas, matplotlib
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.datasets.registry import get_spec


def parse_args():
    """Parse CLI arguments and fill in dataset-derived default paths."""
    p = argparse.ArgumentParser(
        description="Within- vs. across-state pairwise JSD of class distributions, "
                    "dataset-generalized.")
    p.add_argument("--dataset", required=True, choices=["fakeddit", "yelp"],
                   help="Which dataset spec to use (drives stem + default paths).")
    p.add_argument("--k", type=int, required=True,
                   help="HMM state count from the decode step (used for default "
                        "paths; authoritative k is read from the decode npz).")
    p.add_argument("--decode_npz", default=None,
                   help="Defaults to data/hmm_hmm/<dataset>/<dataset>_decode_k<k>.npz")
    p.add_argument("--manifest", default=None,
                   help="Defaults to data/splits/hmm_windows/<dataset>/"
                        "<dataset>_windows_manifest.csv")
    p.add_argument("--output_dir", default=None,
                   help="Defaults to data/jsd_stability/<dataset>/k<k>")
    p.add_argument("--n_perm", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    ds = args.dataset
    if args.decode_npz is None:
        args.decode_npz = f"data/hmm_hmm/{ds}/{ds}_decode_k{args.k}.npz"
    if args.manifest is None:
        args.manifest = f"data/splits/hmm_windows/{ds}/{ds}_windows_manifest.csv"
    if args.output_dir is None:
        args.output_dir = f"data/jsd_stability/{ds}/k{args.k}"
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


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Return the Jensen-Shannon divergence (base-2, range [0, 1])."""
    p = np.asarray(p, float); p /= p.sum()
    q = np.asarray(q, float); q /= q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def load_class_props(manifest_path: str, window_ids: np.ndarray,
                     cls_cols: list[str] | None = None):
    """Load the manifest, align to window_ids, and return (props, cls_cols).

    props is (N, n_classes) of per-window class proportions.
    """
    df = pd.read_csv(manifest_path)
    id_col = next((c for c in df.columns if "window_local" in c.lower()),
                  df.columns[0])
    df = df.set_index(id_col).loc[window_ids]

    if cls_cols is None:
        cls_cols = discover_cls_cols(df)

    for c in cls_cols:
        if c not in df.columns:
            df[c] = 0

    counts = df[cls_cols].fillna(0).values.astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return counts / totals, cls_cols


def main():
    """Run the within- vs across-state JSD analysis and write outputs and plots."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    spec = get_spec(args.dataset)
    print("=" * 60)
    print(f"JSD stability analysis  [{spec.name}]")
    print("=" * 60)

    dec = np.load(args.decode_npz, allow_pickle=True)
    state_seq = dec["state_seq"].astype(int)
    win_ids = dec["window_ids"].astype(int)
    k = int(dec["k"]) if "k" in dec else int(state_seq.max() + 1)
    N = len(state_seq)
    print(f"Loaded: k={k}, N={N} windows")

    props, cls_cols = load_class_props(args.manifest, win_ids)
    print(f"Class columns: {cls_cols}  ({len(cls_cols)} classes)")

    # Full pairwise JSD matrix over windows.
    jsd_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            v = js_divergence(props[i], props[j])
            jsd_mat[i, j] = jsd_mat[j, i] = v

    # Split pairs by whether the two windows share an HMM state.
    within, across = [], []
    for i in range(N):
        for j in range(i + 1, N):
            (within if state_seq[i] == state_seq[j] else across).append(
                jsd_mat[i, j])
    within = np.array(within)
    across = np.array(across)

    print(f"Within-state pairs : n={len(within):4d}  "
          f"mean={within.mean():.4f}  var={within.var():.5f}")
    print(f"Across-state pairs : n={len(across):4d}  "
          f"mean={across.mean():.4f}  var={across.var():.5f}")

    # Permutation test of across-mean > within-mean: shuffle the pooled values
    # once per iteration and split, so the null preserves the pooled composition.
    obs_diff = across.mean() - within.mean()
    all_vals = np.concatenate([within, across])
    n_w = len(within)

    null = np.empty(args.n_perm)
    for b in range(args.n_perm):
        perm = rng.permutation(all_vals)
        null[b] = perm[n_w:].mean() - perm[:n_w].mean()

    p_val = float((null >= obs_diff).mean())
    print(f"Δ mean (across−within) = {obs_diff:+.4f}  "
          f"permutation p = {p_val:.4f}  (N={args.n_perm})")

    # Per-state within-state JSD values.
    per_state = {}
    for s in range(k):
        idx = np.where(state_seq == s)[0]
        vals = [jsd_mat[idx[a], idx[b]]
                for a in range(len(idx))
                for b in range(a + 1, len(idx))]
        per_state[s] = np.array(vals) if vals else np.array([np.nan])

    rows = [
        {"group": "within-state", "n": len(within),
         "mean_jsd": within.mean(), "var_jsd": within.var()},
        {"group": "across-state", "n": len(across),
         "mean_jsd": across.mean(), "var_jsd": across.var()},
    ]
    for s in range(k):
        v = per_state[s][~np.isnan(per_state[s])]
        rows.append({
            "group": f"state_{s}_within",
            "n": len(v),
            "mean_jsd": v.mean() if len(v) else np.nan,
            "var_jsd": v.var() if len(v) else np.nan,
        })
    csv_path = os.path.join(args.output_dir, "jsd_stability_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.5f")
    print(f"Saved: {csv_path}")

    np.save(os.path.join(args.output_dir, "jsd_stability_null.npy"), null)
    np.save(os.path.join(args.output_dir, "jsd_within.npy"), within)
    np.save(os.path.join(args.output_dir, "jsd_across.npy"), across)
    print(f"Saved: jsd_stability_null.npy  jsd_within.npy  jsd_across.npy")

    # Plot 1: violin comparison plus per-state mean bars.
    bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                  "#9467bd", "#8c564b", "#17becf", "#e377c2"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    colors = ["#2ca02c", "#d62728"]
    parts = ax.violinplot([within, across], positions=[0, 1],
                          showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.45)
    for key in ("cmedians", "cbars", "cmaxes", "cmins"):
        parts[key].set_color("black")
    jitter_rng = np.random.default_rng(1)
    for pos, grp, col in zip([0, 1], [within, across], colors):
        jit = jitter_rng.uniform(-0.07, 0.07, len(grp))
        ax.scatter(pos + jit, grp, color=col, alpha=0.4, s=10, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"Within-State\n($n={len(within)}$, $\\mu={within.mean():.3f}$)",
         f"Across-State\n($n={len(across)}$, $\\mu={across.mean():.3f}$)"],
        fontsize=11,
    )
    ax.set_ylabel("Pairwise Jensen\u2013Shannon Divergence", fontsize=12)
    ax.set_title("Within- vs. Across-State Pairwise JSD", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25, ls="--")
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    state_means = []
    state_labels = []
    for s in range(k):
        v = per_state[s][~np.isnan(per_state[s])]
        state_means.append(v.mean() if len(v) else np.nan)
        state_labels.append(f"$S_{{{s}}}$")
    ax2.bar(range(k), state_means,
            color=[bar_colors[s % len(bar_colors)] for s in range(k)],
            alpha=0.75, edgecolor="white")
    ax2.axhline(within.mean(), color="#2ca02c", ls="--", lw=1.5,
                label=f"Pooled within mean ({within.mean():.3f})")
    ax2.axhline(across.mean(), color="#d62728", ls="--", lw=1.5,
                label=f"Pooled across mean ({across.mean():.3f})")
    ax2.set_xticks(range(k))
    ax2.set_xticklabels(state_labels, fontsize=11)
    ax2.set_ylabel("Mean Pairwise JSD (Within State)", fontsize=12)
    ax2.set_title("Per-State Mean Within-State JSD", fontsize=12)
    ax2.legend(fontsize=10, framealpha=0.85)
    ax2.grid(True, axis="y", alpha=0.25, ls="--")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"JSD Stability: Within- vs. Across-State Window Pairs  "
        f"[$K={k}$, {spec.name}]",
        fontsize=14,
    )
    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "plot_jsd_stability.png")
    fig.savefig(plot_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")

    # Plot 2: permutation null with the observed difference of means.
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    pct95 = np.percentile(null, 95)
    ax.hist(null, bins=60, color="#94A3B8", edgecolor="white",
            linewidth=0.3, alpha=0.85, label="Permutation Null")
    ax.axvline(obs_diff, color="#DC2626", lw=2.5,
               label=f"Observed Δμ = {obs_diff:+.4f}  (p = {p_val:.4f})")
    ax.axvline(pct95, color="#1f2937", lw=1.2, linestyle="--", alpha=0.7,
               label=f"Null 95th pct = {pct95:.4f}")
    ax.set_xlabel(
        "Mean JSD (Across-State) \u2212 Mean JSD (Within-State)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Permutation Test: Across- vs. Within-State JSD  [$K={k}$, {spec.name}]",
        fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    null_plot_path = os.path.join(args.output_dir, "plot_jsd_stability_null.png")
    fig2.savefig(null_plot_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {null_plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()