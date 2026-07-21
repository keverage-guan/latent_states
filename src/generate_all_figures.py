#!/usr/bin/env python3
"""
generate_all_figures.py

Single script that regenerates every figure whose underlying data is saved to
disk, for a given dataset.  Outputs go to figures/<dataset>/ (k-independent) and
figures/<dataset>/k{k}/ (per-k).  All input/output paths are derived from the
dataset stem, so nothing is hardcoded to a particular label scheme.

Covers:
  Global (once)
    - window_distributions.png          — sample counts per window
    - window_stats.csv                  — matching summary table
    - cross_window_f1_heatmap.png       — train-vs-test macro F1 heatmap (col-centered)
    - hmm_model_selection.png           — BIC and LOO-CV vs k
    - hmm_ari_stability.png             — Viterbi ARI across inits vs k
    - hmm_loo_per_fold.png              — per-seed LOO curves
    - pca_sanity_before.png             — PC1/PC2 scatter of UNALIGNED weights
    - pca_sanity_after.png              — PC1/PC2 scatter after alignment
    - pca_component_selection.png       — scree plot with elbow-retained cutoff

  Per-k
    - k{k}/transition_matrix.png        — HMM transition probability heatmap
    - k{k}/state_strip.png              — Viterbi state sequence (scatter+step)
    - k{k}/state_timeline_seeds.png     — Viterbi decode on centroid
    - k{k}/timeline_with_classes.png    — state bar + stacked class proportions
    - k{k}/f1_vs_distance.png           — within/across F1 vs temporal lag
    - k{k}/distance_conditioned_null.png — permutation-test null histogram
    - k{k}/statepair_heatmap.png        — mean col-centered F1 per (train-state, test-state)
    - k{k}/f1_heatmap_with_states.png   — full col-centered F1 matrix sorted by HMM state
    - k{k}/eq_f1_vs_distance_both.png   — within/across F1 vs lag for both segmentations
    - k{k}/eq_direct_perm_test.png      — null distribution for (HMM gap - equal gap)
    - k{k}/eq_segmentation_comparison.png — window-by-window colour strip comparison

  The equal-segmentation figures are read from pre-computed outputs saved by
  check_equal_windows.py under --eq_dir/k{k}/.  Run check_equal_windows.py
  for each k before running this script.

Usage
-----
    # defaults derived from the dataset stem:
    python -m src.generate_all_figures --dataset fakeddit --k 11 \
        --output_dir figures/fakeddit \
        --hmm_dir    data/hmm_hmm/fakeddit \
        --perf_dir   data/hmm_perf/fakeddit \
        --wa_dir     data/hmm_within_across/fakeddit \
        --pca_npz    data/hmm_weights/fakeddit/weights_pca.npz \
        --manifest   data/splits/hmm_windows/fakeddit/fakeddit_windows_manifest.csv \
        --eq_dir     data/check_equal_windows/fakeddit \
        --dpi 300

    python -m src.generate_all_figures --dataset yelp --k 16 \
        --output_dir figures/yelp \
        --hmm_dir    data/hmm_hmm/yelp \
        --perf_dir   data/hmm_perf/yelp \
        --wa_dir     data/hmm_within_across/yelp \
        --pca_npz    data/hmm_weights/yelp/weights_pca.npz \
        --manifest   data/splits/hmm_windows/yelp/yelp_windows_manifest.csv \
        --eq_dir     data/check_equal_windows/yelp \
        --dpi 300
"""

import os
import re
import sys
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.datasets.registry import get_spec

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Shared constants ──────────────────────────────────────────────────────────

# Human-readable names/colours for the Fakeddit 6-way scheme (the default).
# Any other class count falls back to generic names + the tab10 palette.
CLASS_NAMES_DEFAULT = [
    "True", "Satire", "False Connection",
    "Imposter Content", "Manipulated Content", "Misleading Content",
]
CLASS_COLORS_DEFAULT = [
    "#009E73", "#56B4E9", "#E69F00",
    "#CC79A7", "#D55E00", "#0072B2",
]

TAB10         = plt.cm.tab10.colors
STATE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#17becf", "#e377c2",
]
COLORS_WA = {"within": "#2563EB", "across": "#DC2626"}
COLORS_EQ = {"hmm": "#1f77b4", "equal": "#ff7f0e"}


def _class_names_colors(n):
    """Return (names, colors) for n classes — the default scheme when it fits,
    otherwise generic labels cycled over tab10."""
    if n == len(CLASS_NAMES_DEFAULT):
        return CLASS_NAMES_DEFAULT, CLASS_COLORS_DEFAULT
    return ([f"Class {i}" for i in range(n)],
            [TAB10[i % len(TAB10)] for i in range(n)])


def _pfmt(p):
    """Format a p-value for a plot label, clamping tiny values.

    p = 0 (or anything below the 1/n_perm resolution) is reported as
    'p < 0.0001' rather than the misleading 'p = 0.0000'.
    """
    if p is None:
        return ""
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Regenerate all saved-data figures for the HMM analysis, "
                    "dataset-generalized."
    )
    p.add_argument("--dataset", required=True, choices=["fakeddit", "yelp"],
                   help="Which dataset spec to use (drives stem + default paths).")
    p.add_argument("--k", nargs="+", type=int, default=None,
               help="HMM state counts to plot. "
                    "Defaults to 2–12 for fakeddit, 2–18 for yelp.")
    p.add_argument("--output_dir", default=None,
                   help="Defaults to figures/<dataset>")
    p.add_argument("--hmm_dir",  default=None,
                   help="Contains <dataset>_decode_k*.npz. "
                        "Defaults to data/hmm_hmm/<dataset>")
    p.add_argument("--select_dir", default=None,
                   help="Contains hmm_scores.npz (model-selection output). "
                        "Defaults to data/hmm_select/<dataset>")
    p.add_argument("--perf_dir", default=None,
                   help="Contains cross_window_f1_colcentered.npz. "
                        "Defaults to data/hmm_perf/<dataset>")
    p.add_argument("--wa_dir",   default=None,
                   help="Parent of k{k}/ within-across output dirs. "
                        "Defaults to data/hmm_within_across/<dataset>")
    p.add_argument("--pca_npz",  default=None,
                   help="Defaults to data/hmm_weights/<dataset>/weights_pca.npz")
    p.add_argument("--manifest", default=None,
                   help="Window manifest CSV with start/end dates and cls_* columns. "
                        "Defaults to data/splits/hmm_windows/<dataset>/"
                        "<dataset>_windows_manifest.csv")
    p.add_argument("--splits_dir", default=None,
                   help="Dir with per-window split files (for window_distributions). "
                        "Defaults to data/splits/hmm_windows/<dataset>")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--eq_dir",   default=None,
                   help="Root dir written by check_equal_windows.py (sub-dirs k{k}/). "
                        "Defaults to data/check_equal_windows/<dataset>")
    p.add_argument("--corr_dir", default=None,
                   help="Root dir written by state_pair_correlation.py (sub-dirs k{k}/). "
                        "Defaults to data/state_pair_correlation/<dataset>")
    p.add_argument("--jsd_dir",  default=None,
                   help="Root of jsd_stability_analysis outputs (subdirs k{k}/). "
                        "Defaults to data/jsd_stability/<dataset>")
    p.add_argument("--partial_jsd_dir", default=None,
                   help="Root of partial_jsd_transfer.py outputs (contains k{k}/). "
                        "Defaults to data/partial_jsd/<dataset>")
    return p.parse_args()


def fill_defaults(args):
    """Derive per-dataset default paths for any argument left unset."""
    ds = args.dataset
    splits_root = os.path.join("data/splits/hmm_windows", ds)

    # Dataset-specific default k search window
    if args.k is None:
        _max_k = {"fakeddit": 12, "yelp": 18}.get(ds, 12)
        args.k = list(range(2, _max_k + 1))

    if args.output_dir is None:
        args.output_dir = os.path.join("figures", ds)
    if args.hmm_dir is None:
        args.hmm_dir = os.path.join("data/hmm_hmm", ds)
    if args.select_dir is None:
        args.select_dir = os.path.join("data/hmm_select", ds)
    if args.perf_dir is None:
        args.perf_dir = os.path.join("data/hmm_perf", ds)
    if args.wa_dir is None:
        args.wa_dir = os.path.join("data/hmm_within_across", ds)
    if args.pca_npz is None:
        args.pca_npz = os.path.join("data/hmm_weights", ds, "weights_pca.npz")
    if args.manifest is None:
        args.manifest = os.path.join(splits_root, f"{ds}_windows_manifest.csv")
    if args.splits_dir is None:
        args.splits_dir = splits_root
    if args.eq_dir is None:
        args.eq_dir = os.path.join("data/check_equal_windows", ds)
    if args.corr_dir is None:
        args.corr_dir = os.path.join("data/state_pair_correlation", ds)
    if args.jsd_dir is None:
        args.jsd_dir = os.path.join("data/jsd_stability", ds)
    if args.partial_jsd_dir is None:
        args.partial_jsd_dir = os.path.join("data/partial_jsd", ds)
    return args


# ── Utility ───────────────────────────────────────────────────────────────────

def savefig(fig, path, dpi=150):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def load_npz(path, label=""):
    if not os.path.exists(path):
        print(f"  [skip] {label or path} — file not found")
        return None
    return np.load(path, allow_pickle=True)


def banner(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def contiguous_runs(seq):
    """Return list of (state, start_idx, end_idx) for each run in seq."""
    runs, i = [], 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        runs.append((seq[i], i, j - 1))
        i = j
    return runs


def window_calendar_dates(window_ids, manifest_path):
    """
    Map window indices → (start_dates, end_dates) as numpy datetime64[D].
    Tries the manifest first; falls back to 60-day windows from 2013-01-01.
    """
    origin = np.datetime64("2013-01-01", "D")

    candidates = [
        manifest_path,
        os.path.join(os.path.dirname(manifest_path), "manifest.csv"),
    ]
    for mpath in candidates:
        if not os.path.exists(mpath):
            continue
        mdf = pd.read_csv(mpath)

        id_col = next(
            (c for c in mdf.columns
             if "window" in c.lower() and ("id" in c.lower() or "idx" in c.lower())),
            mdf.columns[0],
        )
        start_col = next((c for c in mdf.columns if "start" in c.lower()), None)
        end_col   = next((c for c in mdf.columns if "end"   in c.lower()), None)

        if start_col and end_col:
            mdf = mdf.set_index(id_col)
            starts, ends = [], []
            for wid in window_ids:
                if wid in mdf.index:
                    starts.append(np.datetime64(str(mdf.loc[wid, start_col])[:10], "D"))
                    ends.append(  np.datetime64(str(mdf.loc[wid, end_col  ])[:10], "D"))
                else:
                    starts.append(origin + np.timedelta64(int(wid) * 60,      "D"))
                    ends.append(  origin + np.timedelta64(int(wid) * 60 + 60, "D"))
            return np.array(starts), np.array(ends)

    # Fallback: synthetic 60-day windows
    starts = np.array([origin + np.timedelta64(int(w) * 60,      "D") for w in window_ids])
    ends   = np.array([origin + np.timedelta64(int(w) * 60 + 60, "D") for w in window_ids])
    return starts, ends


def date_to_mpl(dt64):
    return mdates.date2num(pd.Timestamp(dt64).to_pydatetime())


def _get_f1_matrix(d):
    """
    Return the column-centered cross-window F1 matrix if present, else the raw
    matrix.  Macro F1s are column-centered (per test-window mean removed) in the
    current pipeline, so prefer that array when the npz provides it.
    """
    if "f1_matrix_colcentered" in d.files:
        return d["f1_matrix_colcentered"]
    return d["f1_matrix"]


# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL FIGURES
# ═════════════════════════════════════════════════════════════════════════════

def _plot_perm_null(null, obs, p_one, output_path, *, title, xlabel, dpi):
    """Standard permutation-null histogram (null + observed line + 95th pct)."""
    pct95 = float(np.percentile(null, 95))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null, bins=60, color="#94A3B8", edgecolor="white",
            lw=0.3, alpha=0.85, label="Permutation null")
    ax.axvline(obs, color="#DC2626", lw=2.5,
               label=f"Observed = {obs:+.4f}  ({_pfmt(p_one)})")
    ax.axvline(pct95, color="black", lw=1.2, ls="--", alpha=0.7,
               label=f"Null 95th percentile = {pct95:.4f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Permutation Count")
    ax.set_title(title)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3, ls="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, output_path, dpi=dpi)


# ── 1. Window distributions ───────────────────────────────────────────────────

def figure_window_distributions(spec, splits_dir, out_dir, dpi):
    banner("Window distributions")

    wc = spec.window
    if wc is None:
        print(f"  [skip] window_distributions — dataset '{spec.name}' has no WindowConfig")
        return
    ts_col = wc.timestamp_col
    days   = wc.window_days

    def _load():
        if not os.path.isdir(splits_dir):
            return None
        files = (glob.glob(os.path.join(splits_dir, "*.tsv"))
                 + glob.glob(os.path.join(splits_dir, "*.jsonl")))
        frames = []
        for f in files:
            try:
                if f.endswith(".jsonl"):
                    df = pd.read_json(f, lines=True)
                else:
                    df = pd.read_csv(f, sep="\t", low_memory=False)
            except Exception:
                continue
            if ts_col in df.columns:
                keep = [c for c in (ts_col, spec.id_col) if c in df.columns]
                frames.append(df[keep])
        if not frames:
            return None

        data = pd.concat(frames, ignore_index=True)
        if spec.id_col in data.columns:
            data = data.drop_duplicates(subset=spec.id_col)

        if wc.timestamp_kind == "unix_s":
            secs = pd.to_numeric(data[ts_col], errors="coerce")
            data = data.assign(_secs=secs).dropna(subset=["_secs"])
            data["created_dt"] = pd.to_datetime(data["_secs"], unit="s", utc=True)
            data = data.drop(columns="_secs")
        else:  # "iso" and anything pandas can parse
            data["created_dt"] = pd.to_datetime(data[ts_col], errors="coerce", utc=True)
            data = data.dropna(subset=["created_dt"])

        return data.sort_values("created_dt").reset_index(drop=True)

    data = _load()
    if data is None or len(data) == 0:
        print(f"  [skip] window_distributions — no usable split files in {splits_dir}")
        return

    origin = data["created_dt"].min().floor("D")
    data["window_idx"] = ((data["created_dt"] - origin).dt.days // days).astype(int)

    wdf = (data.groupby("window_idx")
               .agg(count=("window_idx", "size"),
                    window_start=("created_dt", "min"))
               .reset_index())
    wdf["window_start"] = wdf["window_start"].dt.tz_localize(None)

    # Actual days in each window
    last_date = data["created_dt"].max().tz_localize(None)
    actual_days_list = []
    for idx in wdf["window_idx"]:
        win_end   = origin.tz_localize(None) + pd.Timedelta(days=(idx + 1) * days)
        win_end   = min(win_end, last_date)
        win_start = origin.tz_localize(None) + pd.Timedelta(days=idx * days)
        actual_days_list.append(max(1, (win_end - win_start).days))
    wdf["actual_days"] = actual_days_list

    ts    = wdf["window_start"]
    color = "#3B82F6"

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(ts, wdf["count"], width=pd.Timedelta(days=days * 0.85),
           color=color, alpha=0.85, linewidth=0.3, edgecolor="white")
    mean_val = wdf["count"].mean()
    ax.axhline(mean_val, color="black", lw=1.0, ls="--", alpha=0.6, zorder=5)
    ax.text(wdf["window_start"].iloc[-1], mean_val * 1.05,
            f"mean={mean_val:,.0f}", fontsize=8,
            va="bottom", ha="right", alpha=0.75)
    last_days = int(wdf["actual_days"].iloc[-1])
    note = f"  [last window = {last_days} days]" if last_days != days else ""
    ax.set_title(
        f"{spec.name} — Sample counts per {days}-day window  "
        f"(n_windows={len(wdf)}, total={wdf['count'].sum():,}){note}",
        fontsize=12, fontweight="bold")
    ax.set_ylabel("Samples per window", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(ts.min(), ts.max())
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "window_distributions.png"), dpi=dpi)

    # CSV
    c = wdf["count"]
    pd.DataFrame([{"Resolution": f"{days} days", "N windows": len(wdf),
                   "Total samples": int(c.sum()), "Mean/window": f"{c.mean():.0f}",
                   "Std": f"{c.std():.0f}", "Min": int(c.min()),
                   "Max": int(c.max())}]).to_csv(
        os.path.join(out_dir, "window_stats.csv"), index=False)
    print(f"    Saved: {os.path.join(out_dir, 'window_stats.csv')}")


# ── 2. Cross-window F1 heatmap ────────────────────────────────────────────────

def figure_cross_window_heatmap(perf_dir, out_dir, dpi):
    banner("Cross-window F1 heatmap")
    d = load_npz(os.path.join(perf_dir, "cross_window_f1_colcentered.npz"),
                 "cross_window_f1_colcentered.npz")
    if d is None:
        return

    f1_matrix = _get_f1_matrix(d)
    valid_ids = d["valid_ids"].tolist()
    n = len(valid_ids)

    plot_mat = f1_matrix.copy().astype(float)
    np.fill_diagonal(plot_mat, np.nan)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.4), max(5, n * 0.4)))
    im = ax.imshow(plot_mat, aspect="auto", cmap="RdYlGn",
                   vmin=np.nanmin(plot_mat), vmax=np.nanmax(plot_mat))
    plt.colorbar(im, ax=ax, label="Column-centered macro F1")
    tick_labels = [f"W{i:03d}" for i in valid_ids]
    ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Test window", fontsize=11)
    ax.set_ylabel("Train window", fontsize=11)
    ax.set_title("Cross-window column-centered macro F1", fontsize=13)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "cross_window_f1_heatmap.png"), dpi=dpi)


# ── 3. HMM model selection ────────────────────────────────────────────────────

def figure_model_selection(select_dir, out_dir, dpi, max_k=None):
    banner("HMM model selection")

    scores_path = None
    for candidate in [os.path.join(select_dir, "hmm_scores.npz"),
                      os.path.join(os.path.dirname(select_dir), "hmm_scores.npz")]:
        if os.path.exists(candidate):
            scores_path = candidate
            break
    if scores_path is None:
        print(f"  [skip] hmm_scores.npz not found under {select_dir}")
        return

    d = np.load(scores_path)
    k_arr    = d["k_range"]
    bic_best = d["bic_best"] if "bic_best" in d.files else None
    loo_mean = d["loo_mean"]
    loo_std  = d["loo_std"]
    ari_all  = d["ari_all"]
    loo_all  = d["loo_all"]

    # Clip to dataset-specific search window
    if max_k is not None:
        mask     = k_arr <= max_k
        k_arr    = k_arr[mask]
        if bic_best is not None:
            bic_best = bic_best[mask]
        loo_mean = loo_mean[mask]
        loo_std  = loo_std[mask]
        ari_all  = ari_all[mask]
        loo_all  = loo_all[mask] if loo_all.ndim == 1 else loo_all[mask, :]

    valid = np.isfinite(loo_mean)
    k_loo = k_arr[valid][np.argmax(loo_mean[valid])] if valid.any() else None

    # ── BIC + LOO-CV side by side ────────────────────────────────────────────
    n_panels = 2 if bic_best is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    ax_i = 0

    # BIC panel
    if bic_best is not None:
        ax = axes[ax_i]; ax_i += 1
        k_bic = k_arr[np.argmin(bic_best)]
        ax.plot(k_arr, bic_best, "o-", color="#2563EB", lw=2, ms=6,
                label="BIC (best init)")
        ax.axvline(k_bic, color="#2563EB", ls="--", alpha=0.6,
                   label=rf"$k = {k_bic}$ (min BIC)")
        ax.set_xlabel("Number of States ($k$)", fontsize=12)
        ax.set_ylabel("BIC", fontsize=12)
        ax.set_title("Bayesian Information Criterion", fontsize=12)
        ax.set_xticks(k_arr)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        print("    [info] bic_best not in hmm_scores.npz — BIC panel omitted")

    # LOO-CV panel
    ax = axes[ax_i]
    if valid.any():
        ax.plot(k_arr[valid], loo_mean[valid], "o-", color="#16A34A",
                lw=2, ms=6, label="Mean LOO-CV")
        ax.fill_between(k_arr[valid],
                        loo_mean[valid] - loo_std[valid],
                        loo_mean[valid] + loo_std[valid],
                        alpha=0.15, color="#16A34A")
    if k_loo is not None:
        ax.axvline(k_loo, color="#16A34A", ls="--", alpha=0.6,
                   label=rf"$k = {k_loo}$ (max LOO-CV)")
    ax.set_xlabel("Number of States ($k$)", fontsize=12)
    ax.set_ylabel("Mean Log-Likelihood per Observation", fontsize=12)
    ax.set_title("Leave-One-Out Cross-Validation", fontsize=12)
    ax.set_xticks(k_arr)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle("HMM State Selection", fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "hmm_model_selection.png"), dpi=dpi)

    # ── ARI stability ────────────────────────────────────────────────────────
    ari_mean_r = np.nanmean(ari_all, axis=1)
    ari_std_r  = np.nanstd(ari_all,  axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_arr, ari_mean_r, "o-", color="#EA580C", lw=2, ms=6)
    ax.fill_between(k_arr, ari_mean_r - ari_std_r, ari_mean_r + ari_std_r,
                    alpha=0.15, color="#EA580C")
    ax.set_xlabel("Number of States ($k$)", fontsize=12)
    ax.set_ylabel("Mean Pairwise ARI", fontsize=12)
    ax.set_title("Viterbi Decode Stability Across Random Initialisations", fontsize=12)
    ax.set_xticks(k_arr)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "hmm_ari_stability.png"), dpi=dpi)

    # ── LOO per-fold ─────────────────────────────────────────────────────────
    n_folds = loo_all.shape[1] if loo_all.ndim > 1 else 0
    if n_folds > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        for fold in range(n_folds):
            ax.plot(k_arr, loo_all[:, fold], "o--", lw=1, ms=4, alpha=0.6,
                    label=f"Fold {fold}")
        ax.plot(k_arr[valid], loo_mean[valid], "o-", color="black",
                lw=2.5, ms=6, label="Mean", zorder=5)
        ax.set_xlabel("Number of States ($k$)", fontsize=12)
        ax.set_ylabel("Held-Out Log-Likelihood per Observation", fontsize=12)
        ax.set_title("LOO-CV Log-Likelihood per Fold", fontsize=12)
        ax.set_xticks(k_arr)
        ax.legend(fontsize=8, ncol=min(n_folds, 5))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        savefig(fig, os.path.join(out_dir, "hmm_loo_per_fold.png"), dpi=dpi)


# ── 4. PCA sanity check ───────────────────────────────────────────────────────

def figure_pca_sanity(pca_npz, out_dir, dpi):
    banner("PCA sanity check")
    d = load_npz(pca_npz, "weights_pca.npz")
    if d is None:
        return

    Z_scaled   = d["Z_scaled"].astype(np.float32)
    window_ids = d["window_ids"].astype(int)
    seed_ids   = d["seed_ids"].astype(int)
    evr        = d["explained_variance_ratio"]                 # retained PCs
    evr_full   = d["evr_full"] if "evr_full" in d.files else evr  # all fitted PCs

    unique_wins  = np.unique(window_ids)
    unique_seeds = np.unique(seed_ids)
    n_wins  = len(unique_wins)
    MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "+"]
    win_to_idx  = {w: i for i, w in enumerate(unique_wins)}
    seed_to_idx = {s: i for i, s in enumerate(unique_seeds)}
    cmap_win = plt.cm.viridis

    def _scatter(Z, title, out_path, pc1_label="PC1", pc2_label="PC2"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for i in range(len(window_ids)):
            c = cmap_win(win_to_idx[window_ids[i]] / max(n_wins - 1, 1))
            m = MARKERS[seed_to_idx[seed_ids[i]] % len(MARKERS)]
            ax.scatter(Z[i, 0], Z[i, 1], color=c, marker=m, s=18, alpha=0.6)

        sm = plt.cm.ScalarMappable(cmap=cmap_win,
                                   norm=plt.Normalize(0, max(n_wins - 1, 1)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label="Window index", pad=0.02)
        cbar.ax.yaxis.labelpad = 8

        ax.set_xlabel(pc1_label, fontsize=16)
        ax.set_ylabel(pc2_label, fontsize=16)
        ax.set_title(title, fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label("Window index", fontsize=14)

        legend_els = [
            mlines.Line2D([], [], color="grey",
                          marker=MARKERS[seed_to_idx[s] % len(MARKERS)],
                          linestyle="None", markersize=7, alpha=0.8, label=f"{s}")
            for s in unique_seeds[:min(len(unique_seeds), 10)]
        ]
        ax.legend(handles=legend_els, fontsize=12, ncol=1,
                  loc="upper left", bbox_to_anchor=(1.32, 1.0),
                  framealpha=0.7, title="Seed", title_fontsize=14)
        fig.tight_layout()
        savefig(fig, out_path, dpi=dpi)

    # After alignment: Z_scaled with variance labels from evr
    pc1_var = evr[0] * 100 if len(evr) > 0 else 0
    pc2_var = evr[1] * 100 if len(evr) > 1 else 0
    _scatter(
        Z_scaled,
        "Principal Component Analysis of\nAligned Representations (Post-Alignment)",
        os.path.join(out_dir, "pca_sanity_after.png"),
        pc1_label=f"PC1 ({pc1_var:.1f}% var)",
        pc2_label=f"PC2 ({pc2_var:.1f}% var)",
    )

    # Before alignment: unaligned weights if available
    if "Z_before_scaled" in d:
        Z_un = d["Z_before_scaled"].astype(np.float32)
        _scatter(
            Z_un,
            "Principal Component Analysis\nof Raw Representations (Pre-Alignment)",
            os.path.join(out_dir, "pca_sanity_before.png"),
            pc1_label="PC1 (unaligned)",
            pc2_label="PC2 (unaligned)",
        )

    # Scree / component-selection plot (elbow-based selection)
    scree      = np.asarray(evr_full, dtype=float).ravel()[:50]   # cap at first 50 PCs
    n_show     = len(scree)
    n_retained = 3

    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(range(1, n_show + 1), scree * 100,
             "o-", color="#3B82F6", lw=2, ms=4, label="Individual EVR")
    if n_retained <= n_show:
        ax1.axvline(n_retained + 0.5, color="#16A34A", lw=2, ls="--",
                    label=f"Scree elbow: {n_retained} components retained")

    ax1.set_xlim(0.5, n_show + 0.5)
    ax1.set_xlabel("Principal Component", fontsize=14)
    ax1.set_ylabel("Explained Variance (%)", fontsize=14)
    ax1.set_title("PCA Component Selection (Scree Elbow)", fontsize=16)
    ax1.tick_params(axis="both", labelsize=12)
    ax1.legend(fontsize=12)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "pca_component_selection.png"), dpi=dpi)


# ═════════════════════════════════════════════════════════════════════════════
# JSD STABILITY FIGURES
#   Reads pre-computed outputs from jsd_stability_analysis.py under
#   --jsd_dir/k{k}/.  Regenerates with publication styling:
#     k{k}/plot_jsd_stability.png      — violin + per-state bar (side by side)
#     k{k}/plot_jsd_stability_null.png — permutation null + observed Δμ
# ═════════════════════════════════════════════════════════════════════════════

def figure_jsd_stability(k, jsd_dir, k_out, dpi, state_seq=None):
    banner(f"  JSD stability figures  (k={k})")

    src_dir     = os.path.join(jsd_dir, f"k{k}")
    csv_path    = os.path.join(src_dir, "jsd_stability_summary.csv")
    null_path   = os.path.join(src_dir, "jsd_stability_null.npy")
    within_path = os.path.join(src_dir, "jsd_within.npy")
    across_path = os.path.join(src_dir, "jsd_across.npy")

    if not os.path.exists(csv_path):
        print(f"    [skip] all jsd_stability figures — {csv_path} not found "
              f"(re-run jsd_stability_analysis.py)")
        return

    df = pd.read_csv(csv_path)
    within_mean = float(df.loc[df["group"] == "within-state", "mean_jsd"].iloc[0])
    across_mean = float(df.loc[df["group"] == "across-state", "mean_jsd"].iloc[0])

    state_rows = df[df["group"].str.match(r"^state_\d+_within$")].copy()
    state_rows["native_id"] = (
        state_rows["group"].str.extract(r"state_(\d+)_within")[0].astype(int)
    )

    # Build chronological relabeling (native HMM id → 0-based chron index)
    if state_seq is not None:
        relabel = {}
        for s in state_seq:
            if s not in relabel:
                relabel[s] = len(relabel)
        # States present in the JSD summary but never visited in the decode
        # (possible under a left-to-right topology) get chron ids appended after
        # the visited ones, in native order — so the map never yields NaN.
        for nid in sorted(state_rows["native_id"].unique()):
            if nid not in relabel:
                relabel[nid] = len(relabel)
        state_rows["chron_id"] = state_rows["native_id"].map(relabel).astype(int)
        state_rows = state_rows.sort_values("chron_id").reset_index(drop=True)
        x_labels = [f"$S_{{{int(row.chron_id)}}}$"
                    for _, row in state_rows.iterrows()]
    else:
        state_rows = state_rows.sort_values("native_id").reset_index(drop=True)
        x_labels = [f"$S_{{{int(row.native_id)}}}$"
                    for _, row in state_rows.iterrows()]

    k_inferred  = len(state_rows)
    state_means = state_rows["mean_jsd"].values.astype(float)

    bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                  "#9467bd", "#8c564b", "#17becf", "#e377c2"]

    # ── Side-by-side: violin (if raw arrays available) or bar + per-state bars
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]

    if os.path.exists(within_path) and os.path.exists(across_path):
        within = np.load(within_path)
        across = np.load(across_path)
        colors = ["#2ca02c", "#d62728"]
        parts = ax.violinplot([within, across], positions=[0, 1],
                              showmedians=True, showextrema=True)
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col); pc.set_alpha(0.45)
        for key in ("cmedians", "cbars", "cmaxes", "cmins"):
            parts[key].set_color("black")
        jitter_rng = np.random.default_rng(1)
        for pos, grp, col in zip([0, 1], [within, across], colors):
            jit = jitter_rng.uniform(-0.07, 0.07, len(grp))
            ax.scatter(pos + jit, grp, color=col, alpha=0.4, s=10, zorder=3)
        n_w, n_a = len(within), len(across)
    else:
        ax.bar([0, 1], [within_mean, across_mean],
               color=["#2ca02c", "#d62728"], alpha=0.75, edgecolor="white")
        n_w = int(df.loc[df["group"] == "within-state", "n"].iloc[0])
        n_a = int(df.loc[df["group"] == "across-state", "n"].iloc[0])

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"Within-State\n($n={n_w}$, $\\mu={within_mean:.3f}$)",
         f"Across-State\n($n={n_a}$, $\\mu={across_mean:.3f}$)"],
        fontsize=11,
    )
    ax.set_ylabel("Pairwise Jensen\u2013Shannon Divergence", fontsize=12)
    ax.set_title("Within- vs. Across-State Pairwise JSD", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25, ls="--")
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    ax2.bar(range(k_inferred), state_means,
            color=[bar_colors[s % len(bar_colors)] for s in range(k_inferred)],
            alpha=0.75, edgecolor="white")
    ax2.axhline(within_mean, color="#2ca02c", ls="--", lw=1.5,
                label=f"Pooled within mean ({within_mean:.3f})")
    ax2.axhline(across_mean, color="#d62728", ls="--", lw=1.5,
                label=f"Pooled across mean ({across_mean:.3f})")
    ax2.set_xticks(range(k_inferred))
    ax2.set_xticklabels(x_labels, fontsize=10)
    ax2.set_ylabel("Mean Pairwise JSD (Within State)", fontsize=12)
    ax2.set_title("Per-State Mean Within-State JSD", fontsize=12)
    ax2.legend(fontsize=9, framealpha=0.85)
    ax2.grid(True, axis="y", alpha=0.25, ls="--")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"JSD Stability: Within- vs. Across-State Window Pairs ($K={k}$)",
        fontsize=14,
    )
    fig.tight_layout()
    savefig(fig, os.path.join(k_out, "plot_jsd_stability.png"), dpi=dpi)

    # ── Null distribution + observed Δμ ───────────────────────────────────────
    if not os.path.exists(null_path):
        print(f"    [skip] plot_jsd_stability_null — {null_path} not found "
              f"(re-run jsd_stability_analysis.py)")
        return

    null     = np.load(null_path)
    obs_diff = across_mean - within_mean
    p_val    = float((null >= obs_diff).mean())
    pct95    = np.percentile(null, 95)

    fig2, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(null, bins=60, color="#94A3B8", edgecolor="white",
            linewidth=0.3, alpha=0.85, label="Permutation Null")
    ax.axvline(obs_diff, color="#DC2626", lw=2.5,
               label=f"Observed mean difference = {obs_diff:+.4f}  ({_pfmt(p_val)})")
    ax.axvline(pct95, color="#1f2937", lw=1.2, linestyle="--", alpha=0.7,
               label=f"Null 95th Pct = {pct95:.4f}")
    ax.set_xlabel("Mean JSD (Across-State) \u2212 Mean JSD (Within-State)", fontsize=13)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Permutation Test: Across- vs. Within-State JSD ($K={k}$)",
        fontsize=13,
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    savefig(fig2, os.path.join(k_out, "plot_jsd_stability_null.png"), dpi=dpi)


def figure_partial_jsd(k, partial_jsd_dir, k_out, dpi):
    """
    Re-render the partial-JSD figures from pre-computed .npz output
    of partial_jsd_transfer.py.
    """
    banner(f"Partial-JSD transfer figures  (k={k})")
    k_in = os.path.join(partial_jsd_dir, f"k{k}")
    os.makedirs(k_out, exist_ok=True)

    null_path = os.path.join(k_in, "partial_jsd_null.npz")
    if not os.path.exists(null_path):
        print(f"    [skip] partial_jsd figures — {null_path} not found")
        return

    d = np.load(null_path)

    # ── Plot 1: JSD+lag-residualised F1 by state membership ──────────────
    pair_path = os.path.join(k_in, "window_pair_table.csv")
    if os.path.exists(pair_path):
        df = pd.read_csv(pair_path)
        # Reconstruct residuals from saved betas
        beta = d["beta_full"]          # [intercept, jsd, lag, same_state]
        X = np.column_stack([
            np.ones(len(df)),
            df["jsd"].values,
            df["lag"].values,
            df["same_state"].values,
        ])
        resid = df["f1"].values - X[:, :3] @ beta[:3]  # remove JSD+lag, keep state
        same_state = df["same_state"].values.astype(int)

        from src.analyses.partial_jsd_transfer import plot_partial_residual
        plot_partial_residual(
            resid, same_state,
            os.path.join(k_out, "partial_jsd_residual.png"))
    else:
        print(f"    [skip] partial_jsd_residual — {pair_path} not found")

    # ── Plot 2: Freedman-Lane null histogram ─────────────────────────────
    from src.analyses.partial_jsd_transfer import plot_indicator_null
    plot_indicator_null(
        d["fl_null"], float(d["fl_obs"]), float(d["fl_p_one"]),
        os.path.join(k_out, "partial_jsd_indicator_null.png"))

    # ── Plot 3: two-stage residual null (free label shuffle) ──────────────
    if "resid_free_null" in d.files:
        free_null = d["resid_free_null"]
        free_obs  = float(d["resid_free_obs"])
        free_p    = float((free_null >= free_obs).mean())
        _plot_perm_null(
            free_null, free_obs, free_p,
            os.path.join(k_out, "partial_jsd_residual_free_null.png"),
            title="Two-Stage Residual Test: Within- vs. Across-State\n"
                  "(Free Label Shuffle, JSD and Lag Removed)",
            xlabel="Within \u2212 Across Mean F1 Residual",
            dpi=dpi)
    else:
        print("    [skip] partial_jsd_residual_free_null — key not in npz")

    # ── Plot 4: two-stage residual null (lag-stratified, harmonic-weighted) ─
    if "resid_lag_null" in d.files:
        lag_null = d["resid_lag_null"]
        lag_obs  = float(d["resid_lag_obs"])
        lag_p    = float((lag_null >= lag_obs).mean())
        _plot_perm_null(
            lag_null, lag_obs, lag_p,
            os.path.join(k_out, "partial_jsd_residual_lagstratified_null.png"),
            title="Two-Stage Residual Test: Distance-Conditioned\n"
                  "(Lag-Stratified Shuffle, JSD and Lag Removed)",
            xlabel="Distance-Conditioned Within \u2212 Across F1 Residual Gap",
            dpi=dpi)
    else:
        print("    [skip] partial_jsd_residual_lagstratified_null — key not in npz")


# ═════════════════════════════════════════════════════════════════════════════
# PER-K HELPER FIGURES
# ═════════════════════════════════════════════════════════════════════════════

def _transition_matrix(trans_mat, k, out_path, dpi):
    fig, ax = plt.subplots(figsize=(max(5, k * 0.9), max(4, k * 0.8)))
    im = ax.imshow(trans_mat, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Transition probability")
    labels = [f"State {s}" for s in range(k)]
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("To state", fontsize=11)
    ax.set_ylabel("From state", fontsize=11)
    ax.set_title(f"HMM Transition Matrix (k={k})", fontsize=12)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{trans_mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if trans_mat[i,j] > 0.6 else "black")
    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)


def _state_strip(state_seq, start_dates, end_dates, k, out_path, dpi):
    # Relabel states in order of first chronological appearance
    relabel = {}
    next_id = 0
    for s in state_seq:
        if s not in relabel:
            relabel[s] = next_id
            next_id += 1
    state_seq = np.array([relabel[s] for s in state_seq])
    k_plot    = next_id

    n_win     = len(state_seq)
    x_vals    = np.arange(n_win)
    colors    = [TAB10[s % 10] for s in state_seq]
    mid_dates = [s + (e - s) / 2 for s, e in zip(start_dates, end_dates)]

    fig, ax = plt.subplots(figsize=(5, 2.5))

    # Filled squares on a grid, one per window, colored by state
    for x, s, c in zip(x_vals, state_seq, colors):
        ax.add_patch(Rectangle(
            (x - 0.5, s - 0.5), 1.0, 1.0,
            facecolor=c, edgecolor="white", linewidth=0.5, zorder=3,
        ))

    for idx in np.where(np.diff(state_seq) != 0)[0]:
        ax.axvline(idx + 0.5, color="black", linewidth=0.7,
                   linestyle="--", alpha=0.6)

    tick_step = max(1, n_win // 8)
    tick_idxs = list(range(0, n_win, tick_step))
    ax.set_xticks(tick_idxs)
    ax.set_xticklabels(
        [pd.Timestamp(mid_dates[i]).strftime("%b %Y") for i in tick_idxs],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_yticks([])
    ax.set_ylabel("HMM State", fontsize=11)
    ax.set_xlabel("Window (chronological)", fontsize=11)
    ax.set_title(f"Viterbi State Sequence on Seed Centroid  (k={k})", fontsize=12)
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_xlim(-0.5, n_win - 0.5)
    ax.set_ylim(-0.5, k_plot - 0.5)

    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)


def _state_timeline_seeds(dec, start_dates, end_dates, k, out_path, dpi):
    state_seq    = dec["state_seq"].astype(int)
    state_matrix = dec["state_matrix"] if "state_matrix" in dec else None
    seeds_dec    = dec["seed_ids_decoded"].tolist() if "seed_ids_decoded" in dec else []
    n_win        = len(state_seq)

    per_seed = {}
    if state_matrix is not None:
        for idx, s in enumerate(seeds_dec):
            per_seed[s] = state_matrix[idx]

    if not per_seed:
        print(f"    [info] state_timeline_seeds: no per-seed decode data in npz "
              f"(state_matrix/seed_ids_decoded absent) — showing centroid decode only")

    valid_seeds = list(per_seed.keys())
    n_rows      = len(valid_seeds) + 1

    fig, axes = plt.subplots(n_rows, 1,
                              figsize=(14, max(4, n_rows * 1.1)), sharex=True)
    if n_rows == 1:
        axes = [axes]

    boundary_nums = []
    for idx in np.where(np.diff(state_seq) != 0)[0]:
        boundary_nums.append(date_to_mpl(end_dates[idx]))

    def draw_row(ax, seq, title):
        for i in range(n_win):
            left  = date_to_mpl(start_dates[i])
            right = date_to_mpl(end_dates[i])
            ax.barh(0, right - left, left=left, height=0.8,
                    color=TAB10[seq[i] % 10], alpha=0.85, linewidth=0)
        for bn in boundary_nums:
            ax.axvline(bn, color="black", lw=1.2, ls="--", alpha=0.7, zorder=5)
        ax.set_yticks([0]); ax.set_yticklabels([title], fontsize=9)
        ax.set_ylim(-0.6, 0.6); ax.xaxis_date()
        ax.grid(axis="x", alpha=0.25, ls=":")

    for ax, s in zip(axes[:-1], valid_seeds):
        draw_row(ax, per_seed[s], f"seed {s}")
    draw_row(axes[-1], state_seq, "Viterbi (centroid)")
    axes[-1].set_facecolor("#F0F4FF")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate(rotation=30, ha="right")

    legend_patches = [mpatches.Patch(color=TAB10[s % 10], label=f"State {s}")
                      for s in range(k)]
    axes[0].legend(handles=legend_patches, loc="upper right", fontsize=8, ncol=k)
    fig.suptitle(f"HMM State Timeline — Viterbi on per-window centroids (k={k})",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)


def _timeline_with_classes(dec, manifest_path, k, out_path, dpi):
    """State colour bar (top) + stacked class proportions (bottom).

    The number of classes is inferred from the cls_* columns in the manifest,
    so this works for any label scheme (6-way Fakeddit, 5-way Yelp, ...).
    """
    state_seq  = dec["state_seq"].astype(int)
    window_ids = dec["window_ids"].astype(int)

    def _has_cls(cols):
        return any(re.fullmatch(r"cls_\d+", str(c)) for c in cols)

    mdf = None
    for mpath in [manifest_path,
                  os.path.join(os.path.dirname(manifest_path), "manifest.csv")]:
        if not os.path.exists(mpath):
            continue
        _m = pd.read_csv(mpath)
        if _has_cls(_m.columns):
            mdf = _m
            break

    if mdf is None:
        print("    [skip] timeline_with_classes — manifest without cls_* columns not found")
        return

    cls_cols = sorted([c for c in mdf.columns if re.fullmatch(r"cls_\d+", str(c))],
                      key=lambda c: int(str(c).split("_")[1]))
    n_cls = len(cls_cols)
    class_names, class_colors = _class_names_colors(n_cls)

    id_col    = next((c for c in mdf.columns if "window" in c.lower()), mdf.columns[0])
    start_col = next((c for c in mdf.columns if "start" in c.lower()), None)
    mdf       = mdf.set_index(id_col)
    origin    = np.datetime64("2013-01-01", "D")

    starts, props = [], []
    for wid in window_ids:
        if start_col and wid in mdf.index:
            row = mdf.loc[wid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            starts.append(pd.Timestamp(str(row[start_col])[:10]))
        else:
            starts.append(pd.Timestamp(origin + np.timedelta64(int(wid) * 60, "D")))

        if wid in mdf.index:
            row = mdf.loc[wid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            cls_vals = []
            for c in cls_cols:
                try:
                    v = row[c]
                    v = 0.0 if pd.isna(v) else float(v)
                except (TypeError, ValueError):
                    v = 0.0
                cls_vals.append(max(v, 0.0))
            total = sum(cls_vals)
            if total > 0:
                props.append([v / total for v in cls_vals])
            else:
                props.append([0.0] * n_cls)
        else:
            props.append([0.0] * n_cls)

    props_arr = np.array(props)
    N         = len(window_ids)
    runs      = contiguous_runs(state_seq)

    seen_order = {}
    for state, _, _ in runs:
        if state not in seen_order:
            seen_order[state] = len(seen_order)

    def chron_label(state):
        return seen_order[state]

    mpl_starts = [date_to_mpl(s) for s in starts]
    if N > 1:
        gaps = [mpl_starts[i + 1] - mpl_starts[i] for i in range(min(N - 1, 20))]
        window_width = float(np.median(gaps))
    else:
        window_width = 60.0

    def run_right(e_idx):
        if e_idx + 1 < N:
            return mpl_starts[e_idx + 1]
        return mpl_starts[-1] + window_width

    fig, (ax_bar, ax_stack) = plt.subplots(
        2, 1, figsize=(10, 4),
        gridspec_kw={"height_ratios": [1, 9]},
        sharex=False,
    )

    # ── Top panel: state colour bar ─────────────────────────────────────────
    for state, s_idx, e_idx in runs:
        left  = mpl_starts[s_idx]
        right = run_right(e_idx)
        label = chron_label(state)
        ax_bar.barh(0, right - left, left=left, height=1,
                    color=plt.cm.tab10.colors[label % 10], alpha=0.9, linewidth=0)
        mid = (left + right) / 2
        ax_bar.text(mid, 0, f"{label}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white")
    ax_bar.set_yticks([]); ax_bar.set_xticks([])
    ax_bar.set_title(
        f"HMM Latent State Sequence and Class Composition ($K={k}$)",
        fontsize=16,
    )
    ax_bar.set_xlim(mpl_starts[0], mpl_starts[-1] + window_width)
    ax_bar.set_ylim(-0.6, 0.6)

    # ── Bottom panel: stacked class proportions ──────────────────────────────
    x_dates = [mdates.date2num(s.to_pydatetime()) for s in starts]

    x_dates_ext = x_dates + [x_dates[-1] + window_width]
    props_ext   = np.vstack([props_arr, props_arr[-1:]])

    order  = np.argsort(props_arr.mean(axis=0))[::-1]
    bottom = np.zeros(len(x_dates_ext))
    for ci in order:
        y = props_ext[:, ci]
        ax_stack.fill_between(x_dates_ext, bottom, bottom + y,
                              color=class_colors[ci], alpha=0.85,
                              label=class_names[ci], step=None)
        ax_stack.plot(x_dates_ext, bottom + y, color="white", lw=0.3, alpha=0.5)
        bottom += y

    for _, _, e_idx in runs[:-1]:
        if e_idx + 1 < N:
            ax_stack.axvline(mpl_starts[e_idx + 1],
                             color="black", lw=1.0, ls="--", alpha=0.5, zorder=5)

    ax_stack.set_xlim(x_dates_ext[0], x_dates_ext[-1])
    ax_stack.set_ylim(0, 1)
    ax_stack.set_ylabel("Class Proportion", fontsize=14)
    ax_stack.set_xlabel("Date", fontsize=14)
    ax_stack.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_stack.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax_stack.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=12)

    ax_stack.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax_stack.spines[["top", "right"]].set_visible(False)

    class_patches = [mpatches.Patch(color=class_colors[ci], label=class_names[ci])
                     for ci in order]
    ax_stack.legend(
        handles=class_patches,
        loc="upper right",
        bbox_to_anchor=(-0.12, 1.0),
        fontsize=11,
        ncol=1,
        framealpha=0.85,
    )

    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)


def _f1_vs_distance(summary_df, out_path, dpi):
    df = summary_df[summary_df["distance"] != "pooled"].copy()
    df["distance"] = df["distance"].astype(int)
    df = df.sort_values("distance")

    fig, ax = plt.subplots(figsize=(5, 3))
    for kind, color, label in [
        ("within", COLORS_WA["within"], "Within-state"),
        ("across", COLORS_WA["across"], "Across-state"),
    ]:
        m  = df[f"{kind}_mean"].values
        lo = df[f"{kind}_lo"].values
        hi = df[f"{kind}_hi"].values
        d  = df["distance"].values
        ax.plot(d, m, color=color, lw=2, marker="o", ms=4, label=label)
        ax.fill_between(d, lo, hi, color=color, alpha=0.15)

    ax.set_xlabel("Temporal Distance |i − j| (windows)", fontsize=12)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title("Cross-Window F1 vs. Temporal Distance by State Membership",
                 fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, ls="--")
    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)


def _distance_conditioned_null(null_npz_path, out_path, dpi):
    d = load_npz(null_npz_path, "distance_conditioned_null.npz")
    if d is None:
        return
    null_gaps    = d["null_gaps"]
    observed_gap = float(d["observed_gap"])
    p_value      = float(np.mean(null_gaps >= observed_gap))
    pct95        = np.percentile(null_gaps, 95)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_gaps, bins=60, color="#94A3B8", edgecolor="white",
            lw=0.3, alpha=0.85, label="Label-shuffle null")
    ax.axvline(observed_gap, color="#DC2626", lw=2.5,
               label=f"Observed = {observed_gap:.4f}  ({_pfmt(p_value)})")
    ax.axvline(pct95, color="black", lw=1.2, linestyle="--", alpha=0.7,
               label=f"95th pct = {pct95:.4f}")
    ax.set_xlabel("Within − Across F1 Gap", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distance-Conditioned Permutation Test", fontsize=13)
    ax.legend(fontsize=11, loc='upper right'); ax.grid(True, alpha=0.3, ls="--")
    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)


def _statepair_heatmap(f1_matrix, valid_ids_f1, state_seq, window_ids_dec,
                       k, out_path, dpi):
    wid_to_state = dict(zip(window_ids_dec, state_seq))
    valid_arr    = np.array(valid_ids_f1)

    sp_mat = np.full((k, k), np.nan)
    for si in range(k):
        for sj in range(k):
            vals = [f1_matrix[ri, ci]
                    for ri, wi in enumerate(valid_arr)
                    if wid_to_state.get(wi) == si
                    for ci, wj in enumerate(valid_arr)
                    if wid_to_state.get(wj) == sj and not np.isnan(f1_matrix[ri, ci])]
            if vals:
                sp_mat[si, sj] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(max(5, k), max(4, k - 1)))
    im = ax.imshow(sp_mat, aspect="auto", cmap="RdYlGn",
                   vmin=np.nanmin(sp_mat), vmax=np.nanmax(sp_mat))
    plt.colorbar(im, ax=ax, label="Mean column-centered macro F1")
    labels = [f"State {s}" for s in range(k)]
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Test-window state", fontsize=11)
    ax.set_ylabel("Train-window state", fontsize=11)
    ax.set_title(f"Mean cross-window col-centered F1 by state pair (k={k})",
                 fontsize=12)
    for si in range(k):
        for sj in range(k):
            v = sp_mat[si, sj]
            if not np.isnan(v):
                ax.text(sj, si, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, color="black" if 0.3 < v < 0.85 else "white")
    for s in range(k):
        rect = mpatches.FancyBboxPatch((s - 0.5, s - 0.5), 1, 1,
                                       boxstyle="square,pad=0", lw=2.5,
                                       edgecolor="navy", facecolor="none")
        ax.add_patch(rect)
    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)


def _f1_heatmap_with_states(f1_matrix, valid_ids_f1, state_seq, window_ids_dec,
                             k, out_path, dpi):
    """Full F1 heatmap in natural window order, with state boundaries marked."""
    wid_to_state = dict(zip(window_ids_dec, state_seq))
    valid_arr    = np.array(valid_ids_f1)
    N            = len(valid_arr)
    states_valid = np.array([wid_to_state.get(w, -1) for w in valid_arr])

    plot_mat = f1_matrix.copy().astype(float)
    np.fill_diagonal(plot_mat, np.nan)

    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(plot_mat, aspect="auto", cmap="RdYlGn",
                   vmin=np.nanmin(plot_mat), vmax=np.nanmax(plot_mat),
                   extent=[-0.5, N - 0.5, N - 0.5, -0.5])

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("Column-centered macro F1", fontsize=14)

    boundaries = [i - 0.5 for i in range(1, N)
                  if states_valid[i] != states_valid[i - 1]
                  and states_valid[i] >= 0 and states_valid[i - 1] >= 0]
    for b in boundaries:
        for spine in (
            dict(color="white",   lw=2.5, ls="-", alpha=1.0, zorder=3),
            dict(color="#1a1a1a", lw=1.2, ls="-", alpha=1.0, zorder=4),
        ):
            ax.axvline(b, **spine)
            ax.axhline(b, **spine)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Test Window", fontsize=14)
    ax.set_ylabel("Train Window", fontsize=14)
    ax.set_title("Cross-Window Column-Centered Macro F1", fontsize=16)

    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)


# ═════════════════════════════════════════════════════════════════════════════
# EQUAL-SEGMENTATION FIGURES  (reads pre-computed data from check_equal_windows.py)
# ═════════════════════════════════════════════════════════════════════════════

def figure_equal_segmentation(k, eq_dir, k_out, dpi):
    """
    Re-plot the equal-segmentation figures from the files saved by
    check_equal_windows.py under eq_dir/k{k}/.

    Expected files
    --------------
    comparison_summary.csv       — rows: HMM, equal_duration, HMM_minus_equal
    distance_stratified_both.csv — per-lag within/across means, method column
    null_diffs.npy               — null distribution for direct perm test
    decode_snapshot.npz          — arrays: hmm_labels, equal_labels, N
    """
    banner(f"  Equal-segmentation figures  (k={k})")

    src = os.path.join(eq_dir, f"k{k}")

    summary_path = os.path.join(src, "comparison_summary.csv")
    strat_path   = os.path.join(src, "distance_stratified_both.csv")
    null_path    = os.path.join(src, "null_diffs.npy")
    snap_path    = os.path.join(src, "decode_snapshot.npz")

    if not os.path.exists(summary_path):
        print(f"    [skip] all eq figures — {summary_path} not found")
        return

    summary = pd.read_csv(summary_path)

    # ── 1. F1 vs distance (both segmentations) ───────────────────────────────
    if os.path.exists(strat_path):
        try:
            strat     = pd.read_csv(strat_path)
            strat_hmm = strat[strat["method"] == "HMM"].copy()
            strat_eq  = strat[strat["method"] == "equal_duration"].copy()

            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            for ax, s, label, color in [
                (axes[0], strat_hmm, "HMM Segmentation",           COLORS_EQ["hmm"]),
                (axes[1], strat_eq,  "Equal-Duration Segmentation", COLORS_EQ["equal"]),
            ]:
                has_w  = set(s.loc[s["n_within"] > 0, "distance"])
                has_a  = set(s.loc[s["n_across"] > 0, "distance"])
                common = sorted(has_w & has_a)
                if not common:
                    ax.set_title(label + " (no overlapping distances)")
                    continue
                s = s[s["distance"].isin(common)].sort_values("distance")
                d = s["distance"].values
                ax.plot(d, s["within_mean"].values, color=color, lw=2,
                        marker="o", ms=4, label="Within-group")
                ax.plot(d, s["across_mean"].values, color=color, lw=2,
                        marker="s", ms=4, linestyle="--", label="Across-group",
                        alpha=0.7)
                ax.fill_between(d, s["within_mean"].values, s["across_mean"].values,
                                alpha=0.10, color=color)
                ax.set_title(label, fontsize=12)
                ax.set_xlabel("Temporal Distance |i − j| (Windows)", fontsize=12)
                ax.set_xlim(min(common) - 0.5, max(common) + 0.5)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, linestyle="--")
            axes[0].set_ylabel("Macro F1", fontsize=12)
            fig.suptitle(
                "Within- vs. Across-Group Macro F1 by Temporal Lag",
                fontsize=14)
            fig.tight_layout()
            savefig(fig, os.path.join(k_out, "eq_f1_vs_distance_both.png"), dpi=dpi)
        except Exception as e:
            print(f"    [skip] eq_f1_vs_distance_both — {e}")
    else:
        print(f"    [skip] eq_f1_vs_distance_both — {strat_path} not found")

    # ── 2. Direct permutation test (HMM gap − equal gap) ─────────────────────
    if os.path.exists(null_path):
        try:
            null_diffs   = np.load(null_path)
            hmm_row      = summary[summary["method"] == "HMM"].iloc[0]
            eq_row       = summary[summary["method"] == "equal_duration"].iloc[0]
            obs_diff     = float(hmm_row["gap"]) - float(eq_row["gap"])
            p_direct_row = summary[summary["method"].str.startswith("HMM_minus")].iloc[0]
            p_direct     = float(p_direct_row["direct_perm_p"])

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(null_diffs, bins=60, color="#94A3B8", edgecolor="white",
                    linewidth=0.3, alpha=0.85, label="HMM label-shuffle null")
            ax.axvline(obs_diff, color="#DC2626", lw=2.5,
                       label=f"Observed gap = {obs_diff:+.4f}   ({_pfmt(p_direct)})")
            ax.set_xlabel("HMM Pooled Gap \u2212 Equal-Duration Pooled Gap", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(
                "Direct Permutation Test: HMM vs. Equal-Duration Generalisation Gap",
                fontsize=12,
            )
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, linestyle="--")
            fig.tight_layout()
            savefig(fig, os.path.join(k_out, "eq_direct_perm_test.png"), dpi=dpi)
        except Exception as e:
            print(f"    [skip] eq_direct_perm_test — {e}")
    else:
        print(f"    [skip] eq_direct_perm_test — {null_path} not found")

    # ── 3. Segmentation comparison strip ─────────────────────────────────────
    if os.path.exists(snap_path):
        try:
            snap         = np.load(snap_path, allow_pickle=True)
            hmm_labels   = snap["hmm_labels"].astype(int)
            equal_labels = snap["equal_labels"].astype(int)
            N            = int(snap["N"])

            n_groups = max(hmm_labels.max(), equal_labels.max()) + 1
            colors   = [TAB10[i % 10] for i in range(n_groups)]

            fig, axes = plt.subplots(2, 1, figsize=(14, 3),
                                     gridspec_kw={"hspace": 0.6})
            for ax, lbl_arr, title in [
                (axes[0], hmm_labels,   "HMM Segmentation"),
                (axes[1], equal_labels, "Equal-Duration Segmentation"),
            ]:
                for i, lbl in enumerate(lbl_arr):
                    ax.bar(i, 1, color=colors[lbl], edgecolor="white", linewidth=0.3)
                ax.set_xlim(-0.5, N - 0.5)
                ax.set_ylim(0, 1)
                ax.set_yticks([])
                ax.set_xlabel("Window Index", fontsize=9)
                ax.set_title(title, fontsize=10)
                patches = [mpatches.Patch(color=colors[g], label=f"G{g}")
                           for g in range(n_groups)]
                ax.legend(handles=patches, loc="upper right", fontsize=7,
                          ncol=n_groups, framealpha=0.7)
            fig.suptitle("Window Group Assignments: HMM vs. Equal-Duration Segmentation",
                         fontsize=11)
            savefig(fig, os.path.join(k_out, "eq_segmentation_comparison.png"), dpi=dpi)
        except Exception as e:
            print(f"    [skip] eq_segmentation_comparison — {e}")
    else:
        print(f"    [skip] eq_segmentation_comparison — {snap_path} not found")


# ═════════════════════════════════════════════════════════════════════════════
# STATE-PAIR CORRELATION FIGURES
#   Reads pre-computed data written by state_pair_correlation.py under
#   --corr_dir/k{k}/.  Regenerates, with publication styling:
#     k{k}/plot_f1_vs_jsd.png        — F1 vs Jensen–Shannon divergence
#     k{k}/plot_f1_vs_pca_dist.png   — F1 vs PCA centroid distance
#     k{k}/plot_correlation_null.png — permutation null distributions
# ═════════════════════════════════════════════════════════════════════════════

def _fmt_p(p):
    """Format a permutation p-value for display (clamped)."""
    return _pfmt(p)


def _corr_scatter(x, y, state_i, k, xlabel, ylabel, title,
                  r, p_perm, out_path, dpi):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 7))

    # OLS trend line — stats ride along in its legend label
    mask = ~(np.isnan(x) | np.isnan(y))
    ols_line = None
    if mask.sum() >= 3:
        coeffs = np.polyfit(x[mask], y[mask], 1)
        xr = np.linspace(x[mask].min(), x[mask].max(), 200)
        ols_line, = ax.plot(xr, np.polyval(coeffs, xr),
                            color="#334155", lw=1.6, zorder=2)

    # Build the statistics string (legend entry)
    stat_txt = None
    if r is not None:
        stat_txt = rf"$\rho$ = {r:+.3f}"

    handles = []
    if ols_line is not None:
        ols_line.set_label("OLS")
        handles.append(ols_line)
    elif stat_txt is not None:
        handles.append(mlines.Line2D([], [], color="none", label=stat_txt))

    # Points coloured by train-state
    for xi, yi, si in zip(x, y, state_i):
        ax.scatter(xi, yi, color=STATE_PALETTE[int(si) % len(STATE_PALETTE)],
                   s=70, zorder=3, edgecolors="white", linewidths=0.8)

    # State legend
    state_handles = [
        mlines.Line2D([], [], marker="o", color="white",
                      markerfacecolor=STATE_PALETTE[s % len(STATE_PALETTE)],
                      markeredgecolor="white", markersize=8, label=f"S{s}")
        for s in range(k)
    ]
    leg = ax.legend(handles=handles + state_handles, title="Train state",
                    fontsize=13, title_fontsize=13, loc="best",
                    framealpha=0.92, ncol=1 if k <= 4 else 2)
    leg._legend_box.align = "left"

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(title, fontsize=17, pad=10)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.grid(True, alpha=0.25, ls="--", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)


def _corr_null_plot(null_jsd, obs_jsd, p_jsd,
                    null_pca, obs_pca, p_pca, out_path, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, null, obs, p, measure in [
        (axes[0], null_jsd, obs_jsd, p_jsd, "Jensen–Shannon divergence"),
        (axes[1], null_pca, obs_pca, p_pca, "PCA centroid distance"),
    ]:
        null = np.asarray(null, dtype=float)
        ax.hist(null, bins=50, color="#94A3B8", edgecolor="white",
                lw=0.3, alpha=0.85, label="Permutation null")
        ax.axvline(obs, color="#DC2626", lw=2.2,
                   label=rf"Observed $\rho$ = {obs:+.3f}, {_fmt_p(p)}")
        pct5 = np.percentile(null, 5)
        ax.axvline(pct5, color="#1f2937", lw=1.1, ls="--", alpha=0.7,
                   label=f"Null 5th pct = {pct5:+.3f}")
        ax.set_xlabel(r"Spearman $\rho$", fontsize=14)
        ax.set_title(f"F1 vs. {measure}", fontsize=14)
        ax.grid(True, axis="y", alpha=0.25, ls="--", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.92)

    axes[0].set_ylabel("Frequency", fontsize=12)
    fig.suptitle("Permutation Tests: State-Pair Dissimilarity vs. Generalization",
                 fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, out_path, dpi=dpi)


def figure_state_pair_correlation(k, corr_dir, k_out, dpi):
    banner(f"  State-pair correlation figures  (k={k})")

    src      = os.path.join(corr_dir, f"k{k}")
    pair_csv = os.path.join(src, "statepair_f1.csv")
    corr_csv = os.path.join(src, "correlation_results.csv")
    null_npz = os.path.join(src, "correlation_null.npz")

    if not os.path.exists(pair_csv):
        print(f"    [skip] all state-pair correlation figures — {pair_csv} not found")
        return

    df  = pd.read_csv(pair_csv)
    f1  = df["mean_f1"].values
    jsd = df["jsd"].values
    pca = df["pca_dist"].values
    si  = df["state_i"].values

    # Spearman r + permutation p for annotations (from correlation_results.csv)
    r_jsd = r_pca = p_jsd = p_pca = None
    if os.path.exists(corr_csv):
        cdf = pd.read_csv(corr_csv).set_index("measure")
        if "JSD" in cdf.index:
            r_jsd, p_jsd = float(cdf.loc["JSD", "spearman_r"]), float(cdf.loc["JSD", "p_perm"])
        if "PCA_dist" in cdf.index:
            r_pca, p_pca = float(cdf.loc["PCA_dist", "spearman_r"]), float(cdf.loc["PCA_dist", "p_perm"])

    _corr_scatter(
        jsd, f1, si, k,
        xlabel="Jensen–Shannon divergence between class distributions",
        ylabel="Mean column-centered macro F1",
        title="Transfer Performance vs. Class-Distribution Dissimilarity",
        r=r_jsd, p_perm=p_jsd,
        out_path=os.path.join(k_out, "plot_f1_vs_jsd.png"), dpi=dpi)

    _corr_scatter(
        pca, f1, si, k,
        xlabel="Euclidean distance between PCA centroids (z-scored space)",
        ylabel="Mean column-centered macro F1",
        title="Transfer Performance vs. Weight-Space Dissimilarity",
        r=r_pca, p_perm=p_pca,
        out_path=os.path.join(k_out, "plot_f1_vs_pca_dist.png"), dpi=dpi)

    if os.path.exists(null_npz):
        nd = np.load(null_npz, allow_pickle=True)
        _corr_null_plot(
            null_jsd=nd["null_jsd"], obs_jsd=float(nd["obs_jsd"]), p_jsd=float(nd["p_jsd"]),
            null_pca=nd["null_pca"], obs_pca=float(nd["obs_pca"]), p_pca=float(nd["p_pca"]),
            out_path=os.path.join(k_out, "plot_correlation_null.png"), dpi=dpi)
    else:
        print(f"    [skip] plot_correlation_null — {null_npz} not found "
              f"(re-run state_pair_correlation.py to save it)")


# ═════════════════════════════════════════════════════════════════════════════
# PER-K
# ═════════════════════════════════════════════════════════════════════════════

def run_per_k(dataset, k, hmm_dir, perf_dir, wa_dir, manifest_path,
              out_dir, eq_dir, corr_dir, jsd_dir, partial_jsd_dir, dpi):
    print(f"\n{'═'*60}")
    print(f"  Generating figures for {dataset}, k = {k}")
    print(f"{'═'*60}")

    k_out = os.path.join(out_dir, f"k{k}")
    os.makedirs(k_out, exist_ok=True)

    # Load HMM decode file  (new name: <dataset>_decode_k<k>.npz)
    decode_path = os.path.join(hmm_dir, f"{dataset}_decode_k{k}.npz")
    if not os.path.exists(decode_path):
        # legacy fallback
        legacy = os.path.join(hmm_dir, f"final_decode_k{k}.npz")
        decode_path = legacy if os.path.exists(legacy) else decode_path
    dec = load_npz(decode_path, os.path.basename(decode_path))
    if dec is None:
        print(f"  [skip] all k={k} figures — decode file missing")
        return

    state_seq  = dec["state_seq"].astype(int)
    window_ids = dec["window_ids"].astype(int)
    trans_mat  = dec["transition_matrix"]

    start_dates, end_dates = window_calendar_dates(window_ids, manifest_path)

    _transition_matrix(trans_mat, k,
                       os.path.join(k_out, "transition_matrix.png"), dpi)

    _state_strip(state_seq, start_dates, end_dates, k,
                 os.path.join(k_out, "state_strip.png"), dpi)

    _state_timeline_seeds(dec, start_dates, end_dates, k,
                          os.path.join(k_out, "state_timeline_seeds.png"), dpi)

    _timeline_with_classes(dec, manifest_path, k,
                           os.path.join(k_out, "timeline_with_classes.png"), dpi)

    # Within / across figures
    wa_k_dir    = os.path.join(wa_dir, f"k{k}")
    summary_csv = os.path.join(wa_k_dir, "within_across_summary.csv")
    null_npz    = os.path.join(wa_k_dir, "distance_conditioned_null.npz")

    if os.path.exists(summary_csv):
        _f1_vs_distance(pd.read_csv(summary_csv),
                        os.path.join(k_out, "f1_vs_distance.png"), dpi)
    else:
        print(f"    [skip] f1_vs_distance — {summary_csv} not found")

    _distance_conditioned_null(null_npz,
                               os.path.join(k_out, "distance_conditioned_null.png"), dpi)

    # State-pair and sorted-heatmap figures (need cross-window F1 matrix)
    f1d = load_npz(os.path.join(perf_dir, "cross_window_f1_colcentered.npz"))
    if f1d is not None:
        f1_matrix    = _get_f1_matrix(f1d)
        valid_ids_f1 = f1d["valid_ids"].tolist()
        wids_list    = window_ids.tolist()

        _statepair_heatmap(f1_matrix, valid_ids_f1, state_seq, wids_list, k,
                           os.path.join(k_out, "statepair_heatmap.png"), dpi)

        _f1_heatmap_with_states(f1_matrix, valid_ids_f1, state_seq, wids_list, k,
                                os.path.join(k_out, "f1_heatmap_with_states.png"), dpi)
    else:
        print("    [skip] statepair_heatmap and f1_heatmap_with_states — "
              "cross_window_f1_colcentered.npz not found")

    # Equal-segmentation figures (reads pre-computed data from check_equal_windows.py)
    figure_equal_segmentation(k, eq_dir, k_out, dpi)
    # State-pair correlation figures (reads pre-computed data from state_pair_correlation.py)
    figure_state_pair_correlation(k, corr_dir, k_out, dpi)
    # JSD stability figures (reads pre-computed data from jsd_stability_analysis.py)
    figure_jsd_stability(k, jsd_dir, k_out, dpi, state_seq=state_seq)
    # Partial-JSD transfer figures
    figure_partial_jsd(k, partial_jsd_dir, k_out, dpi)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    spec = get_spec(args.dataset)          # validate + drive window cadence
    fill_defaults(args)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  generate_all_figures.py")
    print(f"  dataset    : {spec.name}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  k values   : {args.k}")
    print(f"  eq_dir     : {args.eq_dir}")
    print(f"  dpi        : {args.dpi}")
    print(f"{'═'*60}")

    # Global (k-independent) figures
    figure_window_distributions(spec, args.splits_dir, args.output_dir, args.dpi)
    figure_cross_window_heatmap(args.perf_dir, args.output_dir, args.dpi)
    _max_k = {"fakeddit": 12, "yelp": 18}.get(args.dataset, None)
    figure_model_selection(args.select_dir, args.output_dir, args.dpi, max_k=_max_k)
    figure_pca_sanity(args.pca_npz, args.output_dir, args.dpi)

    # Per-k figures (including equal-segmentation)
    for k in args.k:
        run_per_k(spec.name, k, args.hmm_dir, args.perf_dir, args.wa_dir,
                  args.manifest, args.output_dir, args.eq_dir, args.corr_dir,
                  args.jsd_dir, args.partial_jsd_dir, args.dpi)

    print(f"\n{'═'*60}")
    print(f"  Done.  All figures in: {args.output_dir}/")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()