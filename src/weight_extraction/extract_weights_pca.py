"""
extract_weights_pca.py

Extracts MLP head weights from all trained per-window models, aligns hidden
units across seeds to a synthetic centroid via linear assignment, reduces via
PCA, z-scores, computes per-window seed centroids, and saves everything needed
for HMM fitting. Dataset-agnostic: the architecture is read off the first
checkpoint, so the same code runs unchanged on Fakeddit (2816, 1024, 6) and
Yelp (768, 256, 5).

PCA is fit with batch sklearn PCA over the materialised (N_models x D) matrix.
No component-selection heuristic is applied here: all reasonable principal
components (n_fit = min(n_rows - 1, D)) are fit and the full-width Z_scaled,
centroid matrix C, and complete explained-variance spectrum are saved. The
number of components to use downstream is chosen in select_hmm_states.py /
decode_hmm.py via their --n_pcs argument.

The before/after sanity-check visualisations are preserved: the unaligned
weights get their own PCA (so seed clustering is visible), and the aligned
weights get the full PCA that is saved and used downstream. Only the leading
two components are needed for the scatter plots.

Pipeline:
  1. Load best_model.pt for every (window, seed) pair. Extract structured
     weight matrices W_h, b_h, W_o, b_o (shapes derived from the checkpoint).
  2. Per-window iterative alignment (joint fingerprint): align all seeds to a
     synthetic centroid via iterated Hungarian assignment until convergence.
     Fingerprint[i] = concat(W_h[i,:], W_o[:,i]). Permutation applied
     consistently to W_h rows, b_h, and W_o columns; b_o unchanged.
  3. Fit batch PCA on the flattened aligned weights, keeping all
     n_fit = min(n_rows - 1, D) components. An independent 2-component PCA is
     fit over the UNALIGNED weights for the before-plot.
  4. Project aligned weights -> Z (full width); z-score -> Z_scaled.
  5. Compute per-window centroid in PCA space -> C.
  6. Save outputs + before/after sanity plots + a scree plot of the full EVR.

Outputs (written to --output_dir):
  weights_pca.npz          <- Z_scaled (full width), Z_before_scaled, C,
                              window_ids, seed_ids, centroid_wins,
                              explained_variance_ratio, evr_full
  pca_model.pkl            <- fitted sklearn PCA object (aligned-space, all comps)
  scaler_model.pkl         <- fitted sklearn StandardScaler object
  pca_summary.json
  sanity_check_before.png
  sanity_check_after.png
  scree.png

Usage:
    python src/extract_weights_pca.py \\
        --runs_dir   runs/hmm_windows/fakeddit \\
        --output_dir data/hmm_weights/fakeddit \\
        --n_windows  35 \\
        --n_seeds    10

    # Yelp:
    python src/extract_weights_pca.py \\
        --runs_dir   runs/hmm_windows/yelp \\
        --output_dir data/hmm_weights/yelp \\
        --n_windows  84 \\
        --n_seeds    10

    # dry-run (checks paths only, no loading):
    python src/extract_weights_pca.py --runs_dir runs/hmm_windows/yelp --dry_run
"""

import os
import sys
import json
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor

# Shared, dataset-agnostic model helpers: the single source of truth for the
# architecture, the flat-vector layout, and the state-dict <-> numpy I/O.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import (
    load_structured_weights,
    flatten_weights,
    flat_dim,
    infer_arch_from_checkpoint,
)


def ckpt_path(runs_dir: str, window_idx: int, seed_idx: int) -> str:
    """Return the best_model.pt path for a given (window, seed)."""
    return os.path.join(
        runs_dir,
        f"window_{window_idx:03d}",
        f"seed_{seed_idx}",
        "best_model.pt",
    )


def collect_weights(runs_dir, n_windows, n_seeds, dry_run=False, n_workers=8):
    """Load structured weights for every (window, seed) pair in parallel."""
    weights_by_window = [{} for _ in range(n_windows)]
    window_ids, seed_ids, missing = [], [], []

    # Walk paths first with a cheap stat, splitting found vs missing.
    todo = []  # (w, s, path)
    for w in range(n_windows):
        for s in range(n_seeds):
            path = ckpt_path(runs_dir, w, s)
            if os.path.exists(path):
                todo.append((w, s, path))
                window_ids.append(w)
                seed_ids.append(s)
            else:
                missing.append(path)

    if dry_run:
        return weights_by_window, window_ids, seed_ids, missing

    # Load in parallel; assign by (w, s) key so order is completion-independent.
    def _load(item):
        w, s, path = item
        return w, s, load_structured_weights(path)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for w, s, sw in ex.map(_load, todo):
            weights_by_window[w][s] = sw

    return weights_by_window, window_ids, seed_ids, missing


def first_existing_ckpt(runs_dir: str, n_windows: int, n_seeds: int) -> str:
    """Return the path of the first checkpoint that exists, for arch inference."""
    for w in range(n_windows):
        for s in range(n_seeds):
            path = ckpt_path(runs_dir, w, s)
            if os.path.exists(path):
                return path
    return ""


def unit_fingerprints(sw: dict) -> np.ndarray:
    """Per-unit fingerprint matrix of shape (hidden_size, input_dim + num_classes).

    Row i = concat(W_h[i, :], W_o[:, i]), encoding both what unit i receives
    (W_h row) and what it sends (W_o column). Shapes are read from the arrays,
    so this is architecture-agnostic.
    """
    return np.concatenate([sw["W_h"], sw["W_o"].T], axis=1)


def cost_matrix(ref: dict, src: dict) -> np.ndarray:
    """Squared-Euclidean cost matrix between fingerprints of ref and src.

    Shape (hidden_size, hidden_size); M[i,j] = ||fp_ref[i] - fp_src[j]||^2,
    computed via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b.
    """
    fp_ref = unit_fingerprints(ref)
    fp_src = unit_fingerprints(src)
    ref_sq = (fp_ref ** 2).sum(axis=1, keepdims=True)
    src_sq = (fp_src ** 2).sum(axis=1, keepdims=True)
    M      = ref_sq + src_sq.T - 2.0 * (fp_ref @ fp_src.T)
    return np.maximum(M, 0.0)   # Clamp tiny negatives from float error.


def apply_permutation(sw: dict, perm: np.ndarray) -> dict:
    """Return a new weight dict with hidden units reordered by perm.

    W_h rows, b_h entries, and W_o columns are permuted consistently. The
    output bias b_o has no hidden-unit indexing and is unchanged.
    """
    return {
        "W_h": sw["W_h"][perm, :],
        "b_h": sw["b_h"][perm],
        "W_o": sw["W_o"][:, perm],
        "b_o": sw["b_o"],
    }


def match_to_reference(ref: dict, src: dict) -> dict:
    """Permute src's hidden units to minimise fingerprint distance to ref."""
    _, col_ind = linear_sum_assignment(cost_matrix(ref, src))
    return apply_permutation(src, col_ind)


def centroid_weights(aligned_seeds: dict) -> dict:
    """Return the mean weight matrices across a dict of aligned seed weights."""
    keys = list(aligned_seeds.keys())
    return {
        "W_h": np.mean([aligned_seeds[s]["W_h"] for s in keys], axis=0),
        "b_h": np.mean([aligned_seeds[s]["b_h"] for s in keys], axis=0),
        "W_o": np.mean([aligned_seeds[s]["W_o"] for s in keys], axis=0),
        "b_o": np.mean([aligned_seeds[s]["b_o"] for s in keys], axis=0),
    }


def align_window(seeds: dict, max_iter: int = 20, tol: float = 1e-4) -> dict:
    """Align all seeds in a window to a synthetic centroid via iterated Hungarian.

    No bootstrap reference seed is used. Iteration always re-aligns from each
    seed's original (unaligned) weights to avoid permutation-composition drift.
    """
    seed_indices = sorted(seeds.keys())
    current = dict(seeds)

    prev_centroid = None
    for _ in range(max_iter):
        synth = centroid_weights(current)

        # Stop once the centroid stops moving between iterations.
        if prev_centroid is not None:
            delta = (
                np.concatenate([synth[k].ravel() for k in ["W_h", "b_h", "W_o", "b_o"]])
              - np.concatenate([prev_centroid[k].ravel() for k in ["W_h", "b_h", "W_o", "b_o"]])
            )
            if np.linalg.norm(delta) < tol:
                break

        prev_centroid = synth
        current = {s: match_to_reference(synth, seeds[s]) for s in seed_indices}

    return current


def ordered_rows(by_window: dict, n_windows: int) -> list:
    """Return a deterministic (window, seed) row order shared by every pass.

    Ascending window then seed order, restricted to windows present in
    `by_window`. Using one canonical order for fit, transform, and the id
    arrays guarantees row i of Z corresponds to (window_ids[i], seed_ids[i]).
    """
    rows = []
    for w in range(n_windows):
        sw = by_window.get(w) if isinstance(by_window, dict) else by_window[w]
        if not sw:
            continue
        for s in sorted(sw.keys()):
            rows.append((w, s))
    return rows


def _stack_rows(rows, by_window, D):
    """Materialise the full (n_rows x D) float32 weight matrix in memory.

    Built once and reused for both fit and transform so we don't flatten twice.
    """
    return np.stack(
        [flatten_weights(by_window[w][s]) for (w, s) in rows], axis=0
    ).astype(np.float32)


def dense_fit(rows, by_window, n_components, D, tag="", _cache=None):
    """Fit a batch sklearn PCA over the materialised weight matrix.

    `_cache` optionally receives the materialised matrix so a following
    dense_transform can reuse it instead of re-stacking.
    """
    W = _stack_rows(rows, by_window, D)
    if _cache is not None:
        _cache["W"] = W
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(W)
    if tag:
        print(f"  {tag}PCA fit over {W.shape[0]} rows "
              f"(D={D}, n_components={n_components}, {W.nbytes/1e6:.0f} MB)")
    return pca


def dense_transform(rows, by_window, pca, out_dim, D, _cache=None):
    """Project a materialised matrix; reuse a cached stack when available."""
    if _cache is not None and "W" in _cache:
        W = _cache["W"]
    else:
        W = _stack_rows(rows, by_window, D)
    return pca.transform(W)[:, :out_dim].astype(np.float32)


def fit_scaler(Z: np.ndarray) -> StandardScaler:
    """Fit and return a StandardScaler on Z."""
    scaler = StandardScaler()
    scaler.fit(Z)
    return scaler


def compute_centroids(Z_scaled, window_ids, seed_ids, n_windows, n_seeds) -> tuple:
    """Average z-scored PCA vectors across seeds, per window.

    Returns
    -------
    C             : (n_valid_windows, n_components) centroid matrix
    centroid_wins : list of window indices with >= 1 seed
    """
    n_components = Z_scaled.shape[1]
    accum  = np.zeros((n_windows, n_components), dtype=np.float64)
    counts = np.zeros(n_windows, dtype=int)

    for i, (w, _) in enumerate(zip(window_ids, seed_ids)):
        accum[w]  += Z_scaled[i]
        counts[w] += 1

    # Warn about windows that are missing seeds before averaging.
    incomplete = [(w, counts[w]) for w in range(n_windows)
                  if 0 < counts[w] < n_seeds]
    if incomplete:
        print(f"  WARNING: {len(incomplete)} window(s) have < {n_seeds} seeds:")
        for w, cnt in incomplete:
            print(f"    window_{w:03d}: {cnt}/{n_seeds} seeds")

    valid = [w for w in range(n_windows) if counts[w] > 0]
    C     = (accum[valid] / counts[valid, None]).astype(np.float32)
    return C, valid


SEED_MARKERS = ['o', 's', '^', 'D', 'P', 'v', '*', 'X', 'h', '+']


def scatter_plot(Z, window_ids, seed_ids, n_windows, title, out_path) -> None:
    """2-D scatter of PC1 vs PC2. Color = window index; marker = seed index."""
    color_values  = np.linspace(0.05, 0.95, n_windows)
    try:
        cmap_cont = matplotlib.colormaps["nipy_spectral"]
    except AttributeError:  # matplotlib < 3.5
        cmap_cont = cm.get_cmap("nipy_spectral")
    window_colors = {w: cmap_cont(color_values[w]) for w in range(n_windows)}

    fig, ax = plt.subplots(figsize=(11, 7))

    for i, (w, s) in enumerate(zip(window_ids, seed_ids)):
        ax.scatter(Z[i, 0], Z[i, 1],
                   color=window_colors[w],
                   marker=SEED_MARKERS[s % len(SEED_MARKERS)],
                   s=45, alpha=0.80, linewidths=0.3,
                   edgecolors="none", zorder=2)

    sm = plt.cm.ScalarMappable(
             cmap=plt.matplotlib.colors.ListedColormap(
                 [window_colors[w] for w in range(n_windows)]),
             norm=plt.Normalize(vmin=-0.5, vmax=n_windows - 0.5))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02,
                        ticks=np.arange(0, n_windows, max(1, n_windows // 10)))
    cbar.set_label("Window index", fontsize=9)

    unique_seeds = sorted(set(seed_ids))
    seed_handles = [
        plt.Line2D([0], [0],
                   marker=SEED_MARKERS[s % len(SEED_MARKERS)],
                   color="0.35", linestyle="none",
                   markersize=7, label=f"seed {s}")
        for s in unique_seeds
    ]
    ax.legend(handles=seed_handles, fontsize=8, loc="upper left",
              title="Seed", title_fontsize=8, framealpha=0.7)

    ax.set_xlabel("PC 1 (z-scored)", fontsize=11)
    ax.set_ylabel("PC 2 (z-scored)", fontsize=11)
    ax.set_title(title, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


def plot_scree(evr, out_path, max_show: int = 60) -> None:
    """Two-panel scree plot (linear + log) of the full explained-variance ratio.

    No cutoff is drawn — the number of components to use is chosen downstream
    (select_hmm_states.py / decode_hmm.py --n_pcs).
    """
    n_show  = min(len(evr), max_show)
    ranks   = np.arange(1, n_show + 1)
    cumvar  = np.cumsum(evr)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("PCA scree — full explained-variance spectrum",
                 fontsize=13, fontweight="bold")

    for ax, yscale in zip(axes, ["linear", "log"]):
        ax.plot(ranks, evr[:n_show] * 100, "o-", color="#2563EB",
                linewidth=1.8, markersize=4, label="Individual EVR %")
        ax2 = ax.twinx()
        ax2.plot(ranks, cumvar[:n_show] * 100, "s--", color="#DC2626",
                 linewidth=1.5, markersize=3, label="Cumulative EVR %")
        ax2.set_ylabel("Cumulative EVR (%)", fontsize=11)
        ax.set_xlabel("PC rank", fontsize=11)
        ax.set_ylabel("Variance explained (%)", fontsize=11)
        ax.set_title(f"{'Linear' if yscale == 'linear' else 'Log'} scale",
                     fontsize=11)
        ax.set_yscale(yscale)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, n_show + 1)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


def make_plots(Z_before, Z_after, window_ids, seed_ids, output_dir,
               n_windows) -> None:
    """Render the before/after alignment sanity-check scatter plots."""
    scatter_plot(
        Z_before, window_ids, seed_ids, n_windows,
        "BEFORE alignment: MLP weights in PCA space\n"
        "(dominant clustering is by seed — permutation symmetry)",
        os.path.join(output_dir, "sanity_check_before.png"),
    )
    scatter_plot(
        Z_after, window_ids, seed_ids, n_windows,
        "AFTER alignment: MLP weights in PCA space\n"
        "(seeds within each window should now cluster tightly)",
        os.path.join(output_dir, "sanity_check_after.png"),
    )


def main(args):
    """Run the extract/align/PCA/save pipeline for one dataset."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Infer architecture from a real checkpoint (no hardcoding).
    probe = first_existing_ckpt(args.runs_dir, args.n_windows, args.n_seeds)
    if not probe:
        print(f"ERROR: no checkpoints found under {args.runs_dir}.")
        sys.exit(1)
    arch = infer_arch_from_checkpoint(probe)
    D    = flat_dim(arch["input_dim"], arch["hidden_size"], arch["num_classes"])
    print(f"── Architecture (from {os.path.relpath(probe, args.runs_dir)}) ──")
    print(f"  input_dim={arch['input_dim']}  hidden_size={arch['hidden_size']}  "
          f"num_classes={arch['num_classes']}")
    print(f"  flat weight dimension D = {D:,}")

    # Step 1: load checkpoints.
    print(f"\n── Step 1: Loading checkpoints from {args.runs_dir} ──")
    weights_by_window, window_ids, seed_ids, missing = collect_weights(
        args.runs_dir, args.n_windows, args.n_seeds,
        dry_run=args.dry_run, n_workers=args.n_workers,
    )

    n_found    = len(window_ids)
    n_expected = args.n_windows * args.n_seeds
    print(f"  Found : {n_found} / {n_expected} checkpoints")
    if missing:
        print(f"  Missing ({len(missing)}):")
        for p in missing:
            print(f"    {p}")
    if n_found == 0:
        print("ERROR: No checkpoints found. Check --runs_dir.")
        sys.exit(1)
    if args.dry_run:
        print("\nDry run complete. Exiting.")
        return

    # Step 2: per-window iterative alignment.
    print(f"\n── Step 2: Iterative alignment (no bootstrap reference) ──")
    aligned_by_window = {}
    for w in range(args.n_windows):
        sw = weights_by_window[w]
        if not sw:
            continue
        aligned_by_window[w] = align_window(sw)
        seed_str = ", ".join(str(s) for s in sorted(sw.keys()))
        print(f"  window_{w:03d}: {len(sw)} seed(s) [{seed_str}] aligned")

    # Canonical row order shared by every pass and the id arrays.
    before_rows = ordered_rows(weights_by_window, args.n_windows)
    after_rows  = ordered_rows(aligned_by_window, args.n_windows)
    # window_ids / seed_ids track the ALIGNED order used for all saved outputs.
    window_ids = [w for (w, s) in after_rows]
    seed_ids   = [s for (w, s) in after_rows]

    n_rows_after  = len(after_rows)
    n_rows_before = len(before_rows)

    # One materialised aligned matrix is reused across the fit and projection
    # (built once, in dense_fit, via this cache).
    aligned_cache = {}

    # Step 3: full PCA on the aligned weights, keeping all reasonable
    # components. n_fit = min(n_rows - 1, D) is the maximal non-degenerate rank
    # of a centered (n_rows x D) matrix; the downstream --n_pcs truncates.
    n_fit = min(n_rows_after - 1, D)
    n_fit = max(n_fit, 1)

    print(f"\n── Step 3: Full PCA (aligned, {n_fit} comps = all non-degenerate) ──")
    pca = dense_fit(
        after_rows, aligned_by_window, n_fit, D,
        tag="[aligned] ", _cache=aligned_cache,
    )
    evr_full = pca.explained_variance_ratio_
    cumvar   = np.cumsum(evr_full)
    print(f"  Kept {n_fit} components  "
          f"({cumvar[-1]*100:.2f}% of variance captured in total)")

    # Before-plot PCA (unaligned): 2 comps are enough for the scatter.
    print(f"\n── Step 3b: PCA (unaligned, for before-plot) ──")
    before_cache   = {}
    n_before_comps = min(2, n_rows_before - 1, D)
    pca_before = dense_fit(
        before_rows, weights_by_window, n_before_comps, D,
        tag="[unaligned] ", _cache=before_cache,
    )

    # Step 4: project both spaces.
    print(f"\n── Step 4: Projecting ──")
    Z_after     = dense_transform(
        after_rows, aligned_by_window, pca, n_fit, D, _cache=aligned_cache,
    )
    Z_before_2d = dense_transform(
        before_rows, weights_by_window, pca_before, n_before_comps, D,
        _cache=before_cache,
    )
    # Release materialised stacks before the (small) downstream work.
    aligned_cache.clear()
    before_cache.clear()
    print(f"  Z_after: {Z_after.shape}   Z_before: {Z_before_2d.shape}")

    # Step 5: z-score independently for the before-plot and the aligned space.
    print(f"\n── Step 5: Z-scoring ──")
    scaler_before   = fit_scaler(Z_before_2d)
    Z_before_scaled = scaler_before.transform(Z_before_2d).astype(np.float32)

    scaler         = fit_scaler(Z_after)
    Z_after_scaled = scaler.transform(Z_after).astype(np.float32)
    print(f"  Z_after  mean≈{Z_after_scaled.mean():.4f}  "
          f"std≈{Z_after_scaled.std():.4f}  (should be ≈0, ≈1)")

    # Step 6: centroids in aligned PCA space (full width).
    print(f"\n── Step 6: Computing per-window centroids ──")
    C, centroid_wins = compute_centroids(
        Z_after_scaled, window_ids, seed_ids, args.n_windows, args.n_seeds
    )
    print(f"  Centroid matrix: {C.shape}  ({len(centroid_wins)} windows)")

    # Step 7: save outputs.
    print(f"\n── Step 7: Saving outputs → {args.output_dir} ──")
    npz_path = os.path.join(args.output_dir, "weights_pca.npz")
    np.savez(
        npz_path,
        Z_scaled                 = Z_after_scaled,
        Z_before_scaled          = Z_before_scaled,
        C                        = C,
        window_ids               = np.array(window_ids,    dtype=np.int32),
        seed_ids                 = np.array(seed_ids,      dtype=np.int32),
        centroid_wins            = np.array(centroid_wins, dtype=np.int32),
        explained_variance_ratio = pca.explained_variance_ratio_,
        evr_full                 = evr_full,
        input_dim                = np.int32(arch["input_dim"]),
        hidden_size              = np.int32(arch["hidden_size"]),
        num_classes              = np.int32(arch["num_classes"]),
    )
    print(f"  weights_pca.npz  → {npz_path}")

    cum_var = float(np.sum(pca.explained_variance_ratio_))
    pca_summary = {
        "n_components_fit":        int(n_fit),
        "cumulative_variance_pct": round(cum_var * 100, 4),
        "per_component_evr":       [round(float(v), 6) for v in pca.explained_variance_ratio_],
        "input_dim":               arch["input_dim"],
        "hidden_size":             arch["hidden_size"],
        "num_classes":             arch["num_classes"],
        "flat_dim":                int(D),
        "note":                    "All non-degenerate PCs saved. Choose the "
                                   "number to use downstream via --n_pcs.",
    }
    summary_path = os.path.join(args.output_dir, "pca_summary.json")
    with open(summary_path, "w") as f:
        json.dump(pca_summary, f, indent=2)
    print(f"  pca_summary.json → {summary_path}")

    with open(os.path.join(args.output_dir, "pca_model.pkl"), "wb") as f:
        pickle.dump(pca, f)
    print(f"  pca_model.pkl    → {os.path.join(args.output_dir, 'pca_model.pkl')}")

    with open(os.path.join(args.output_dir, "scaler_model.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"  scaler_model.pkl → {os.path.join(args.output_dir, 'scaler_model.pkl')}")

    # Step 8: sanity-check plots.
    print(f"\n── Step 8: Generating sanity-check plots ──")
    make_plots(
        Z_before_scaled, Z_after_scaled,
        window_ids, seed_ids,
        args.output_dir, args.n_windows,
    )
    plot_scree(evr_full, os.path.join(args.output_dir, "scree.png"))

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Checkpoints loaded    : {n_found} / {n_expected}")
    print(f"  Architecture          : ({arch['input_dim']}, "
          f"{arch['hidden_size']}, {arch['num_classes']})  D={D:,}")
    print(f"  PCA components (all)   : {n_fit}  ({cum_var*100:.1f}% variance)")
    print(f"  Centroid matrix       : {C.shape}  → ready for HMM")
    print(f"  Choose n_pcs downstream in select_hmm_states.py / decode_hmm.py")
    print(f"  Outputs               : {args.output_dir}/")
    if missing:
        print(f"\n  WARNING: {len(missing)} missing checkpoints (listed above).")
    print(f"{'='*60}")


def parse_args():
    """Parse CLI arguments for the extraction run."""
    p = argparse.ArgumentParser(
        description="Extract, align, and PCA-reduce MLP weights for HMM fitting. "
                    "Fits all non-degenerate PCs; the number used is chosen "
                    "downstream via --n_pcs."
    )
    p.add_argument("--runs_dir", required=True,
                   help="Root of trained window runs, e.g. runs/hmm_windows/<dataset>")
    p.add_argument("--output_dir", default="data/hmm_weights",
                   help="Where to write outputs (created if absent).")
    p.add_argument("--n_windows", type=int, required=True,
                   help="Number of windows (35 Fakeddit, 84 Yelp).")
    p.add_argument("--n_seeds", type=int, default=10,
                   help="Seeds trained per window.")
    p.add_argument("--dry_run", action="store_true",
                   help="Check checkpoint paths only; skip loading and processing.")
    p.add_argument("--n_workers", type=int, default=8,
               help="Threads for parallel checkpoint loading. I/O-bound, so "
                    "threads (not processes); 4-16 is the useful range, and "
                    "networked storage saturates around 8-16.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())