"""
select_hmm_states.py

Model selection for the number of HMM states k over [k_min, k_max].

Three complementary criteria are evaluated:
  1. BIC    -- computed on all per-seed Z_scaled sequences
               (n_seeds x n_windows obs).
  2. AIC    -- same fit, lighter penalty (2 * n_params). Tends to favour
               slightly larger k.
  3. LOO-CV -- leave-one-seed-out: for each held-out seed, fit HMM on the
               remaining seeds' sequences, score on the held-out seed's
               sequence. Pick k that maximises mean held-out log-likelihood.

The number of principal components fed to the HMM is chosen here via --n_pcs,
which slices the leading columns of the full-width Z_scaled. All criteria use
multiple random initialisations per k for stability. An adjusted rand index
(ARI) matrix gauges Viterbi stability across inits for every k.

Outputs written to --output_dir:
  hmm_scores.npz          -- raw BIC/AIC/LOO-CV arrays, per k, per init, per fold
  hmm_select_summary.csv  -- one row per k: bic, aic, loo_mean, loo_std, ari_mean
  model_selection.png     -- three-panel plot: BIC, AIC, LOO-CV
  ari_stability.png       -- per-k mean ARI across init pairs
  loo_per_fold.png        -- per-seed LOO-CV curves

Usage:
  python src/select_hmm_states.py \
      --input_npz  data/hmm_weights/weights_pca.npz \
      --output_dir data/hmm_hmm \
      --n_pcs 5 \
      --k_min 2 --k_max 10 \
      --n_inits 10 \
      --n_iter  200

Requirements: hmmlearn, scikit-learn, numpy, matplotlib, pandas, tqdm, joblib
"""

import os
import argparse
import warnings
import itertools
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hmmlearn import hmm
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")


def section(title):
    """Print a section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def banner(title):
    """Print a top-level banner header."""
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def elapsed(t0):
    """Return the time since t0 as a human-readable seconds/minutes string."""
    s = time.time() - t0
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


def load_data(npz_path, n_pcs):
    """Load weights_pca.npz and truncate Z_scaled / C to the leading n_pcs columns."""
    data       = np.load(npz_path)
    C          = data["C"].astype(np.float64)
    Z_scaled   = data["Z_scaled"].astype(np.float64)
    window_ids = data["window_ids"].astype(int)
    seed_ids   = data["seed_ids"].astype(int)

    # Validate the requested PC count against the saved width before slicing.
    n_avail = Z_scaled.shape[1]
    if n_pcs > n_avail:
        raise ValueError(
            f"--n_pcs={n_pcs} exceeds the {n_avail} PCs saved in {npz_path}. "
            f"Re-run extract_weights_pca.py or choose n_pcs <= {n_avail}."
        )
    if n_pcs < 1:
        raise ValueError(f"--n_pcs must be >= 1 (got {n_pcs}).")

    Z_scaled = Z_scaled[:, :n_pcs]
    C        = C[:, :n_pcs]
    print(f"  Using {n_pcs} of {n_avail} available PCs.")
    return C, Z_scaled, window_ids, seed_ids


def build_seed_sequences(Z_scaled, window_ids, seed_ids):
    """Group per-(window, seed) PCA vectors into per-seed sequences sorted by window."""
    seeds     = sorted(set(seed_ids))
    seed_seqs = {}
    for s in seeds:
        mask  = seed_ids == s
        order = np.argsort(window_ids[mask])
        seed_seqs[s] = Z_scaled[mask][order]
    return seed_seqs


def make_hmm(k, n_iter, random_state, covariance_type="diag",
             topology="ergodic", max_advance=1, min_covar=1e-3):
    """Build a GaussianHMM, optionally constrained to a left-to-right topology.

    For "left_to_right", seed a state-0 start and an upper-triangular transmat,
    and drop 's'/'t' from init_params so the zero structure survives EM. Means
    and covariances use hmmlearn's default initialisation. 't' stays in params
    so the surviving upper-triangular probabilities are re-estimated. The
    min_covar floor prevents diagonal-covariance underflow on short, low-dim
    sequences.
    """
    model = hmm.GaussianHMM(
        n_components=k, covariance_type=covariance_type, n_iter=n_iter,
        tol=1e-4, random_state=random_state, verbose=False, min_covar=min_covar,
    )
    if topology == "left_to_right":
        startprob = np.zeros(k)
        startprob[0] = 1.0
        transmat  = np.zeros((k, k))
        # Seed each row with uniform mass over the allowed forward band.
        for i in range(k):
            hi = min(i + max_advance, k - 1)
            transmat[i, i:hi + 1] = 1.0 / (hi - i + 1)
        transmat[k - 1, k - 1] = 1.0

        model.init_params = "mc"      # Let hmmlearn init means/covars only.
        model.params      = "stmc"    # Re-estimate all; transmat zeros stay zero.
        model.startprob_  = startprob
        model.transmat_   = transmat
    return model


def fit_on_sequences(seqs, k, n_iter, random_state, covariance_type="diag",
                     topology="ergodic", max_advance=1, min_covar=1e-2):
    """Fit one HMM on concatenated sequences; return the model or None on failure."""
    lengths = [s.shape[0] for s in seqs]
    X       = np.concatenate(seqs, axis=0)

    model = make_hmm(k, n_iter, random_state,
                     covariance_type=covariance_type,
                     topology=topology, max_advance=max_advance,
                     min_covar=min_covar)
    try:
        model.fit(X, lengths)
    except Exception:
        return None
    return model


def score_sequence(model, seq):
    """Score a sequence and divide by its length so folds are comparable.

    Returning a per-observation LL keeps held-out scores comparable across folds
    even when sequence lengths differ across seeds.
    """
    try:
        total_ll = model.score(seq, lengths=[seq.shape[0]])
        return total_ll / seq.shape[0]
    except Exception:
        return np.nan


def _n_params(k, D, covariance_type, topology="ergodic", max_advance=1):
    """Return the number of free parameters for a GaussianHMM."""
    cov_params = {
        "diag":      k * D,
        "full":      k * D * (D + 1) // 2,
        "tied":      D * (D + 1) // 2,
        "spherical": k,
    }.get(covariance_type, k * D)

    if topology == "left_to_right":
        # Start prob is fixed at state 0, so it contributes no free params.
        # Free transition params per row = nonzero band entries minus the
        # row-sum=1 normalisation constraint.
        start_params = 0
        trans_params = 0
        for i in range(k):
            hi = min(i + max_advance, k - 1)
            row_nonzero = hi - i + 1
            trans_params += max(row_nonzero - 1, 0)
    else:
        start_params = (k - 1)
        trans_params = k * (k - 1)

    return start_params + trans_params + k * D + cov_params


def _score_all(model, seqs):
    """Score all sequences jointly; return -inf on failure."""
    lengths = [s.shape[0] for s in seqs]
    X       = np.concatenate(seqs, axis=0)
    try:
        return model.score(X, lengths=lengths)
    except Exception:
        return -np.inf


def compute_bic(model, seqs, topology="ergodic", max_advance=1):
    """Return the BIC for a fitted model over seqs (inf if scoring fails)."""
    k     = model.n_components
    D     = seqs[0].shape[1]
    n_obs = sum(s.shape[0] for s in seqs)
    ll    = _score_all(model, seqs)
    if not np.isfinite(ll):
        return np.inf
    return -2.0 * ll + _n_params(k, D, model.covariance_type,
                                 topology, max_advance) * np.log(n_obs)


def compute_aic(model, seqs, topology="ergodic", max_advance=1):
    """Return the AIC for a fitted model over seqs (inf if scoring fails)."""
    k  = model.n_components
    D  = seqs[0].shape[1]
    ll = _score_all(model, seqs)
    if not np.isfinite(ll):
        return np.inf
    return -2.0 * ll + 2.0 * _n_params(k, D, model.covariance_type,
                                       topology, max_advance)


def run_bic_aic(seed_seqs, k_range, n_inits, n_iter, covariance_type,
                topology="ergodic", max_advance=1, min_covar=1e-2):
    """Fit n_inits HMMs on all seeds per k and report best BIC/AIC.

    Returns bic_best, bic_all, aic_best, aic_all, and a dict of fitted models
    keyed by (ki, init) so the ARI phase can reuse them instead of refitting
    identical random seeds.
    """
    all_seqs   = list(seed_seqs.values())
    bic_all    = np.full((len(k_range), n_inits), np.inf)
    aic_all    = np.full((len(k_range), n_inits), np.inf)
    fitted_models = {}

    print(f"\n  Fitting {len(k_range) * n_inits} HMMs total "
          f"({n_inits} inits x {len(k_range)} k-values)\n")

    pbar_k = tqdm(
        enumerate(k_range), total=len(k_range),
        desc="  BIC/AIC", unit="k", position=0, leave=True,
        bar_format="  {l_bar}{bar}| k {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
    for ki, k in pbar_k:
        pbar_k.set_description(f"  BIC/AIC  k={k}")
        t0     = time.time()
        failed = 0

        pbar_init = tqdm(
            range(n_inits), total=n_inits,
            desc=f"    init", unit="init", position=1, leave=False,
            bar_format="    {l_bar}{bar}| {n_fmt}/{total_fmt}",
        )
        for init in pbar_init:
            model = fit_on_sequences(all_seqs, k, n_iter,
                                     random_state=init * 100 + k,
                                     covariance_type=covariance_type,
                                     topology=topology, max_advance=max_advance,
                                     min_covar=min_covar)
            if model is None:
                failed += 1
                pbar_init.set_postfix({"last": "FAIL", "failed": failed})
                continue
            bic_val           = compute_bic(model, all_seqs,
                                             topology, max_advance)
            aic_val           = compute_aic(model, all_seqs,
                                             topology, max_advance)
            bic_all[ki, init] = bic_val
            aic_all[ki, init] = aic_val
            fitted_models[(ki, init)] = model   # Cache for the ARI phase.
            pbar_init.set_postfix({
                "bic":    f"{bic_val:.1f}",
                "aic":    f"{aic_val:.1f}",
                "failed": failed,
            })
        pbar_init.close()

        best_bic  = float(np.min(bic_all[ki]))
        best_aic  = float(np.min(aic_all[ki]))
        best_init = int(np.argmin(bic_all[ki]))
        finite    = bic_all[ki][np.isfinite(bic_all[ki])]
        rng       = f"[{finite.min():.1f}, {finite.max():.1f}]" if len(finite) else "N/A"
        print(
            f"  k={k:2d} | best BIC={best_bic:.2f}  best AIC={best_aic:.2f} "
            f"(init {best_init}) | range {rng} | failed={failed}/{n_inits} | {elapsed(t0)}"
        )

    bic_best = np.min(bic_all, axis=1)
    aic_best = np.min(aic_all, axis=1)
    k_arr    = np.array(k_range)
    print(f"\n  ✓ BIC optimum: k={k_arr[np.argmin(bic_best)]}  "
          f"(BIC={bic_best.min():.2f})")
    print(f"  ✓ AIC optimum: k={k_arr[np.argmin(aic_best)]}  "
          f"(AIC={aic_best.min():.2f})")
    return bic_best, bic_all, aic_best, aic_all, fitted_models


def _loo_one_cell(ki, k, fi, held_seed, train_seqs, held_seq,
                  n_inits, n_iter, covariance_type,
                  topology="ergodic", max_advance=1, min_covar=1e-2):
    """Run all inits for one (k, held-out-seed) cell; return held-out per-obs LL.

    Returns (ki, fi, held_ll, failed) where held_ll is np.nan if every init
    failed. The random_state matches the serial code so results are identical.
    """
    best_ll_train = -np.inf
    best_model    = None
    failed        = 0

    lengths = [s.shape[0] for s in train_seqs]
    X_train = np.concatenate(train_seqs, axis=0)

    for init in range(n_inits):
        model = fit_on_sequences(
            train_seqs, k, n_iter,
            random_state=init * 1000 + fi * 100 + k,
            covariance_type=covariance_type,
            topology=topology, max_advance=max_advance,
            min_covar=min_covar,
        )
        if model is None:
            failed += 1
            continue
        try:
            train_ll = model.score(X_train, lengths=lengths)
        except Exception:
            failed += 1
            continue
        # Select the fit with the best training LL to score on the held-out seed.
        if train_ll > best_ll_train:
            best_ll_train = train_ll
            best_model    = model

    if best_model is not None:
        held_ll = score_sequence(best_model, held_seq)
    else:
        held_ll = np.nan
    return ki, fi, held_ll, failed


def run_loo_cv(seed_seqs, k_range, n_inits, n_iter, covariance_type,
               topology="ergodic", max_advance=1, min_covar=1e-2, n_jobs=-1):
    """Parallel leave-one-seed-out CV; return (loo_mean, loo_std, loo_all).

    n_jobs is passed to joblib; -1 uses all available cores, 1 forces serial
    behaviour for debugging.
    """
    seeds   = sorted(seed_seqs.keys())
    n_seeds = len(seeds)
    # nan-initialised so failed cells drop out of nanmean/nanstd.
    loo_all = np.full((len(k_range), n_seeds), np.nan)

    total_fits = len(k_range) * n_seeds * n_inits
    print(f"\n  Fitting {total_fits} HMMs total "
          f"({n_inits} inits x {n_seeds} folds x {len(k_range)} k-values) "
          f"| n_jobs={n_jobs}\n")

    # Build the flat list of independent (k, fold) cells.
    tasks = []
    for ki, k in enumerate(k_range):
        for fi, held_seed in enumerate(seeds):
            train_seqs = [seed_seqs[s] for s in seeds if s != held_seed]
            held_seq   = seed_seqs[held_seed]
            tasks.append((ki, k, fi, held_seed, train_seqs, held_seq))

    # A single cell-level progress bar is the honest granularity here, since
    # nested per-init bars don't survive parallel dispatch.
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_loo_one_cell)(
            ki, k, fi, held_seed, train_seqs, held_seq,
            n_inits, n_iter, covariance_type,
            topology, max_advance, min_covar,
        )
        for (ki, k, fi, held_seed, train_seqs, held_seq)
        in tqdm(tasks, desc="  LOO cells", unit="cell",
                bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}]")
    )

    for ki, fi, held_ll, failed in results:
        loo_all[ki, fi] = held_ll

    # Per-k reporting for the log.
    for ki, k in enumerate(k_range):
        mean_ll = float(np.nanmean(loo_all[ki]))
        std_ll  = float(np.nanstd(loo_all[ki]))
        n_valid = int(np.sum(np.isfinite(loo_all[ki])))
        per_fold_str = "  ".join(
            f"seed_{s}: {loo_all[ki, fi]:.4f}" if np.isfinite(loo_all[ki, fi])
            else f"seed_{s}: FAIL"
            for fi, s in enumerate(seeds)
        )
        print(
            f"  k={k:2d} | mean held-out LL/obs={mean_ll:.4f} ± {std_ll:.4f} "
            f"({n_valid}/{n_seeds} folds)\n"
            f"        | per fold → {per_fold_str}"
        )

    loo_mean = np.nanmean(loo_all, axis=1)
    loo_std  = np.nanstd(loo_all,  axis=1)
    # Blank out k-values where every fold failed.
    all_failed = np.all(~np.isfinite(loo_all), axis=1)
    loo_mean[all_failed] = np.nan
    loo_std[all_failed]  = np.nan

    valid  = np.isfinite(loo_mean)
    k_arr  = np.array(k_range)
    k_best = k_arr[valid][np.argmax(loo_mean[valid])] if valid.any() else "N/A"
    best_v = loo_mean[valid].max() if valid.any() else float("nan")
    print(f"\n  ✓ LOO-CV optimum: k={k_best}  (mean held-out LL/obs={best_v:.4f})")
    return loo_mean, loo_std, loo_all


def run_ari_stability(seed_seqs, k_range, n_inits, n_iter, covariance_type,
                      fitted_models, topology="ergodic", max_advance=1):
    """Decode BIC/AIC-phase models with Viterbi and compute pairwise ARI per k.

    Models are reused from run_bic_aic (same random seeds, no extra EM runs);
    higher ARI means a more stable solution across random starts.
    """
    all_seqs = list(seed_seqs.values())
    X_full   = np.concatenate(all_seqs, axis=0)
    lengths  = [s.shape[0] for s in all_seqs]
    n_pairs  = n_inits * (n_inits - 1) // 2
    ari_all  = np.full((len(k_range), max(n_pairs, 1)), np.nan)

    print(f"\n  {n_inits} inits per k → {n_pairs} pairs per k")
    print(f"  Reusing models from BIC/AIC phase (no extra EM fits)\n")

    pbar_k = tqdm(
        enumerate(k_range), total=len(k_range),
        desc="  ARI", unit="k", position=0, leave=True,
        bar_format="  {l_bar}{bar}| k {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
    for ki, k in pbar_k:
        pbar_k.set_description(f"  ARI  k={k}")
        t0           = time.time()
        viterbi_seqs = []
        failed       = 0

        for init in range(n_inits):
            model = fitted_models.get((ki, init))
            if model is None:
                failed += 1
                continue
            try:
                _, states = model.decode(X_full, lengths=lengths,
                                         algorithm="viterbi")
                viterbi_seqs.append(states)
            except Exception:
                failed += 1

        pairs = list(itertools.combinations(range(len(viterbi_seqs)), 2))
        for pi, (a, b) in enumerate(pairs):
            ari_all[ki, pi] = adjusted_rand_score(viterbi_seqs[a],
                                                   viterbi_seqs[b])

        mean_ari = float(np.nanmean(ari_all[ki]))
        stable   = "✓ STABLE" if mean_ari >= 0.9 else "✗ unstable"
        print(
            f"  k={k:2d} | mean ARI={mean_ari:.4f} {stable} | "
            f"n_pairs={len(pairs)} | decoded={len(viterbi_seqs)}/{n_inits} | "
            f"failed={failed}/{n_inits} | {elapsed(t0)}"
        )

    ari_mean = np.nanmean(ari_all, axis=1)
    k_arr    = np.array(k_range)
    print(f"\n  ✓ Most stable k: k={k_arr[np.argmax(ari_mean)]}  "
          f"(mean ARI={ari_mean.max():.4f})")
    return ari_mean, ari_all


def plot_model_selection(k_range, bic_best, aic_best, loo_mean, loo_std, output_path):
    """Render the three-panel BIC / AIC / LOO-CV selection figure."""
    k_arr = np.array(k_range)
    k_bic = k_arr[np.argmin(bic_best)]
    k_aic = k_arr[np.argmin(aic_best)]
    valid = np.isfinite(loo_mean)
    k_loo = k_arr[valid][np.argmax(loo_mean[valid])] if valid.any() else None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("HMM State Selection", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(k_arr, bic_best, "o-", color="#2563EB", linewidth=2, markersize=6,
            label="BIC (best init)")
    ax.axvline(k_bic, color="#2563EB", linestyle="--", alpha=0.6,
               label=f"k={k_bic} (min BIC)")
    ax.set_xlabel("Number of states (k)", fontsize=12)
    ax.set_ylabel("BIC", fontsize=12)
    ax.set_title("BIC — lower is better", fontsize=12)
    ax.set_xticks(k_arr)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(k_arr, aic_best, "o-", color="#9333EA", linewidth=2, markersize=6,
            label="AIC (best init)")
    ax.axvline(k_aic, color="#9333EA", linestyle="--", alpha=0.6,
               label=f"k={k_aic} (min AIC)")
    ax.set_xlabel("Number of states (k)", fontsize=12)
    ax.set_ylabel("AIC", fontsize=12)
    ax.set_title("AIC — lower is better", fontsize=12)
    ax.set_xticks(k_arr)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(k_arr[valid], loo_mean[valid], "o-", color="#16A34A", linewidth=2,
            markersize=6, label="LOO-CV mean LL/obs")
    ax.fill_between(
        k_arr[valid],
        loo_mean[valid] - loo_std[valid],
        loo_mean[valid] + loo_std[valid],
        color="#16A34A", alpha=0.15, label="± 1 std (across folds)",
    )
    if k_loo is not None:
        ax.axvline(k_loo, color="#16A34A", linestyle="--", alpha=0.6,
                   label=f"k={k_loo} (max LOO LL/obs)")
    ax.set_xlabel("Number of states (k)", fontsize=12)
    ax.set_ylabel("Mean held-out log-likelihood per observation", fontsize=12)
    ax.set_title("LOO-CV — higher is better", fontsize=12)
    ax.set_xticks(k_arr)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_ari_stability(k_range, ari_all, output_path):
    """Render the per-k mean pairwise ARI bar chart with a 0.9 threshold line."""
    k_arr    = np.array(k_range)
    ari_mean = np.nanmean(ari_all, axis=1)
    ari_std  = np.nanstd(ari_all,  axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#6366F1" if a >= 0.9 else "#A5B4FC" for a in ari_mean]
    ax.bar(k_arr, ari_mean, color=colors, edgecolor="white", linewidth=0.8,
           yerr=ari_std, capsize=4,
           error_kw={"elinewidth": 1.2, "ecolor": "#475569"})
    ax.axhline(0.9, color="#DC2626", linestyle="--", linewidth=1.2,
               label="ARI = 0.9 threshold")
    ax.set_xlabel("Number of states (k)", fontsize=12)
    ax.set_ylabel("Mean pairwise ARI (across inits)", fontsize=12)
    ax.set_title("Viterbi Stability across Random Initialisations", fontsize=12)
    ax.set_xticks(k_arr)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_loo_per_fold(k_range, loo_all, output_path):
    """Render per-seed held-out LL curves with the mean overlaid."""
    k_arr       = np.array(k_range)
    n_seeds     = loo_all.shape[1]
    seed_colors = plt.cm.tab10(np.linspace(0, 0.9, n_seeds))

    fig, ax = plt.subplots(figsize=(9, 5))
    for fi in range(n_seeds):
        vals = loo_all[:, fi]
        mask = np.isfinite(vals)
        ax.plot(k_arr[mask], vals[mask], "o--", color=seed_colors[fi],
                alpha=0.8, linewidth=1.2, markersize=5,
                label=f"seed_{fi} held out")

    loo_mean = np.nanmean(loo_all, axis=1)
    # Blank out k-values where every fold failed.
    all_failed = np.all(~np.isfinite(loo_all), axis=1)
    loo_mean[all_failed] = np.nan
    valid = np.isfinite(loo_mean)
    ax.plot(k_arr[valid], loo_mean[valid], "k-", linewidth=2.5, label="mean")
    ax.set_xlabel("Number of states (k)", fontsize=12)
    ax.set_ylabel("Held-out log-likelihood per observation", fontsize=12)
    ax.set_title("LOO-CV — Per-seed held-out log-likelihood per observation", fontsize=12)
    ax.set_xticks(k_arr)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def build_summary(k_range, bic_best, aic_best, loo_mean, loo_std, ari_mean):
    """Assemble the per-k summary DataFrame with rank columns for each criterion."""
    k_arr = np.array(k_range)
    df = pd.DataFrame({
        "k":        k_arr,
        "bic":      bic_best,
        "aic":      aic_best,
        "loo_mean": loo_mean,
        "loo_std":  loo_std,
        "ari_mean": ari_mean,
    })
    df["bic_rank"] = df["bic"].rank(ascending=True).astype(int)
    df["aic_rank"] = df["aic"].rank(ascending=True).astype(int)
    valid_loo = df["loo_mean"].notna()
    df.loc[valid_loo, "loo_rank"] = (
        df.loc[valid_loo, "loo_mean"].rank(ascending=False).astype(int)
    )
    df["loo_rank"] = df["loo_rank"].fillna(np.nan)
    return df


def main(args):
    """Run BIC/AIC, LOO-CV, and ARI selection and write all outputs."""
    t_global = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    banner("HMM STATE SELECTION")
    print(f"  input_npz  : {args.input_npz}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  n_pcs      : {args.n_pcs}")
    print(f"  k range    : {args.k_min} – {args.k_max}  "
          f"({args.k_max - args.k_min + 1} values)")
    print(f"  n_inits    : {args.n_inits}")
    print(f"  n_iter     : {args.n_iter}  (Baum-Welch EM steps)")
    print(f"  cov_type   : {args.covariance_type}")
    print(f"  topology   : {args.topology}"
          + (f"  (max_advance={args.max_advance})"
             if args.topology == "left_to_right" else ""))

    section("Loading data")
    C, Z_scaled, window_ids, seed_ids = load_data(args.input_npz, args.n_pcs)
    seed_seqs = build_seed_sequences(Z_scaled, window_ids, seed_ids)
    seeds     = sorted(seed_seqs.keys())
    n_windows = seed_seqs[seeds[0]].shape[0]
    D         = seed_seqs[seeds[0]].shape[1]

    print(f"  Centroid matrix C   : {C.shape}  (windows × PCA dims)")
    print(f"  Individual Z_scaled : {Z_scaled.shape}")
    print(f"  Seeds               : {seeds}  ({len(seeds)} total)")
    print(f"  Windows per seed    : {n_windows}")
    print(f"  PCA dimensions (D)  : {D}")

    # Feasibility check shown for both full-data and LOO (n_seeds-1) phases so
    # degenerate k-values are flagged before any fitting begins.
    cov_label     = args.covariance_type
    n_obs_full    = n_windows * len(seeds)
    n_obs_loo     = n_windows * (len(seeds) - 1)
    print(f"\n  Parameter counts per k  (covariance_type={cov_label}):")
    print(f"  {'k':>4}  {'n_params':>10}  "
          f"{'n_obs_full':>12}  {'ok_full?':>10}  "
          f"{'n_obs_loo':>11}  {'ok_loo?':>9}")
    print(f"  {'-'*62}")
    for k_chk in range(args.k_min, args.k_max + 1):
        np_chk   = _n_params(k_chk, D, args.covariance_type,
                             args.topology, args.max_advance)
        ok_full  = "ok"         if np_chk < n_obs_full else "DEGENERATE"
        ok_loo   = "ok"         if np_chk < n_obs_loo  else "DEGENERATE"
        print(f"  {k_chk:>4}  {np_chk:>10}  "
              f"{n_obs_full:>12}  {ok_full:>10}  "
              f"{n_obs_loo:>11}  {ok_loo:>9}")
    print()

    k_range        = list(range(args.k_min, args.k_max + 1))
    n_k            = len(k_range)
    n_s            = len(seeds)
    n_pairs_per_k  = args.n_inits * (args.n_inits - 1) // 2

    print(f"\n  Expected HMM fits:")
    print(f"    BIC/AIC phase : {n_k * args.n_inits}  (models reused for ARI)")
    print(f"    LOO phase     : {n_k * n_s * args.n_inits}  "
          f"({n_s} folds × {args.n_inits} inits × {n_k} k-values)")
    print(f"    ARI phase     : 0 extra fits  (reuses BIC/AIC models)")
    print(f"    Total EM fits : {n_k * args.n_inits * (1 + n_s)}")
    print(f"\n  ARI pairs per k: {n_pairs_per_k}")

    section(f"BIC + AIC  —  fitting on all {n_s} seeds combined")
    t0 = time.time()
    bic_best, bic_all, aic_best, aic_all, fitted_models = run_bic_aic(
        seed_seqs, k_range, args.n_inits, args.n_iter, args.covariance_type,
        topology=args.topology, max_advance=args.max_advance,
        min_covar=args.min_covar,
    )
    print(f"\n  BIC/AIC phase complete in {elapsed(t0)}")

    section(f"LOO-CV  —  leave-one-seed-out ({n_s} folds)")
    t0 = time.time()
    loo_mean, loo_std, loo_all = run_loo_cv(
        seed_seqs, k_range, args.n_inits, args.n_iter, args.covariance_type,
        topology=args.topology, max_advance=args.max_advance,
        min_covar=args.min_covar, n_jobs=args.n_jobs,
    )
    print(f"\n  LOO-CV phase complete in {elapsed(t0)}")

    section(f"ARI stability  —  {n_pairs_per_k} Viterbi pairs per k")
    t0 = time.time()
    ari_mean, ari_all = run_ari_stability(
        seed_seqs, k_range, args.n_inits, args.n_iter, args.covariance_type,
        fitted_models,
        topology=args.topology, max_advance=args.max_advance,
    )
    print(f"\n  ARI phase complete in {elapsed(t0)}")

    section("Saving outputs")
    npz_path = os.path.join(args.output_dir, "hmm_scores.npz")
    np.savez(
        npz_path,
        k_range  = np.array(k_range),
        n_pcs    = np.int32(args.n_pcs),
        bic_best = bic_best,
        bic_all  = bic_all,
        aic_best = aic_best,
        aic_all  = aic_all,
        loo_mean = loo_mean,
        loo_std  = loo_std,
        loo_all  = loo_all,
        ari_mean = ari_mean,
        ari_all  = ari_all,
    )
    print(f"  hmm_scores.npz         → {npz_path}")

    df       = build_summary(k_range, bic_best, aic_best, loo_mean, loo_std, ari_mean)
    csv_path = os.path.join(args.output_dir, "hmm_select_summary.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  hmm_select_summary.csv → {csv_path}")

    plot_model_selection(
        k_range, bic_best, aic_best, loo_mean, loo_std,
        os.path.join(args.output_dir, "model_selection.png"),
    )
    plot_ari_stability(
        k_range, ari_all,
        os.path.join(args.output_dir, "ari_stability.png"),
    )
    plot_loo_per_fold(
        k_range, loo_all,
        os.path.join(args.output_dir, "loo_per_fold.png"),
    )

    k_arr   = np.array(k_range)
    k_bic   = k_arr[np.argmin(bic_best)]
    k_aic   = k_arr[np.argmin(aic_best)]
    valid   = np.isfinite(loo_mean)
    k_loo   = k_arr[valid][np.argmax(loo_mean[valid])] if valid.any() else "N/A"
    k_ari   = k_arr[np.argmax(ari_mean)]

    # Summarise how well the criteria agree on a single k.
    trio = {k_bic, k_aic, k_loo} if k_loo != "N/A" else {k_bic, k_aic}
    if len(trio) == 1:
        agree_str = "✓ ALL AGREE"
    elif k_bic == k_loo or k_bic == k_aic:
        agree_str = "~ PARTIAL AGREEMENT — check plots"
    else:
        agree_str = "✗ DISAGREE — check plots"

    banner("SELECTION SUMMARY")
    print(f"  BIC optimum    : k = {k_bic:2}  "
          f"(BIC = {bic_best[np.argmin(bic_best)]:.2f})")
    print(f"  AIC optimum    : k = {k_aic:2}  "
          f"(AIC = {aic_best[np.argmin(aic_best)]:.2f})")
    print(f"  LOO-CV optimum : k = {str(k_loo):>2}  "
          f"(mean held-out LL/obs = "
          f"{loo_mean[valid].max() if valid.any() else float('nan'):.4f})")
    print(f"  Most stable    : k = {k_ari:2}  "
          f"(mean ARI = {ari_mean.max():.4f})")
    print(f"  Consensus      : {agree_str}")
    print()

    hdr = (f"{'k':>4}  {'BIC':>12}  {'AIC':>12}  {'LOO mean':>10}  "
           f"{'LOO std':>8}  {'ARI mean':>9}  {'BIC#':>5}  {'AIC#':>5}  {'LOO#':>5}")
    print(f"  {hdr}")
    print(f"  {'─' * len(hdr)}")
    for _, row in df.iterrows():
        markers = []
        if row["k"] == k_bic: markers.append("BIC")
        if row["k"] == k_aic: markers.append("AIC")
        if row["k"] == k_loo: markers.append("LOO")
        marker = (" ← " + " + ".join(markers)) if markers else ""
        loo_r = f"{int(row['loo_rank'])}" if pd.notna(row["loo_rank"]) else "N/A"
        print(
            f"  {int(row['k']):>4}  "
            f"{row['bic']:>12.2f}  "
            f"{row['aic']:>12.2f}  "
            f"{row['loo_mean']:>10.4f}  "
            f"{row['loo_std']:>8.4f}  "
            f"{row['ari_mean']:>9.4f}  "
            f"{int(row['bic_rank']):>5}  "
            f"{int(row['aic_rank']):>5}  "
            f"{loo_r:>5}"
            f"{marker}"
        )

    print(f"\n  Total wall time: {elapsed(t_global)}")
    print(f"\n  → Make your choice, then run "
          f"src/decode_hmm.py --k <chosen_k> --n_pcs {args.n_pcs}")
    print("═" * 60)


def parse_args():
    """Parse CLI arguments for the selection run."""
    p = argparse.ArgumentParser(
        description="Select optimal number of HMM states via BIC, AIC, and LOO-CV."
    )
    p.add_argument("--input_npz",  default="data/hmm_weights/weights_pca.npz",
                   help="Path to weights_pca.npz from src/extract_weights_pca.py")
    p.add_argument("--output_dir", default="data/hmm_hmm",
                   help="Directory for outputs (created if absent)")
    p.add_argument("--n_pcs", type=int, required=True,
                   help="Number of leading principal components to feed the HMM. "
                        "extract_weights_pca.py saves all non-degenerate PCs; "
                        "this slices the leading n_pcs of Z_scaled. Must be "
                        "<= the number of PCs saved in the npz.")
    p.add_argument("--k_min",   type=int, default=2,
                   help="Minimum number of HMM states to evaluate")
    p.add_argument("--k_max",   type=int, default=10,
                   help="Maximum number of HMM states to evaluate")
    p.add_argument("--n_inits", type=int, default=10,
                   help="Random initialisations per k (more = more stable)")
    p.add_argument("--n_iter",  type=int, default=200,
                   help="Baum-Welch EM iterations per init")
    p.add_argument("--covariance_type", default="diag",
                   choices=["diag", "full", "tied", "spherical"],
                   help="HMM covariance type. 'diag' is strongly recommended "
                        "when n_obs << D*(D+1)/2 (the default). 'full' will "
                        "produce degenerate solutions with small datasets.")
    p.add_argument("--n_jobs", type=int, default=-1,
               help="Parallel workers for LOO-CV. -1 = all cores; set to "
                    "$SLURM_CPUS_PER_TASK in the job script to bind to the "
                    "allocation.")
    p.add_argument("--topology", default="ergodic",
                   choices=["ergodic", "left_to_right"],
                   help="HMM transition topology. 'left_to_right' (Bakis) forbids "
                        "backward transitions and starts in state 0 — appropriate "
                        "when the sequence is a monotonic data-time trajectory. "
                        "NOTE: this is a deviation from the paper's ergodic model.")
    p.add_argument("--max_advance", type=int, default=1,
                   help="Max state-index jump per step under left_to_right. "
                        "1 = strict no-skip; 2 = classic Bakis (allows one skip).")
    p.add_argument("--min_covar", type=float, default=1e-2,
                   help="Floor on diagonal covariances. Prevents variance "
                        "collapse / degenerate fits on short, low-dim sequences "
                        "(the cause of 'all inits failed' at large k). Must match "
                        "decode_hmm.py.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)