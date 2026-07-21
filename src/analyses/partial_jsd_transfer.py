"""
partial_jsd_transfer.py

Reviewer-requested control, generalized across datasets.

Does the within-vs-across-state advantage in cross-window transfer survive after
per-pair class-distribution divergence (JSD) is partialled out, alongside
temporal lag?

Two results could, in principle, be the same effect:
  (1) within_across_states.py  — within-state pairs transfer better than
      across-state pairs, even after temporal lag is controlled for.
  (2) state_pair_correlation.py — transfer falls as the JSD between two states'
      class distributions rises.
If "sharing a state" predicts transfer only because same-state windows have
similar class distributions, then holding JSD (and lag) fixed should make the
within/across indicator add nothing. This script tests exactly that, at the
window-pair level.

Key design choice: JSD is computed PER WINDOW PAIR, between each window's own
class distribution (from the manifest cls_* counts), not between two states'
aggregated distributions. The state-level measure would be identically zero for
every within-state pair (perfect collinearity with the indicator); the
per-window measure is continuous and varies both within and across states, so it
can genuinely compete with the indicator.

Two analyses
------------
  A. Combined model  f1 ~ b0 + b_jsd*JSD + b_lag*lag + b_state*same_state,
     with b_state tested by a lag-stratified Freedman-Lane partial permutation,
     plus cluster-robust SEs on the unordered pair.
  B. Two-stage residual model: residualize f1 on JSD+lag, then test same_state
     on the residuals via (i) a free label shuffle and (ii) the paper's
     harmonic-weighted, lag-stratified distance-conditioned gap.

Outputs  (--output_dir)
-----------------------
  window_pair_table.csv           one row per ordered (i, j) pair
  partial_jsd_results.csv         coefficient table + permutation p-values
  partial_jsd_cluster_robust.csv  (if statsmodels available)
  partial_jsd_null.npz            null distributions
  plot_partial_residual.png       JSD+lag-residualized F1, within vs across
  plot_indicator_null.png         Freedman-Lane null for beta_state

Usage
-----
  python -m src.hmm.partial_jsd_transfer --dataset fakeddit --k 7
  python -m src.hmm.partial_jsd_transfer --dataset yelp --k 5 \
      --output_dir data/partial_jsd/yelp/k5 --n_permutations 10000

Requirements: numpy, pandas, matplotlib, tqdm
              statsmodels (optional — only for the cluster-robust SE table)
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
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.datasets.registry import get_spec

warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_args():
    """Parse CLI arguments and fill in dataset-derived default paths."""
    p = argparse.ArgumentParser(
        description="Partial-out per-pair class-distribution divergence (JSD) and "
                    "temporal lag from within-vs-across-state transfer, "
                    "dataset-generalized.")
    p.add_argument("--dataset", required=True, choices=["fakeddit", "yelp"],
                   help="Which dataset spec to use (drives stem + default paths).")
    p.add_argument("--k", type=int, required=True,
                   help="HMM state count from the decode step (used for default "
                        "paths; authoritative k is read from the decode npz).")
    p.add_argument("--decode_npz", default=None,
                   help="HMM decode file. Defaults to "
                        "data/hmm_hmm/<dataset>/<dataset>_decode_k<k>.npz")
    p.add_argument("--f1_npz", default=None,
                   help="Col-centered cross-window F1. Defaults to "
                        "data/hmm_perf/<dataset>/cross_window_f1_colcentered.npz")
    p.add_argument("--manifest", default=None,
                   help="Window manifest with cls_* columns. Defaults to "
                        "data/splits/hmm_windows/<dataset>/"
                        "<dataset>_windows_manifest.csv")
    p.add_argument("--output_dir", default=None,
                   help="Defaults to data/partial_jsd/<dataset>/k<k>")
    p.add_argument("--n_permutations", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    ds = args.dataset
    if args.decode_npz is None:
        args.decode_npz = f"data/hmm_hmm/{ds}/{ds}_decode_k{args.k}.npz"
    if args.f1_npz is None:
        args.f1_npz = f"data/hmm_perf/{ds}/cross_window_f1_colcentered.npz"
    if args.manifest is None:
        args.manifest = f"data/splits/hmm_windows/{ds}/{ds}_windows_manifest.csv"
    if args.output_dir is None:
        args.output_dir = f"data/partial_jsd/{ds}/k{args.k}"
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


def load_inputs(decode_npz, f1_npz, manifest_path):
    """Load the decode, F1 matrix, and aligned manifest with class columns."""
    dec = np.load(decode_npz, allow_pickle=True)
    state_seq = dec["state_seq"].astype(int)
    window_ids = dec["window_ids"].astype(int)
    k = int(dec["k"]) if "k" in dec else int(state_seq.max() + 1)

    f1_data = np.load(f1_npz, allow_pickle=True)
    f1_matrix = f1_data["f1_matrix_colcentered"].astype(float)

    N = len(state_seq)
    assert f1_matrix.shape == (N, N), (
        f"state_seq length {N} != f1_matrix shape {f1_matrix.shape}")

    # Align manifest rows to the decode window order, discover class columns.
    manifest = pd.read_csv(manifest_path)
    manifest = manifest.set_index("window_local_idx").loc[window_ids].reset_index()
    cls_cols = discover_cls_cols(manifest)
    for c in cls_cols:
        manifest[c] = manifest[c].fillna(0).astype(int)

    print(f"Loaded: N={N} windows, k={k} states, {len(cls_cols)} classes {cls_cols}")
    print(f"State sequence: {state_seq}")
    return state_seq, window_ids, k, f1_matrix, manifest, cls_cols


def window_class_props(manifest: pd.DataFrame, cls_cols: list[str]) -> np.ndarray:
    """Return (N, n_classes) per-window class proportions (per window, not state)."""
    counts = manifest[cls_cols].values.astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    return counts / np.where(totals > 0, totals, 1.0)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Return the Jensen-Shannon divergence (base-2)."""
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def build_window_pair_table(state_seq: np.ndarray,
                            f1_matrix: np.ndarray,
                            class_props: np.ndarray) -> pd.DataFrame:
    """Build one row per ordered off-diagonal window pair (i, j), i != j.

    Columns: train_win, test_win, lag, same_state, state_i, state_j, jsd,
    pair_id, f1. jsd is between the two windows' own class distributions.
    """
    N = len(state_seq)
    rows = []

    def pair_id(i, j):
        a, b = (i, j) if i < j else (j, i)
        return a * N + b

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            v = f1_matrix[i, j]
            if np.isnan(v):
                continue
            rows.append(dict(
                train_win=i,
                test_win=j,
                lag=abs(i - j),
                same_state=int(state_seq[i] == state_seq[j]),
                state_i=int(state_seq[i]),
                state_j=int(state_seq[j]),
                jsd=js_divergence(class_props[i], class_props[j]),
                pair_id=pair_id(i, j),
                f1=float(v),
            ))

    df = pd.DataFrame(rows)
    df["jsd"] = df["jsd"].round(6)
    return df


def design_matrix(df: pd.DataFrame, cols) -> np.ndarray:
    """Build a [1, col1, col2, ...] design matrix from the named columns."""
    return np.column_stack([np.ones(len(df))] +
                           [df[c].values.astype(float) for c in cols])


def ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the OLS coefficient vector for X, y via least squares."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def freedman_lane_indicator_test(df: pd.DataFrame,
                                 n_perm: int,
                                 rng: np.random.Generator):
    """Partial-test the same_state indicator in f1 ~ JSD + lag + same_state.

    Uses the Freedman-Lane scheme with residual permutation stratified by lag.
    Returns a dict with the observed coef, full-model coefs, null array, and
    one- and two-sided p-values.
    """
    y = df["f1"].values.astype(float)

    X_full = design_matrix(df, ["jsd", "lag", "same_state"])
    beta_full = ols_beta(X_full, y)
    obs_coef = float(beta_full[3])   # [intercept, jsd, lag, same_state]

    X_red = design_matrix(df, ["jsd", "lag"])
    beta_red = ols_beta(X_red, y)
    fitted_red = X_red @ beta_red
    resid_red = y - fitted_red

    lag_vals = df["lag"].values
    strata = {d: np.where(lag_vals == d)[0] for d in np.unique(lag_vals)}

    null_coefs = np.empty(n_perm)
    for b in tqdm(range(n_perm), desc="Freedman-Lane (indicator)"):
        # Permute the reduced-model residuals within each lag stratum, so the
        # null preserves the lag structure while breaking the same_state signal.
        resid_perm = resid_red.copy()
        for idx in strata.values():
            if idx.size > 1:
                resid_perm[idx] = rng.permutation(resid_red[idx])
        y_star = fitted_red + resid_perm
        beta_star = ols_beta(X_full, y_star)
        null_coefs[b] = beta_star[3]

    p_one = float((null_coefs >= obs_coef).mean())
    p_two = float((np.abs(null_coefs) >= abs(obs_coef)).mean())

    return {"obs_coef": obs_coef, "beta_full": beta_full,
            "null_coefs": null_coefs, "p_one": p_one, "p_two": p_two}


def cluster_robust_table(df: pd.DataFrame):
    """Fit f1 ~ JSD + lag + same_state with SEs clustered on the unordered pair.

    Returns a DataFrame, or None if statsmodels is not installed (the
    permutation test is the headline inference regardless).
    """
    try:
        import statsmodels.api as sm
    except Exception as e:
        print(f"  [info] statsmodels unavailable ({e}); "
              f"skipping cluster-robust SE table.")
        return None

    X = sm.add_constant(df[["jsd", "lag", "same_state"]].astype(float))
    y = df["f1"].astype(float)
    model = sm.OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": df["pair_id"].values})
    return pd.DataFrame({
        "term": ["intercept", "jsd", "lag", "same_state"],
        "coef": model.params.values,
        "se_clustered": model.bse.values,
        "t": model.tvalues.values,
        "p_param": model.pvalues.values,
    })


def residualize_on_jsd_and_lag(df: pd.DataFrame) -> np.ndarray:
    """Return F1 residuals after regressing f1 ~ JSD + lag (step 1)."""
    y = df["f1"].values.astype(float)
    X_red = design_matrix(df, ["jsd", "lag"])
    beta_red = ols_beta(X_red, y)
    return y - X_red @ beta_red


def simple_residual_gap_test(resid: np.ndarray,
                             same_state: np.ndarray,
                             n_perm: int,
                             rng: np.random.Generator):
    """Step 2, simple form: within-minus-across mean residual, free-shuffle null.

    Lag has already been removed in step 1, so the label shuffle is unrestricted.
    """
    w = resid[same_state == 1]
    a = resid[same_state == 0]
    obs_gap = float(w.mean() - a.mean())

    null = np.empty(n_perm)
    for b in tqdm(range(n_perm), desc="Residual gap (free shuffle)"):
        perm = rng.permutation(same_state)
        null[b] = resid[perm == 1].mean() - resid[perm == 0].mean()

    p_one = float((null >= obs_gap).mean())
    return {"obs_gap": obs_gap, "null": null, "p_one": p_one,
            "n_within": int(same_state.sum()),
            "n_across": int((same_state == 0).sum())}


def distance_conditioned_gap(value: np.ndarray,
                             same_state: np.ndarray,
                             lag: np.ndarray) -> float:
    """Return the harmonic-count-weighted, lag-stratified within-minus-across gap.

    Matches the Figure-7 statistic in within_across_states.py.
    """
    gaps, weights = [], []
    for d in np.unique(lag):
        m = lag == d
        w = value[m & (same_state == 1)]
        a = value[m & (same_state == 0)]
        if w.size == 0 or a.size == 0:
            continue
        gaps.append(w.mean() - a.mean())
        weights.append(2 * w.size * a.size / (w.size + a.size))
    if not gaps:
        return 0.0
    return float(np.average(gaps, weights=np.array(weights)))


def distance_conditioned_residual_test(resid: np.ndarray,
                                       same_state: np.ndarray,
                                       lag: np.ndarray,
                                       n_perm: int,
                                       rng: np.random.Generator):
    """Step 2, paper-faithful form: distance-conditioned label shuffle on residuals."""
    obs = distance_conditioned_gap(resid, same_state, lag)

    strata, weights = {}, {}
    for d in np.unique(lag):
        idx = np.where(lag == d)[0]
        ss = same_state[idx]
        nw, na = int(ss.sum()), int((ss == 0).sum())
        if nw > 0 and na > 0:
            strata[d] = idx
            weights[d] = 2 * nw * na / (nw + na)
    total_w = sum(weights.values())

    null = np.empty(n_perm)
    for b in tqdm(range(n_perm), desc="Residual gap (lag-stratified)"):
        # Shuffle the same_state labels within each lag stratum and re-form the
        # weighted gap, matching the observed statistic's construction.
        num = 0.0
        for d, idx in strata.items():
            labels = same_state[idx].copy()
            rng.shuffle(labels)
            vals = resid[idx]
            num += weights[d] * (vals[labels == 1].mean() - vals[labels == 0].mean())
        null[b] = num / total_w if total_w else 0.0

    p_one = float((null >= obs).mean())
    return {"obs_gap": obs, "null": null, "p_one": p_one}


def rng_jitter(n, center, width=0.12, seed=0):
    """Return n jittered x-coordinates centred on `center` for strip plotting."""
    r = np.random.default_rng(seed + center)
    return center + r.uniform(-width, width, size=n)


def plot_partial_residual(resid, same_state, output_path):
    """Plot JSD+lag-residualized F1 for within- vs across-state pairs."""
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [resid[same_state == 0], resid[same_state == 1]]
    ax.boxplot(groups, labels=["Across-state", "Within-state"],
               showmeans=True, widths=0.5)
    for i, g in enumerate(groups, start=1):
        x = rng_jitter(len(g), i)
        ax.scatter(x, g, s=6, alpha=0.25,
                   color="#DC2626" if i == 2 else "#2563EB")
    ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("Macro-F1 Residual (JSD and Lag Removed)")
    ax.set_title("Within- vs. Across-State Transfer, Controlling\n"
                 "for Class Divergence and Temporal Lag")
    ax.set_xlabel("State Membership")
    ax.grid(True, axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_indicator_null(null_coefs, obs_coef, p_one, output_path):
    """Plot the Freedman-Lane null distribution for the same_state coefficient."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_coefs, bins=60, color="#94A3B8", edgecolor="white",
            lw=0.3, alpha=0.85, label="Freedman-Lane null")
    ax.axvline(obs_coef, color="#DC2626", lw=2.5,
               label=f"Observed coefficient = {obs_coef:.4f}  (p = {p_one:.4f})")
    pct95 = np.percentile(null_coefs, 95)
    ax.axvline(pct95, color="black", lw=1.2, ls="--", alpha=0.7,
               label=f"Null 95th percentile = {pct95:.4f}")
    ax.set_xlabel("Same-State Coefficient (Macro-F1, JSD and Lag Held Fixed)")
    ax.set_ylabel("Permutation Count")
    ax.set_title("Freedman\u2013Lane Test of Latent State Membership\n"
                 "After Partialling Out Class Divergence and Lag")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    """Run the partial-JSD control analyses and write outputs and plots."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Validate the dataset name against the registry (fail fast on typos).
    spec = get_spec(args.dataset)

    print("=" * 64)
    print(f"Partialling JSD out of cross-window transfer  [{spec.name}]")
    print("(window-pair level)")
    print("=" * 64)

    state_seq, window_ids, k, f1_matrix, manifest, cls_cols = load_inputs(
        args.decode_npz, args.f1_npz, args.manifest)

    class_props = window_class_props(manifest, cls_cols)

    print("\nBuilding window-pair table ...")
    df = build_window_pair_table(state_seq, f1_matrix, class_props)
    print(f"  {len(df):,} ordered off-diagonal observations (per-window-mean, "
          f"col-centered)")
    print(f"  within-state: {int(df['same_state'].sum()):,}   "
          f"across-state: {int((df['same_state']==0).sum()):,}")
    print(f"  JSD range: [{df['jsd'].min():.4f}, {df['jsd'].max():.4f}]   "
          f"lag range: [{df['lag'].min()}, {df['lag'].max()}]")

    pair_path = os.path.join(args.output_dir, "window_pair_table.csv")
    df.to_csv(pair_path, index=False)
    print(f"  Saved: {pair_path}")

    print(f"\n[A] Combined model  f1 ~ JSD + lag + same_state")
    fl = freedman_lane_indicator_test(df, args.n_permutations, rng)
    b = fl["beta_full"]
    print(f"      intercept   = {b[0]:+.4f}")
    print(f"      beta_JSD    = {b[1]:+.4f}")
    print(f"      beta_lag    = {b[2]:+.4f}")
    print(f"      beta_state  = {b[3]:+.4f}   "
          f"(within-state F1 advantage, JSD & lag held fixed)")
    print(f"      Freedman-Lane p (one-sided, within>across) = {fl['p_one']:.4f}")
    print(f"      Freedman-Lane p (two-sided)                = {fl['p_two']:.4f}")

    crt = cluster_robust_table(df)
    if crt is not None:
        print("\n      Cluster-robust (clustered on unordered window pair):")
        print(crt.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print(f"\n[B] Two-stage: residualize f1 on JSD + lag, then test same_state")
    resid = residualize_on_jsd_and_lag(df)
    same_state = df["same_state"].values.astype(int)
    lag = df["lag"].values.astype(int)

    simple = simple_residual_gap_test(resid, same_state, args.n_permutations, rng)
    print(f"      Residual within\u2212across gap (free shuffle)        = "
          f"{simple['obs_gap']:+.4f}   p = {simple['p_one']:.4f}")

    dcr = distance_conditioned_residual_test(
        resid, same_state, lag, args.n_permutations, rng)
    print(f"      Residual gap (lag-stratified, harmonic-weighted) = "
          f"{dcr['obs_gap']:+.4f}   p = {dcr['p_one']:.4f}")

    rows = [
        {"analysis": "combined_model", "term": "beta_JSD",
         "estimate": b[1], "p_one_sided": np.nan, "p_two_sided": np.nan},
        {"analysis": "combined_model", "term": "beta_lag",
         "estimate": b[2], "p_one_sided": np.nan, "p_two_sided": np.nan},
        {"analysis": "combined_model", "term": "beta_same_state",
         "estimate": b[3], "p_one_sided": fl["p_one"], "p_two_sided": fl["p_two"]},
        {"analysis": "two_stage_residual_free", "term": "within_minus_across_gap",
         "estimate": simple["obs_gap"], "p_one_sided": simple["p_one"],
         "p_two_sided": np.nan},
        {"analysis": "two_stage_residual_lagstratified",
         "term": "distance_conditioned_gap",
         "estimate": dcr["obs_gap"], "p_one_sided": dcr["p_one"],
         "p_two_sided": np.nan},
    ]
    res_df = pd.DataFrame(rows)
    res_path = os.path.join(args.output_dir, "partial_jsd_results.csv")
    res_df.to_csv(res_path, index=False)
    print(f"\n  Saved: {res_path}")
    if crt is not None:
        crt.to_csv(os.path.join(args.output_dir,
                                "partial_jsd_cluster_robust.csv"), index=False)

    np.savez(
        os.path.join(args.output_dir, "partial_jsd_null.npz"),
        fl_null=fl["null_coefs"], fl_obs=fl["obs_coef"],
        fl_p_one=fl["p_one"], fl_p_two=fl["p_two"],
        resid_free_null=simple["null"], resid_free_obs=simple["obs_gap"],
        resid_lag_null=dcr["null"], resid_lag_obs=dcr["obs_gap"],
        beta_full=b,
    )
    print(f"  Saved: {os.path.join(args.output_dir, 'partial_jsd_null.npz')}")

    if not args.no_plots:
        print("\nGenerating plots ...")
        plot_partial_residual(
            resid, same_state,
            os.path.join(args.output_dir, "plot_partial_residual.png"))
        plot_indicator_null(
            fl["null_coefs"], fl["obs_coef"], fl["p_one"],
            os.path.join(args.output_dir, "plot_indicator_null.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()