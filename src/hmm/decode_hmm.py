"""
decode_hmm.py

Final HMM Viterbi decode at a chosen k, generalized across datasets and
supporting a left-to-right (Bakis) topology.

Procedure:
  1. Load per-window, per-seed PCA vectors Z_scaled from weights_pca.npz and
     slice to the leading --n_pcs components.
  2. Build per-seed sequences, then a single centroid sequence by averaging PCA
     vectors across seeds at each window position.
  3. Fit GaussianHMM(n_components=k) with n_inits random inits on the centroid
     sequence under the requested topology; keep the best-LL model.
  4. Decode the Viterbi state sequence over the centroid sequence directly.
  5. Save <dataset>_decode_k{k}.npz (+ pickle) with state_seq, window_ids,
     log_likelihood, transition_matrix, means, covars, startprob, and the
     topology metadata.
  6. Produce the three standard plots (timeline, strip, transition matrix).

The number of PCs is chosen here via --n_pcs and must match whatever --n_pcs
was used in select_hmm_states.py. Under topology="left_to_right" an
upper-triangular transmat and a state-0 start are seeded, 's'/'t' are dropped
from init_params so the seed survives EM, and 't' is kept in params so the
surviving upper-triangular probabilities are re-estimated.

Usage:
    # Fakeddit, 5 states, 5 PCs, left-to-right:
    python -m src.hmm.decode_hmm --dataset fakeddit --k 5 --n_pcs 5 \
        --topology left_to_right \
        --input_npz data/hmm_weights/fakeddit/weights_pca.npz \
        --output_dir data/hmm_hmm/fakeddit \
        --manifest data/splits/hmm_windows/fakeddit/fakeddit_windows_manifest.csv

    # Yelp, 18 states, 3 PCs, left-to-right:
    python -m src.hmm.decode_hmm --dataset yelp --k 18 --n_pcs 3 \
        --topology left_to_right \
        --input_npz data/hmm_weights/yelp/weights_pca.npz \
        --output_dir data/hmm_hmm/yelp \
        --manifest data/splits/hmm_windows/yelp/yelp_windows_manifest.csv

    # Defaults derive per-dataset paths; minimal Fakeddit call:
    python -m src.hmm.decode_hmm --dataset fakeddit --k 5 --n_pcs 5 --topology left_to_right

Requirements: hmmlearn, numpy, matplotlib, pandas
"""

import os
import sys
import time
import argparse
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from hmmlearn import hmm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.datasets.registry import get_spec

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*KMeans.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")

# tab10 palette, consistent with selection plots.
TAB10 = plt.cm.tab10.colors


def elapsed(t0):
    """Return the time since t0 formatted as MM:SS.ss."""
    s = time.time() - t0
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


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
    print(f"    Using {n_pcs} of {n_avail} available PCs.")
    return C, Z_scaled, window_ids, seed_ids


def build_seed_sequences(Z_scaled, window_ids, seed_ids):
    """Group per-(window, seed) PCA vectors into per-seed sequences sorted by window."""
    seeds     = sorted(set(seed_ids))
    seed_seqs = {}
    for s in seeds:
        mask  = seed_ids == s
        order = np.argsort(window_ids[mask])
        seed_seqs[s] = Z_scaled[mask][order]

    # All seeds must share the same window ordering for the centroid to be valid.
    ref_wins = None
    for s in seeds:
        mask = seed_ids == s
        wins = np.sort(window_ids[mask])
        if ref_wins is None:
            ref_wins = wins
        else:
            assert np.array_equal(wins, ref_wins), \
                f"Seed {s} has different window IDs than seed {seeds[0]}"
    return seed_seqs, ref_wins


def compute_centroid_sequence(seed_seqs, seeds):
    """Average PCA vectors across seeds at each window position into (n_windows, D)."""
    stacked = np.stack([seed_seqs[s] for s in seeds], axis=0)
    return stacked.mean(axis=0)


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


def fit_best(centroid_seq, k, n_inits, n_iter, covariance_type="diag",
             topology="ergodic", max_advance=1, min_covar=1e-2):
    """Fit n_inits HMMs on the centroid sequence and return the best (model, LL)."""
    X       = centroid_seq
    lengths = [X.shape[0]]
    best_model = None
    best_ll    = -np.inf
    first_exc  = None

    for seed in range(n_inits):
        model = make_hmm(k, n_iter, random_state=seed * 100 + k,
                         covariance_type=covariance_type,
                         topology=topology, max_advance=max_advance,
                         min_covar=min_covar)
        # Skip inits that fail to fit/score; keep the first exception to report.
        try:
            model.fit(X, lengths)
            ll = model.score(X, lengths)
        except Exception as e:
            if first_exc is None:
                first_exc = e
            continue
        if ll > best_ll:
            best_ll    = ll
            best_model = model

    if best_model is None:
        raise RuntimeError(
            f"All {n_inits} HMM inits failed. "
            f"First exception: {type(first_exc).__name__}: {first_exc}"
        )
    return best_model, best_ll


def decode_centroid(model, centroid_seq):
    """Run Viterbi on the centroid sequence and return an int32 state sequence."""
    _, state_seq = model.decode(
        centroid_seq,
        lengths=[centroid_seq.shape[0]],
        algorithm="viterbi",
    )
    return state_seq.astype(np.int32)


def verify_left_to_right(trans_matrix, max_advance):
    """Check no probability mass sits below the diagonal or beyond the band.

    Returns (ok, max_violation).
    """
    k = trans_matrix.shape[0]
    viol = 0.0
    for i in range(k):
        for j in range(k):
            if j < i or j > i + max_advance:
                viol = max(viol, trans_matrix[i, j])
    return viol < 1e-8, viol


def load_window_dates(manifest_path, window_ids, window_days):
    """Resolve per-window calendar dates from the manifest, else synthesize them.

    The synthetic fallback spaces windows by window_days (from the dataset
    spec), so it stays dataset-correct when the manifest is absent or
    unparsable.
    """
    if manifest_path and os.path.isfile(manifest_path):
        try:
            df = pd.read_csv(manifest_path)
            df.columns = [c.lower().strip() for c in df.columns]
            id_col    = next(c for c in df.columns if "window" in c and "id" in c)
            date_cols = [c for c in df.columns if "date" in c or "start" in c]
            end_cols  = [c for c in df.columns if "end" in c]
            df = df.set_index(id_col)
            starts, ends, mids = [], [], []
            for wid in window_ids:
                if wid in df.index:
                    s = pd.Timestamp(df.loc[wid, date_cols[0]])
                    e = (pd.Timestamp(df.loc[wid, end_cols[0]])
                         if end_cols
                         else s + pd.Timedelta(days=window_days))
                else:
                    # Window absent from manifest: fall back to a synthetic date.
                    s = pd.Timestamp("2013-01-01") + pd.Timedelta(days=int(wid) * window_days)
                    e = s + pd.Timedelta(days=window_days)
                starts.append(s)
                ends.append(e)
                mids.append(s + (e - s) / 2)
            return (
                np.array(starts, dtype="datetime64[D]"),
                np.array(mids,   dtype="datetime64[D]"),
                np.array(ends,   dtype="datetime64[D]"),
            )
        except Exception as exc:
            print(f"  [warn] Could not parse manifest ({exc}); using synthetic dates.")

    origin = np.datetime64("2013-01-01", "D")
    starts = np.array([origin + np.timedelta64(int(w) * window_days, "D") for w in window_ids])
    ends   = starts + np.timedelta64(window_days, "D")
    mids   = starts + np.timedelta64(window_days // 2, "D")
    return starts, mids, ends


def _contiguous_runs(seq):
    """Run-length encode seq into a list of (value, start_idx, end_idx) tuples."""
    runs = []
    i = 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        runs.append((seq[i], i, j - 1))
        i = j
    return runs


def plot_state_timeline(state_seq, window_ids, start_dates, end_dates, k, out_path):
    """Single-row date-axis timeline of the centroid Viterbi state sequence."""
    n_win = len(window_ids)
    fig, ax = plt.subplots(figsize=(14, 2.2))

    # Mark boundaries where the decoded state changes between windows.
    boundary_dates = []
    for idx in np.where(np.diff(state_seq) != 0)[0]:
        boundary_dates.append(
            mdates.date2num(pd.Timestamp(end_dates[idx]).to_pydatetime())
        )

    for i in range(n_win):
        s_dt  = pd.Timestamp(start_dates[i]).to_pydatetime()
        e_dt  = pd.Timestamp(end_dates[i]).to_pydatetime()
        width = (e_dt - s_dt).days
        ax.barh(y=0, width=width,
                left=mdates.date2num(s_dt),
                height=0.8,
                color=TAB10[state_seq[i] % 10],
                alpha=0.85, linewidth=0)

    for bd in boundary_dates:
        ax.axvline(bd, color="black", linewidth=1.2,
                   linestyle="--", alpha=0.7, zorder=5)

    ax.set_yticks([0])
    ax.set_yticklabels(["centroid"], fontsize=9)
    ax.set_ylim(-0.6, 0.6)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.set_xlabel("Date", fontsize=11)

    patches = [mpatches.Patch(color=TAB10[s % 10], label=f"State {s}")
               for s in range(k)]
    ax.legend(handles=patches, loc="upper right", fontsize=8,
              framealpha=0.9, title="HMM State", title_fontsize=8,
              ncol=min(k, 5))
    fig.suptitle(
        f"HMM State Timeline  (k={k}, Viterbi on seed-centroid sequence)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_state_summary_strip(state_seq, window_ids, mid_dates, k, out_path):
    """Compact scatter/step strip of the state sequence vs chronological index."""
    fig, ax = plt.subplots(figsize=(14, 3))
    x_vals  = np.arange(len(window_ids))
    colors  = [TAB10[s % 10] for s in state_seq]

    ax.scatter(x_vals, state_seq, c=colors, s=80, zorder=3,
               edgecolors="white", linewidths=0.5)
    ax.step(x_vals, state_seq, where="mid",
            color="gray", linewidth=0.8, alpha=0.6, zorder=2)

    for idx in np.where(np.diff(state_seq) != 0)[0]:
        ax.axvline(idx + 0.5, color="black", linewidth=1.2,
                   linestyle="--", alpha=0.6)

    # Thin the x tick labels to at most ~8 dates for readability.
    tick_step = max(1, len(window_ids) // 8)
    tick_idxs = list(range(0, len(window_ids), tick_step))
    ax.set_xticks(tick_idxs)
    ax.set_xticklabels(
        [pd.Timestamp(mid_dates[i]).strftime("%b %Y") for i in tick_idxs],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"State {i}" for i in range(k)], fontsize=9)
    ax.set_ylabel("HMM State", fontsize=11)
    ax.set_xlabel("Window (chronological)", fontsize=11)
    ax.set_title(f"Viterbi State Sequence — seed centroid  (k={k})", fontsize=12)
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_xlim(-0.5, len(window_ids) - 0.5)
    ax.set_ylim(-0.5, k - 0.5)

    patches = [mpatches.Patch(color=TAB10[s % 10], label=f"State {s}")
               for s in range(k)]
    ax.legend(handles=patches, loc="upper right", fontsize=8, ncol=min(k, 10),
              framealpha=0.9, title="HMM State", title_fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_transition_matrix(trans_matrix, k, out_path):
    """Heatmap of the HMM transition matrix with the diagonal outlined."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(trans_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Transition probability")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([f"S{i}" for i in range(k)])
    ax.set_yticklabels([f"S{i}" for i in range(k)])
    ax.set_xlabel("To state", fontsize=11)
    ax.set_ylabel("From state", fontsize=11)
    ax.set_title(f"HMM Transition Matrix  (k={k})", fontsize=12)
    for i in range(k):
        # Annotate only cells with non-negligible probability mass.
        for j in range(k):
            val = trans_matrix[i, j]
            if val >= 0.005:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if val > 0.6 else "black")
        rect = mpatches.FancyBboxPatch(
            (i - 0.5, i - 0.5), 1, 1,
            boxstyle="square,pad=0",
            linewidth=2, edgecolor="red", facecolor="none",
        )
        ax.add_patch(rect)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main(args):
    """Run the full fit/decode/plot pipeline for one dataset and k."""
    t_global = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    spec = get_spec(args.dataset)
    stem = spec.name
    window_days = args.window_days or spec.window.window_days

    print("=" * 60)
    print(f"  decode_hmm.py  —  dataset={stem}  k={args.k}  n_pcs={args.n_pcs}  "
          f"topology={args.topology}")
    if args.topology == "left_to_right":
        print(f"  (max_advance={args.max_advance})")
    print("=" * 60)

    # Load Z_scaled and build the centroid sequence.
    print(f"\n[1] Loading Z_scaled from: {args.input_npz}")
    C, Z_scaled, window_ids, seed_ids = load_data(args.input_npz, args.n_pcs)
    seed_seqs, sorted_window_ids = build_seed_sequences(Z_scaled, window_ids, seed_ids)
    seeds = sorted(seed_seqs.keys())

    n_windows = seed_seqs[seeds[0]].shape[0]
    D         = seed_seqs[seeds[0]].shape[1]
    n_seeds   = len(seeds)

    centroid_seq = compute_centroid_sequence(seed_seqs, seeds)

    print(f"    Z_scaled shape   : {Z_scaled.shape}  (obs × PCA dims)")
    print(f"    Seeds            : {seeds}  ({n_seeds} total)")
    print(f"    Windows          : {n_windows}")
    print(f"    PCA dims (D)      : {D}")
    print(f"    Centroid shape   : {centroid_seq.shape}  (mean across {n_seeds} seeds)")

    # A strict Bakis chain cannot visit more states than there are time steps.
    if args.topology == "left_to_right" and args.k > n_windows:
        print(f"\n  [warn] k={args.k} > n_windows={n_windows}. A strict "
              f"left-to-right chain cannot visit more states than there are "
              f"time steps; some states will be unused.")

    # Degeneracy check: diag GaussianHMM free params vs available observations.
    emission_params = 2 * args.k * D
    if emission_params > n_windows:
        print(f"\n  [note] {emission_params} emission params (2*k*D) exceed "
              f"n_windows={n_windows}. The min_covar floor (={args.min_covar}) "
              f"prevents variance collapse, but individual state emissions are "
              f"estimated from few windows — expect wide covariances.")

    # Fit the HMM on the centroid sequence.
    print(f"\n[2] Fitting GaussianHMM(k={args.k}, cov={args.covariance_type}, "
          f"topology={args.topology}) with {args.n_inits} random inits …")
    t_fit = time.time()
    best_model, best_ll = fit_best(
        centroid_seq, args.k, args.n_inits, args.n_iter,
        args.covariance_type, topology=args.topology, max_advance=args.max_advance,
        min_covar=args.min_covar,
    )
    print(f"    Best log-likelihood : {best_ll:.4f}  (wall: {elapsed(t_fit)})")

    # Viterbi decode on the centroid sequence.
    print(f"\n[3] Viterbi decoding on centroid sequence …")
    state_seq = decode_centroid(best_model, centroid_seq)
    print(f"    state_seq : {state_seq.tolist()}")

    # Under left_to_right the sequence must be monotonically non-decreasing.
    if args.topology == "left_to_right":
        mono = bool(np.all(np.diff(state_seq) >= 0))
        ok, viol = verify_left_to_right(best_model.transmat_, args.max_advance)
        print(f"    monotonic non-decreasing : {mono}")
        print(f"    transmat respects band   : {ok}  (max off-band prob={viol:.2e})")
        if not mono:
            print("    [warn] decoded sequence is NOT monotonic — check the "
                  "seed transmat / init_params in make_hmm.")

    runs = _contiguous_runs(state_seq)
    print(f"\n    Run-length encoded ({len(runs)} runs):")
    for state, i0, i1 in runs:
        print(f"      State {state:2d}  windows "
              f"{sorted_window_ids[i0]:03d}–{sorted_window_ids[i1]:03d}  "
              f"({i1 - i0 + 1} windows)")

    # Extract model parameters for saving and plotting.
    trans_matrix = best_model.transmat_
    means        = best_model.means_
    startprob    = best_model.startprob_
    covars       = best_model.covars_

    print(f"\n    Transition matrix (rounded):")
    for i in range(args.k):
        row = "  ".join(f"{p:.3f}" for p in trans_matrix[i])
        print(f"      S{i:2d}: [{row}]")

    # Save .npz and pickle.
    out_npz = os.path.join(args.output_dir, f"{stem}_decode_k{args.k}.npz")
    np.savez(
        out_npz,
        state_seq         = state_seq,
        centroid_seq      = centroid_seq,
        window_ids        = sorted_window_ids,
        log_likelihood    = np.float64(best_ll),
        transition_matrix = trans_matrix,
        means             = means,
        covars            = covars,
        startprob         = startprob,
        k                 = np.int32(args.k),
        n_pcs             = np.int32(args.n_pcs),
        covariance_type   = np.str_(args.covariance_type),
        dataset           = np.str_(stem),
        topology          = np.str_(args.topology),
        max_advance       = np.int32(args.max_advance),
    )
    print(f"\n[4] Saved: {out_npz}")

    out_pkl = os.path.join(args.output_dir, f"{stem}_hmm_k{args.k}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(best_model, f)
    print(f"    Saved: {out_pkl}")

    # Resolve calendar dates for the plots.
    start_dates, mid_dates, end_dates = load_window_dates(
        args.manifest, sorted_window_ids, window_days
    )

    # Render the three standard plots.
    print(f"\n[5] Rendering plots …")
    plot_state_timeline(
        state_seq, sorted_window_ids, start_dates, end_dates, args.k,
        os.path.join(args.output_dir, f"{stem}_state_timeline_k{args.k}.png"),
    )
    plot_state_summary_strip(
        state_seq, sorted_window_ids, mid_dates, args.k,
        os.path.join(args.output_dir, f"{stem}_state_strip_k{args.k}.png"),
    )
    plot_transition_matrix(
        trans_matrix, args.k,
        os.path.join(args.output_dir, f"{stem}_transition_matrix_k{args.k}.png"),
    )

    print(f"\n  Total wall time: {elapsed(t_global)}")
    print("=" * 60)


def parse_args():
    """Parse CLI arguments and derive per-dataset default paths."""
    p = argparse.ArgumentParser(
        description="Final HMM Viterbi decode on the seed-centroid sequence, "
                    "dataset-generalized and left-to-right-aware."
    )
    p.add_argument("--dataset", required=True, choices=["fakeddit", "yelp"],
                   help="Which dataset spec to use (drives stem + window cadence).")
    p.add_argument("--input_npz", default=None,
                   help="weights_pca.npz path. Defaults to "
                        "data/hmm_weights/<dataset>/weights_pca.npz")
    p.add_argument("--output_dir", default=None,
                   help="Output directory. Defaults to data/hmm_hmm/<dataset>")
    p.add_argument("--manifest", default=None,
                   help="Window manifest CSV for calendar dates. Defaults to "
                        "data/splits/hmm_windows/<dataset>/<dataset>_windows_manifest.csv. "
                        "Falls back to synthetic dates if absent.")
    p.add_argument("--k", type=int, required=True,
                   help="Number of HMM states chosen in the selection step "
                        "(5 for fakeddit, 18 for yelp).")
    p.add_argument("--n_pcs", type=int, required=True,
                   help="Number of leading principal components to use. Must "
                        "match the --n_pcs used in select_hmm_states.py and be "
                        "<= the number of PCs saved in the npz.")
    p.add_argument("--n_inits", type=int, default=50,
                   help="Random HMM initialisations (best training LL is kept).")
    p.add_argument("--n_iter", type=int, default=200,
                   help="Baum-Welch EM iterations per init.")
    p.add_argument("--covariance_type", default="diag",
                   choices=["diag", "full", "tied", "spherical"],
                   help="HMM covariance type — must match the selection step.")
    p.add_argument("--topology", default="ergodic",
                   choices=["ergodic", "left_to_right"],
                   help="HMM transition topology. 'left_to_right' (Bakis) forbids "
                        "backward transitions and starts in state 0 — must match "
                        "the topology used in select_hmm_states.py.")
    p.add_argument("--max_advance", type=int, default=1,
                   help="Max state-index jump per step under left_to_right. "
                        "1 = strict no-skip; 2 = classic Bakis (allows one skip). "
                        "Must match the selection step.")
    p.add_argument("--min_covar", type=float, default=1e-2,
                   help="Floor on diagonal covariances. Prevents variance "
                        "collapse / degenerate fits on short, low-dim sequences "
                        "(the cause of 'all inits failed' at large k). Must match "
                        "the selection step.")
    p.add_argument("--window_days", type=int, default=None,
                   help="Override the window cadence for synthetic-date fallback. "
                        "Defaults to the dataset spec (60 fakeddit / 30 yelp).")
    args = p.parse_args()

    # Derive per-dataset default paths when not explicitly provided.
    ds = args.dataset
    if args.input_npz is None:
        args.input_npz = f"data/hmm_weights/{ds}/weights_pca.npz"
    if args.output_dir is None:
        args.output_dir = f"data/hmm_hmm/{ds}"
    if args.manifest is None:
        args.manifest = (
            f"data/splits/hmm_windows/{ds}/{ds}_windows_manifest.csv"
        )
    return args


if __name__ == "__main__":
    main(parse_args())