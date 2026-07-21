"""
cross_window_eval.py

Cross-window F1 evaluation, generalized across datasets.

HMM-agnostic: it never touches a decode file. For a single training window
(--train_window_idx) it loads all seed models and evaluates each on every valid
test window, writing one row file:

    data/hmm_perf/<dataset>/rows/row_NNN.npz

After all array tasks finish, run merge_cross_window.py to assemble the full
N×N F1 matrix. HMM state labels are joined later in within_across_states.py,
keeping the two pipelines independent.

Designed to be run as a SLURM array job (one training window per task).

Usage
-----
    # Fakeddit, 6-way, one training window (SLURM array element):
    python -m src.eval.cross_window_eval --dataset fakeddit \
        --num_classes 6 --train_window_idx $SLURM_ARRAY_TASK_ID

    # Yelp, 5-way, explicit paths:
    python -m src.eval.cross_window_eval --dataset yelp \
        --num_classes 5 \
        --runs_dir   runs/hmm_windows/yelp \
        --splits_dir data/splits/hmm_windows/yelp \
        --output_dir data/hmm_perf/yelp \
        --train_window_idx 0

Requirements: numpy, torch, scikit-learn
"""

import os
import sys
import time
import glob
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.model import MultimodalMLP
from src.dataset import EmbeddingDataset
from src.datasets.registry import get_spec


def elapsed(t0: float) -> str:
    """Format seconds since t0 as a human-readable duration string."""
    s = time.time() - t0
    return f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}min"


def find_valid_windows(runs_dir: str) -> list[int]:
    """
    Return the sorted window indices that have at least one trained seed model
    (best_model.pt) under runs_dir/window_XXX/seed_Y/.
    """
    dirs = sorted(glob.glob(os.path.join(runs_dir, "window_???")))
    valid = []
    for d in dirs:
        name = os.path.basename(d)
        try:
            idx = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if glob.glob(os.path.join(d, "seed_*", "best_model.pt")):
            valid.append(idx)
    return sorted(valid)


def split_path_for_window(splits_dir: str, stem: str, window_idx: int) -> str:
    """
    Path to the prepared split for a window. Matches the naming written by
    prepare_window_splits.py and consumed by train_windows.find_window_files:
    <stem>_window_<WWW>.tsv.
    """
    return os.path.join(splits_dir, f"{stem}_window_{window_idx:03d}.tsv")


def load_seed_models(
    runs_dir: str,
    window_idx: int,
    input_dim: int,
    hidden_size: int,
    num_classes: int,
    device: torch.device,
) -> dict[int, torch.nn.Module]:
    """
    Load every seed model for a given window.

    Returns {seed_idx: model (eval mode, on device)}. input_dim comes from the
    dataset spec, not a hardcoded constant, so the same loader serves text-only
    and text+image datasets.
    """
    window_dir = os.path.join(runs_dir, f"window_{window_idx:03d}")
    models: dict[int, torch.nn.Module] = {}

    for sd in sorted(glob.glob(os.path.join(window_dir, "seed_*"))):
        pt = os.path.join(sd, "best_model.pt")
        if not os.path.isfile(pt):
            continue
        try:
            seed_idx = int(os.path.basename(sd).split("_")[1])
        except (IndexError, ValueError):
            continue

        ckpt = torch.load(pt, map_location=device)
        # Checkpoints may be either a raw state_dict or a dict wrapping one
        # under "model_state"; handle both.
        state_dict = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt

        model = MultimodalMLP(
            input_dim=input_dim, hidden_size=hidden_size, num_classes=num_classes,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models[seed_idx] = model

    return models


@torch.no_grad()
def eval_model_on_window(
    model: torch.nn.Module,
    split_path: str,
    spec,
    num_classes: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """
    Evaluate one model on one window.

    Returns (macro_f1, per_class_f1 of shape (num_classes,)). Modality-agnostic:
    the loader yields (*embs, labels) and the model concatenates however many
    embedding tensors it is given.
    """
    ds = EmbeddingDataset(split_path, spec=spec, n_way=num_classes, train=False)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_preds, all_labels = [], []
    for *embs, labels in dl:
        embs = [e.to(device) for e in embs]
        logits = model(*embs)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    labels_range = list(range(num_classes))

    macro_f1 = float(f1_score(all_labels, all_preds,
                              average="macro", zero_division=0))
    # Pass an explicit labels range so absent classes still get a column.
    per_class = f1_score(all_labels, all_preds, average=None,
                         zero_division=0,
                         labels=labels_range).astype(np.float32)
    return macro_f1, per_class


def plot_heatmap(
    f1_matrix: np.ndarray,
    valid_ids: list[int],
    output_path: str,
    dataset: str | None = None,
) -> None:
    """
    Plain cross-window F1 heatmap — no HMM state annotations.

    Lives here (rather than in merge_cross_window.py) so the eval job and the
    merge job share one implementation; merge imports it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(valid_ids)

    # Mask the diagonal (within-distribution) so it doesn't bias the colormap.
    # The saved matrix is untouched; this copy is plot-only.
    plot_mat = f1_matrix.copy().astype(float)
    np.fill_diagonal(plot_mat, np.nan)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.4), max(5, n * 0.4)))
    im = ax.imshow(plot_mat, aspect="auto", cmap="RdYlGn",
                   vmin=np.nanmin(plot_mat), vmax=np.nanmax(plot_mat))
    plt.colorbar(im, ax=ax, label="Macro F1")

    tick_labels = [f"W{i:03d}" for i in valid_ids]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Test window", fontsize=11)
    ax.set_ylabel("Train window", fontsize=11)
    title = "Cross-window macro F1"
    if dataset:
        title += f"  ({dataset})"
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main(args: argparse.Namespace) -> None:
    """Run cross-window evaluation for one training window and save a row file."""
    t0 = time.time()
    spec = get_spec(args.dataset)
    stem = spec.name
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    rows_dir = os.path.join(args.output_dir, "rows")
    os.makedirs(rows_dir, exist_ok=True)

    print("=" * 60)
    print(f"  cross_window_eval.py  —  dataset={stem}  "
          f"train window {args.train_window_idx}")
    print("=" * 60)
    print(f"  Device      : {device}")
    print(f"  input_dim   : {spec.input_dim}  (modalities={spec.modalities})")
    print(f"  num_classes : {args.num_classes}")

    # Discover valid windows.
    valid_ids = find_valid_windows(args.runs_dir)
    print(f"\n  Valid windows ({len(valid_ids)}): {valid_ids}")

    if args.train_window_idx not in valid_ids:
        print(f"  [skip] Window {args.train_window_idx} has no trained models.")
        return

    row_idx = valid_ids.index(args.train_window_idx)
    n_valid = len(valid_ids)

    # Load seed models.
    print(f"\n[2] Loading models for window {args.train_window_idx:03d} ...")
    models = load_seed_models(
        args.runs_dir, args.train_window_idx,
        spec.input_dim, args.hidden_size, args.num_classes, device,
    )
    if not models:
        print(f"  ERROR: no best_model.pt found for window "
              f"{args.train_window_idx:03d}")
        return

    seed_ids = sorted(models.keys())
    n_seeds = len(seed_ids)
    print(f"  Loaded {n_seeds} seed models: seeds {seed_ids}")

    # Evaluate on every window.
    row_f1 = np.full(n_valid, np.nan, dtype=np.float32)
    row_per_class = np.full((n_valid, args.num_classes), np.nan, dtype=np.float32)
    row_f1_per_seed = np.full((n_seeds, n_valid), np.nan, dtype=np.float32)

    print(f"\n[3] Evaluating on {n_valid} windows ...")
    for j, test_win in enumerate(valid_ids):
        split = split_path_for_window(args.splits_dir, stem, test_win)
        if not os.path.isfile(split):
            print(f"  [warn] split missing: {split} — skipping")
            continue

        seed_f1s, seed_pcs = [], []
        for si, seed_idx in enumerate(seed_ids):
            f1, pc = eval_model_on_window(
                models[seed_idx], split, spec, args.num_classes,
                args.batch_size, args.num_workers, device,
            )
            row_f1_per_seed[si, j] = f1
            seed_f1s.append(f1)
            seed_pcs.append(pc)

        if seed_f1s:
            # Aggregate across seeds with the requested reducer.
            agg = np.mean if args.seed_agg == "mean" else np.median
            row_f1[j] = float(agg(seed_f1s))
            row_per_class[j] = agg(seed_pcs, axis=0)

        marker = " ← (self)" if test_win == args.train_window_idx else ""
        print(f"    test W{test_win:03d}: F1 = {row_f1[j]:.4f}{marker}")

    # Save row file.
    row_path = os.path.join(rows_dir, f"row_{row_idx:03d}.npz")
    np.savez(
        row_path,
        row_idx         = np.int32(row_idx),
        valid_ids       = np.array(valid_ids, dtype=np.int32),
        row_f1          = row_f1,
        row_per_class   = row_per_class,
        row_f1_per_seed = row_f1_per_seed,
        dataset         = np.str_(stem),
        num_classes     = np.int32(args.num_classes),
    )
    print(f"\n  Saved: {row_path}")
    print(f"  Elapsed: {elapsed(t0)}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and fill in per-dataset default paths."""
    p = argparse.ArgumentParser(
        description=(
            "Cross-window F1 evaluation for one training window, "
            "dataset-generalized. HMM-agnostic: does not require a decode file. "
            "Run merge_cross_window.py after all rows are done, then "
            "within_across_states.py to join with the HMM decode."
        )
    )
    p.add_argument("--dataset", required=True, choices=["fakeddit", "yelp"],
                   help="Which dataset spec to use (drives input_dim + stem).")
    p.add_argument("--num_classes", type=int, required=True,
                   help="Label scheme / class count (6 fakeddit, 5 yelp). "
                        "Must match what train_windows.py was run with.")
    p.add_argument("--runs_dir", default=None,
                   help="Root containing window_XXX/seed_Y/best_model.pt. "
                        "Defaults to runs/hmm_windows/<dataset>.")
    p.add_argument("--splits_dir", default=None,
                   help="Dir containing <dataset>_window_XXX.tsv files. "
                        "Defaults to data/splits/hmm_windows/<dataset>.")
    p.add_argument("--output_dir", default=None,
                   help="Output root; row files go to output_dir/rows/. "
                        "Defaults to data/hmm_perf/<dataset>.")
    p.add_argument("--hidden_size", type=int, default=1024,
                   help="Must match the width used in train_windows.py.")
    p.add_argument("--seed_agg", default="mean", choices=["mean", "median"],
                   help="How to aggregate F1 across seeds.")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--train_window_idx", type=int, required=True,
                   help="Which window's models to evaluate (SLURM_ARRAY_TASK_ID).")
    args = p.parse_args()

    ds = args.dataset
    if args.runs_dir is None:
        args.runs_dir = f"runs/hmm_windows/{ds}"
    if args.splits_dir is None:
        args.splits_dir = f"data/splits/hmm_windows/{ds}"
    if args.output_dir is None:
        args.output_dir = f"data/hmm_perf/{ds}"
    return args


if __name__ == "__main__":
    main(parse_args())