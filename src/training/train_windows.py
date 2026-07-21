"""
train_windows.py

Driver that trains the per-window ensemble for the HMM experiment, generalized
across datasets. It discovers the window splits produced by
prepare_window_splits.py and trains `--n_seeds` models per window in
fixed-epoch / no-validation mode (so each weight vector is a clean function of
the training data and seed, matching the paper's protocol).

Output layout (consumed by extract_weights_pca.py and cross_window_eval.py):

    <runs_dir>/window_<WWW>/seed_<S>/best_model.pt
                                     metrics.json

Takes a --dataset and derives input_dim / num_classes from the spec, so it works
unchanged for Yelp (text-only, 5-way) and Fakeddit (text+image, 6-way).

Two execution modes
--------------------
1. Local loop (all windows, all seeds):
       python src/train_windows.py --dataset yelp \
           --splits_dir data/splits/hmm_windows/yelp \
           --runs_dir   runs/hmm_windows/yelp \
           --num_classes 5 --hidden_size 1024 --lr 1e-4 \
           --max_epochs 20 --n_seeds 10

2. SLURM array (one window per task, all its seeds):
       python src/train_windows.py ... --window_idx $SLURM_ARRAY_TASK_ID

   Size the array to the number of window files (printed by --list).
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import MultimodalMLP, seed_everything, compute_f1
from src.dataset import EmbeddingDataset
from src.datasets.registry import get_spec


def find_window_files(splits_dir, stem):
    """
    Return [(window_idx, path), ...] sorted by index for files named
    <stem>_window_<WWW>.tsv in splits_dir.
    """
    pattern = os.path.join(splits_dir, f"{stem}_window_*.tsv")
    out = []
    for p in sorted(glob.glob(pattern)):
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            idx = int(base.split("_")[-1])
        except ValueError:
            # Skip files whose trailing token is not a window index.
            continue
        out.append((idx, p))
    return sorted(out)


def train_one(spec, split_path, num_classes, hidden_size, lr, max_epochs,
              batch_size, num_workers, seed, device, out_dir, force):
    """
    Fixed-epoch, no-validation training of one model. Saves best_model.pt
    (final-epoch weights) and metrics.json. Skips if already complete unless
    force. Returns the metrics dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best_model.pt")
    metrics_path = os.path.join(out_dir, "metrics.json")

    # Reuse a finished run's metrics rather than retraining.
    if os.path.exists(metrics_path) and os.path.exists(ckpt_path) and not force:
        with open(metrics_path) as f:
            return json.load(f)

    # Seed before model init and loader creation so weights and batch order are
    # fully determined by the seed.
    seed_everything(seed)

    ds = EmbeddingDataset(split_path, spec=spec, n_way=num_classes, train=True)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    model = MultimodalMLP(
        input_dim=spec.input_dim, hidden_size=hidden_size, num_classes=num_classes,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for *embs, labels in loader:
            embs = [e.to(device) for e in embs]
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(*embs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        history.append({
            "epoch": epoch,
            "train_loss": run_loss / total,
            "train_acc": correct / total,
        })

    # Save final-epoch weights atomically.
    tmp = ckpt_path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, ckpt_path)

    metrics = {
        "dataset": spec.name,
        "split": os.path.basename(split_path),
        "num_classes": num_classes,
        "hidden_size": hidden_size,
        "input_dim": spec.input_dim,
        "lr": lr,
        "max_epochs": max_epochs,
        "seed": seed,
        "history": history,
    }
    tmp = metrics_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(metrics, f, indent=2)
    os.replace(tmp, metrics_path)
    return metrics


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--splits_dir", required=True)
    p.add_argument("--runs_dir", required=True)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--hidden_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_seeds", type=int, default=10,
                   help="Train seeds 0..n_seeds-1 per window.")
    p.add_argument("--window_idx", type=int, default=None,
                   help="Train only this window (for SLURM arrays). "
                        "Omit to loop over all windows.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    p.add_argument("--list", action="store_true",
                   help="List discovered windows and exit.")
    return p.parse_args()


def main():
    """Discover window splits and train the requested windows and seeds."""
    args = parse_args()
    spec = get_spec(args.dataset)
    device = torch.device(args.device)

    windows = find_window_files(args.splits_dir, spec.name)
    if not windows:
        sys.exit(f"No {spec.name}_window_*.tsv found in {args.splits_dir}. "
                 f"Run prepare_window_splits.py first.")

    if args.list:
        print(f"{len(windows)} windows for dataset '{spec.name}':")
        for idx, path in windows:
            print(f"  window {idx:03d}: {os.path.basename(path)}")
        print(f"\nSLURM array size: --array=0-{len(windows)-1}")
        return

    # In array mode, restrict to the single window this task owns.
    if args.window_idx is not None:
        windows = [(i, p) for i, p in windows if i == args.window_idx]
        if not windows:
            sys.exit(f"No window with index {args.window_idx}.")

    print(f"Training '{spec.name}': {len(windows)} window(s) × {args.n_seeds} seeds "
          f"| input_dim={spec.input_dim} classes={args.num_classes} "
          f"hidden={args.hidden_size} device={device}")

    for w_idx, split_path in windows:
        for seed in range(args.n_seeds):
            out_dir = os.path.join(args.runs_dir, f"window_{w_idx:03d}", f"seed_{seed}")
            m = train_one(
                spec=spec, split_path=split_path,
                num_classes=args.num_classes, hidden_size=args.hidden_size,
                lr=args.lr, max_epochs=args.max_epochs,
                batch_size=args.batch_size, num_workers=args.num_workers,
                seed=seed, device=device, out_dir=out_dir, force=args.force,
            )
            last = m["history"][-1] if m.get("history") else {}
            print(f"  window {w_idx:03d} seed {seed}: "
                  f"train_acc={last.get('train_acc', float('nan')):.4f} → {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()