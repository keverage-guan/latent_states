"""
train_grid.py

Train ONE hyperparameter configuration (lr, hidden_size) on the grid-search
train/val split, with validation-based early stopping. One invocation = one
grid cell; the SLURM array launches many in parallel.

This is the early-stopping counterpart to train_windows.py (which is
fixed-epoch, no-val, for the HMM regime). Both share model.py and
EmbeddingDataset, and both derive input_dim / num_classes from the dataset spec,
so this works unchanged for Yelp (text-only, 5-way) and Fakeddit (text+image).

Early stopping
--------------
Tracks best validation accuracy. If val accuracy does not improve for
--patience consecutive epochs, training stops. The reported result is the best
validation epoch (not the last).

Output (written to <output_dir>/<run_name>/):
    metrics.json   config + per-epoch history + best_val_acc + best_epoch
    best_model.pt  weights at the best validation epoch

A completed run (metrics.json present) is skipped unless --force, so re-launching
a partially-finished array is cheap.

Usage
-----
    python -m src.training.train_grid --dataset yelp \
        --train data/splits/grid_search/yelp_grid_train.tsv \
        --val   data/splits/grid_search/yelp_grid_val.tsv \
        --num_classes 5 --hidden_size 1024 --lr 1e-4 \
        --max_epochs 50 --patience 5 \
        --output_dir runs/grid_search/yelp
"""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import MultimodalMLP, seed_everything, compute_f1
from src.dataset import EmbeddingDataset
from src.datasets.registry import get_spec


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Run the model over loader and return (acc, micro_f1, macro_f1, mean_loss)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    preds, labs = [], []
    tot_loss, tot = 0.0, 0
    for *embs, labels in loader:
        embs = [e.to(device) for e in embs]
        labels = labels.to(device)
        logits = model(*embs)
        tot_loss += criterion(logits, labels).item() * labels.size(0)
        tot += labels.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        labs.extend(labels.cpu().tolist())
    acc = sum(p == l for p, l in zip(preds, labs)) / len(labs)
    micro_f1, macro_f1 = compute_f1(labs, preds, num_classes)
    return acc, micro_f1, macro_f1, tot_loss / tot


def save_atomic(path, payload):
    """Save payload to path atomically via a temp file and os.replace."""
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--hidden_size", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5,
                   help="Stop after this many epochs without val-acc improvement.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", default="runs/grid_search")
    p.add_argument("--run_name", default=None,
                   help="Default: <dataset>_n<hidden>_lr<lr>")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    """Train one grid cell with early stopping and write metrics + best weights."""
    args = parse_args()
    spec = get_spec(args.dataset)
    device = torch.device(args.device)

    if args.run_name is None:
        args.run_name = f"{spec.name}_n{args.hidden_size}_lr{args.lr}"
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.json")
    best_ckpt = os.path.join(run_dir, "best_model.pt")

    # A present metrics.json means this cell already finished; skip it.
    if os.path.exists(metrics_path) and not args.force:
        print(f"[skip] {args.run_name} already complete ({metrics_path}).")
        return

    seed_everything(args.seed)

    train_ds = EmbeddingDataset(args.train, spec=spec, n_way=args.num_classes, train=True)
    val_ds = EmbeddingDataset(args.val, spec=spec, n_way=args.num_classes, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"))

    model = MultimodalMLP(input_dim=spec.input_dim,
                          hidden_size=args.hidden_size,
                          num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[{args.run_name}] input_dim={spec.input_dim} classes={args.num_classes} "
          f"| train={len(train_ds):,} val={len(val_ds):,} | patience={args.patience}")

    history = []
    best_val_acc = -1.0
    best_epoch = 0
    epochs_no_impr = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for *embs, labels in train_loader:
            embs = [e.to(device) for e in embs]
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(*embs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * labels.size(0)
            tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_total += labels.size(0)

        val_acc, val_micro, val_macro, val_loss = evaluate(
            model, val_loader, device, args.num_classes)

        improved = val_acc > best_val_acc
        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss / tr_total, 5),
            "train_acc": round(tr_correct / tr_total, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "val_micro_f1": round(val_micro, 5),
            "val_macro_f1": round(val_macro, 5),
        })
        print(f"  epoch {epoch:02d} | train_acc={tr_correct/tr_total:.4f} "
              f"| val_acc={val_acc:.4f} macro_f1={val_macro:.4f}"
              + (" *" if improved else ""))

        if improved:
            # New best: reset the patience counter and checkpoint these weights.
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_impr = 0
            save_atomic(best_ckpt, model.state_dict())
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= args.patience:
                print(f"  early stop at epoch {epoch} "
                      f"({args.patience} without improvement).")
                break

    # Pull the best epoch's row so the collector reads best-epoch (not last) metrics.
    best_row = next(h for h in history if h["epoch"] == best_epoch)
    metrics = {
        "dataset": spec.name,
        "run_name": args.run_name,
        "num_classes": args.num_classes,
        "hidden_size": args.hidden_size,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "input_dim": spec.input_dim,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 5),
        "best_val_macro_f1": best_row["val_macro_f1"],
        "best_val_micro_f1": best_row["val_micro_f1"],
        "best_val_loss": best_row["val_loss"],
        "train_file": os.path.basename(args.train),
        "val_file": os.path.basename(args.val),
        "history": history,
    }
    save_path = metrics_path + ".tmp"
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    os.replace(save_path, metrics_path)

    print(f"[{args.run_name}] best_val_acc={best_val_acc:.4f} "
          f"@ epoch {best_epoch} → {metrics_path}")


if __name__ == "__main__":
    main()