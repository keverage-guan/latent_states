"""
src/model.py

Single source of truth for the model architecture and everything that must stay
in lockstep with it: the module definition, RNG seeding, the metric helpers, and
the state-dict <-> structured-weight conversions used by the weight extractor.

Previously `MultimodalMLP`, `seed_everything`, `compute_f1`, and the state-dict
key list were duplicated across train.py, plot_exp2_figure.py, and
extract_weights_pca.py, with the architecture constants (INPUT_DIM, HIDDEN_SIZE,
NUM_CLASSES, FLAT_DIM) independently hardcoded. Any change to one copy silently
desynced the others. They now live here and are imported everywhere.

The model is modality-agnostic: `forward(*embs)` concatenates however many
embedding tensors it is given, so a text-only dataset (Yelp) and a text+image
dataset (Fakeddit) share one class. The number and dimensionality of the
embeddings is a property of the *dataset*, not the model — see datasets/registry.py.
"""

from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


# ── State-dict layout ─────────────────────────────────────────────────────────
# The parameter order below defines the canonical flat-vector layout used for
# alignment and PCA. Do not reorder without re-running the whole pipeline.
HIDDEN_W_KEY = "net.0.weight"   # (hidden_size, input_dim)
HIDDEN_B_KEY = "net.0.bias"     # (hidden_size,)
OUT_W_KEY    = "net.2.weight"   # (num_classes, hidden_size)
OUT_B_KEY    = "net.2.bias"     # (num_classes,)

STATE_DICT_KEYS = [HIDDEN_W_KEY, HIDDEN_B_KEY, OUT_W_KEY, OUT_B_KEY]


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    """Set all relevant RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN ops (slight perf cost; safe to remove if speed matters)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Model ─────────────────────────────────────────────────────────────────────

class MultimodalMLP(nn.Module):
    """
    One hidden layer feed-forward network.

    Input:  concat of one or more embedding tensors, total width = input_dim
    Output: num_classes logits

    The forward signature is variadic: pass one tensor for a unimodal dataset,
    two for text+image, etc. The dataset is responsible for returning embeddings
    in the fixed order the model was configured with (see DatasetSpec.modalities).
    """

    def __init__(self, input_dim: int, hidden_size: int = 512, num_classes: int = 2):
        super().__init__()
        self.input_dim   = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, *embs: torch.Tensor) -> torch.Tensor:
        x = embs[0] if len(embs) == 1 else torch.cat(embs, dim=1)
        return self.net(x)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_f1(labels, preds, num_classes):
    """
    Dependency-free micro/macro F1.

    In multiclass single-label classification:
      micro F1  = accuracy  (TP_total / N, since sum FP == sum FN == total wrong)
      macro F1  = unweighted mean of per-class F1 scores
    """
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for true, pred in zip(labels, preds):
        if pred == true:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    f1s = []
    for c in range(num_classes):
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1s.append(2 * tp[c] / denom if denom > 0 else 0.0)
    macro_f1 = sum(f1s) / num_classes

    total_tp, total_fp, total_fn = sum(tp), sum(fp), sum(fn)
    denom    = 2 * total_tp + total_fp + total_fn
    micro_f1 = 2 * total_tp / denom if denom > 0 else 0.0

    return micro_f1, macro_f1


# ── Structured weight I/O (used by the weight extractor) ──────────────────────

def load_structured_weights(ckpt_path: str) -> Dict[str, np.ndarray]:
    """
    Load a checkpoint and return the four weight tensors as numpy arrays,
    kept structured (not flat) so hidden-unit alignment can operate on them.

    Returns dict with keys:
        W_h : (hidden_size, input_dim)
        b_h : (hidden_size,)
        W_o : (num_classes, hidden_size)
        b_o : (num_classes,)
    """
    state_dict = torch.load(ckpt_path, map_location="cpu")
    for k in STATE_DICT_KEYS:
        if k not in state_dict:
            raise KeyError(
                f"Key '{k}' missing from state_dict at {ckpt_path}. "
                f"Keys present: {list(state_dict.keys())}"
            )
    return {
        "W_h": state_dict[HIDDEN_W_KEY].cpu().float().numpy(),
        "b_h": state_dict[HIDDEN_B_KEY].cpu().float().numpy(),
        "W_o": state_dict[OUT_W_KEY].cpu().float().numpy(),
        "b_o": state_dict[OUT_B_KEY].cpu().float().numpy(),
    }


def flatten_weights(sw: Dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate a structured weight dict into a 1-D float32 vector (fixed order)."""
    return np.concatenate([
        sw["W_h"].ravel(),
        sw["b_h"].ravel(),
        sw["W_o"].ravel(),
        sw["b_o"].ravel(),
    ]).astype(np.float32)


def flat_dim(input_dim: int, hidden_size: int, num_classes: int) -> int:
    """
    Dimension of the flattened weight vector for a given architecture.

    Derived from the architecture rather than hardcoded so it can never drift
    from the trained models. For Fakeddit (2816, 1024, 6) this is 2,889,734.
    """
    return (hidden_size * input_dim + hidden_size
            + num_classes * hidden_size + num_classes)


def infer_arch_from_checkpoint(ckpt_path: str) -> Dict[str, int]:
    """
    Read input_dim / hidden_size / num_classes directly off a checkpoint's shapes.
    Lets the weight extractor avoid assuming a fixed architecture.
    """
    sw = load_structured_weights(ckpt_path)
    hidden_size, input_dim = sw["W_h"].shape
    num_classes            = sw["W_o"].shape[0]
    return {"input_dim": int(input_dim),
            "hidden_size": int(hidden_size),
            "num_classes": int(num_classes)}