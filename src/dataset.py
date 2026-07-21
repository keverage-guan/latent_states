"""
src/dataset.py

EmbeddingDataset: loads a split file (TSV or JSONL), returns precomputed
embeddings if a cache exists, otherwise encodes on the fly. What used to be
Fakeddit-specific — column names, label maps, image handling, file format — now
comes from a DatasetSpec (see datasets/registry.py), so the same class serves
Fakeddit (text+image) and Yelp (text-only).

Expected layout is unchanged:
    data/
        splits/          ← split files
        images/          ← only for datasets with an image modality
        embeddings/      ← precomputed .npy caches (optional but fast)

Embedding cache filenames: {split_name}_{modality}.npy, e.g. "OG_train_text.npy".
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
Image.MAX_IMAGE_PIXELS = None  # disable decompression-bomb check for these datasets

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from .datasets.registry import DatasetSpec, get_spec


# ── Image transform (only used by datasets with an image modality) ────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_image_transform(train: bool = True) -> T.Compose:
    crop = T.RandomCrop(224) if train else T.CenterCrop(224)
    return T.Compose([
        T.Resize(256),
        crop,
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    """
    Parameters
    ----------
    path : str
        Path to a split file (format determined by the spec).
    spec : DatasetSpec | str
        Dataset description, or a registered dataset name.
    n_way : int
        Which label scheme to use (keys of spec.label_schemes).
    train : bool
        Controls image augmentation (random vs. center crop). No effect on
        text-only datasets.
    encoders : dict | None
        Optional {modality: callable} for on-the-fly encoding when no cache
        exists. For text: encoder(str) -> vector. For image: encoder(PIL) ->
        vector. If a cache exists for a modality, its encoder is not needed.
    device : str
        Used when encoding on the fly.
    """

    def __init__(
        self,
        path: str,
        spec,
        n_way: int,
        train: bool = True,
        encoders: Optional[Dict[str, callable]] = None,
        device: str = "cpu",
    ):
        self.spec        = spec if isinstance(spec, DatasetSpec) else get_spec(spec)
        self.n_way       = n_way
        self.train       = train
        self.encoders    = encoders or {}
        self.device      = device
        self.scheme      = self.spec.label_scheme(n_way)
        self.transform   = get_image_transform(train)
        self.split_name  = os.path.splitext(os.path.basename(path))[0]

        # ── Load metadata ────────────────────────────────────────────────
        self.df = self.spec.read_split(path).reset_index(drop=True)

        # ── Derive integer targets from the label scheme ─────────────────
        self._build_label_column()

        # ── Load embedding caches (one per modality) ─────────────────────
        # Caches are aligned to the ORIGINAL split file. Validate against the
        # original length, then slice down to the rows that survived filtering.
        n_original = int(self._kept_positions.max()) + 1 if len(self._kept_positions) else 0
        self._embs: Dict[str, Optional[np.ndarray]] = {}
        for m in self.spec.modalities:
            cache = self._try_load_embeddings(m)
            if cache is not None:
                if len(cache) < n_original:
                    raise AssertionError(
                        f"{m} embedding cache has {len(cache)} rows but split file "
                        f"has at least {n_original} rows."
                    )
                cache = cache[self._kept_positions]   # realign to filtered df
                assert len(cache) == len(self.df)
            self._embs[m] = cache

    # ── Label handling ────────────────────────────────────────────────────────

    def _build_label_column(self) -> None:
        scheme = self.scheme
        raw = self.df[scheme.column]

        if pd.api.types.is_numeric_dtype(raw):
            if scheme.numeric_remap is not None:
                labels = raw.map(scheme.numeric_remap)   # unmapped -> NaN (dropped)
            else:
                labels = raw
        else:
            if scheme.str_map is None:
                raise ValueError(
                    f"Column '{scheme.column}' is non-numeric but scheme has no str_map."
                )
            labels = raw.astype(str).str.lower().str.strip().map(scheme.str_map)

        self.df["_label"] = labels

        # Remember which ORIGINAL rows survive, so precomputed embedding caches
        # (which are aligned to the full, unfiltered split file) can be indexed
        # consistently after we drop unmapped/missing-label rows. Without this,
        # a scheme that drops rows (e.g. Yelp 2-way dropping 3-star reviews)
        # would desync the cache from the dataframe.
        keep_mask = self.df["_label"].notna().to_numpy()
        self._kept_positions = np.nonzero(keep_mask)[0]

        before = len(self.df)
        self.df = self.df.loc[keep_mask].reset_index(drop=True)
        dropped = before - len(self.df)
        if dropped:
            print(f"[EmbeddingDataset:{self.spec.name}] "
                  f"Dropped {dropped} rows with unmapped/missing labels.")
        self.df["_label"] = self.df["_label"].astype(int)

    # ── Embedding cache helpers ───────────────────────────────────────────────

    def _cache_path(self, modality: str) -> str:
        return os.path.join(self.spec.embedding_dir,
                            f"{self.split_name}_{modality}.npy")

    def _try_load_embeddings(self, modality: str) -> Optional[np.ndarray]:
        p = self._cache_path(modality)
        if os.path.exists(p):
            print(f"[EmbeddingDataset:{self.spec.name}] Loading cached {modality}: {p}")
            return np.load(p)
        return None

    def _load_image(self, row_id: str):
        if self.spec.image_dir is None:
            return None
        for ext in (".jpg", ".jpeg", ".png", ".gif"):
            path = os.path.join(self.spec.image_dir, f"{row_id}{ext}")
            if os.path.exists(path):
                try:
                    return Image.open(path).convert("RGB")
                except (UnidentifiedImageError, OSError):
                    return None
        return None

    def _encode_modality(self, modality: str, row, idx) -> torch.Tensor:
        cache = self._embs[modality]
        if cache is not None:
            return torch.tensor(cache[idx], dtype=torch.float32)

        encoder = self.encoders.get(modality)
        dim     = self.spec.embedding_dim(modality)

        if modality == "text":
            if encoder is None:
                raise RuntimeError(
                    f"No text cache for {self.split_name} and no text encoder given. "
                    f"Run the precompute step or pass encoders={{'text': ...}}."
                )
            text = str(row[self.spec.text_col]) if pd.notna(row[self.spec.text_col]) else ""
            with torch.no_grad():
                return torch.tensor(encoder(text), dtype=torch.float32)

        if modality == "image":
            img = self._load_image(str(row[self.spec.id_col]))
            if img is None:
                return torch.zeros(dim, dtype=torch.float32)   # consistent fallback
            if encoder is None:
                raise RuntimeError(
                    f"No image cache for {self.split_name} and no image encoder given."
                )
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return encoder(tensor).squeeze(0).cpu()

        raise ValueError(f"Unknown modality {modality!r}")

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = torch.tensor(row["_label"], dtype=torch.long)
        embs  = tuple(self._encode_modality(m, row, idx) for m in self.spec.modalities)
        return (*embs, label)

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        counts  = self.df["_label"].value_counts().sort_index()
        weights = 1.0 / counts.values.astype(float)
        weights = weights / weights.min()
        return torch.tensor(weights, dtype=torch.float32)

    def get_class_distribution(self) -> dict:
        return self.df["_label"].value_counts().sort_index().to_dict()


# ── Backward-compatible shim ──────────────────────────────────────────────────
# Existing call sites do `FakedditDataset(tsv_path, n_way=..., train=...)`.
# This keeps them working while everything migrates to EmbeddingDataset.

def FakedditDataset(tsv_path: str, n_way: int = 2, train: bool = True,
                    text_model=None, resnet=None, device: str = "cpu"):
    encoders = {}
    if text_model is not None:
        encoders["text"] = lambda s: text_model.encode(s, show_progress_bar=False)
    if resnet is not None:
        encoders["image"] = resnet
    return EmbeddingDataset(
        path=tsv_path, spec="fakeddit", n_way=n_way, train=train,
        encoders=encoders, device=device,
    )