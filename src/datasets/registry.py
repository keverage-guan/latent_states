"""
registry.py

Dataset registry. A DatasetSpec captures everything that varies between
datasets, so the dataset class, model, trainer, and weight extractor stay
dataset-agnostic. Adding a new dataset means adding a spec here, not editing
the pipeline. Everything downstream keys off input_dim (derived, never
hardcoded) and the label scheme in the spec.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import pandas as pd


# Embedding dimensions per encoder.
TEXT_DIM = 768     # DistilRoBERTa / sentence-transformer text encoder
IMAGE_DIM = 2048   # ResNet-50 penultimate layer

EMBEDDING_DIMS = {"text": TEXT_DIM, "image": IMAGE_DIM}


@dataclass(frozen=True)
class DatasetSpec:
    """
    Immutable description of a dataset's shape and label scheme.

    Fields
    ------
    name          : short identifier, also used for embedding-cache prefixes.
    file_format   : "tsv" or "jsonl" — how a split file is read into a DataFrame.
    modalities    : ordered tuple of modality names, e.g. ("text", "image") or
                    ("text",). The order fixes the concatenation order the model
                    sees, so it must match what __getitem__ returns.
    text_col      : column holding raw text (used when encoding on the fly).
    id_col        : column holding the row id (used to locate images / cache).
    image_dir     : where images live (None for text-only datasets).
    embedding_dir : where precomputed embedding .npy caches live.
    label_schemes : maps an integer "n_way" selector to a LabelScheme describing
                    which column to read and how to turn it into a 0..K-1 target.
    window        : WindowConfig with defaults + raw-loading instructions for the
                    temporal windowing step (see prepare_window_splits.py).
    """

    name: str
    file_format: str
    modalities: Tuple[str, ...]
    text_col: str
    id_col: str
    embedding_dir: str
    label_schemes: Dict[int, "LabelScheme"]
    image_dir: Optional[str] = None
    window: Optional["WindowConfig"] = None

    @property
    def input_dim(self) -> int:
        """Total concatenated embedding width. Derived, never hardcoded."""
        return sum(EMBEDDING_DIMS[m] for m in self.modalities)

    def embedding_dim(self, modality: str) -> int:
        """Return the embedding width for a single modality."""
        return EMBEDDING_DIMS[modality]

    def label_scheme(self, n_way: int) -> "LabelScheme":
        """Return the LabelScheme for the requested n_way selector."""
        if n_way not in self.label_schemes:
            raise ValueError(
                f"Dataset '{self.name}' has no {n_way}-way label scheme. "
                f"Available: {sorted(self.label_schemes)}"
            )
        return self.label_schemes[n_way]

    def read_split(self, path: str) -> pd.DataFrame:
        """
        Read a prepared split file, dispatching on the file extension.

        The windowing step always writes prepared splits as TSV (matching the
        manifest and the .tsv globs used by the eval pipeline), even for a
        JSONL-sourced dataset, so dispatch is on the extension rather than the
        dataset's declared raw file_format.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in (".tsv", ".tab"):
            return pd.read_csv(path, sep="\t", low_memory=False)
        if ext == ".csv":
            return pd.read_csv(path, low_memory=False)
        if ext in (".jsonl", ".json"):
            return pd.read_json(path, lines=True)
        # Fall back to the declared raw format for extensionless / unknown paths.
        if self.file_format == "tsv":
            return pd.read_csv(path, sep="\t", low_memory=False)
        if self.file_format == "jsonl":
            return pd.read_json(path, lines=True)
        raise ValueError(f"Cannot determine how to read split {path!r}")

    def load_raw(self, data_dir: str) -> pd.DataFrame:
        """
        Load the raw corpus for windowing.

        Concatenate raw_sources, apply the optional row_filter, parse timestamps
        into a tz-aware UTC "created_dt" column, drop unparseable timestamps, and
        return sorted by time. Requires self.window to be set. The returned frame
        always has a "created_dt" column (datetime64[ns, UTC]) used by the
        windowing script, so downstream code never needs to know the raw
        timestamp format.
        """
        import os
        if self.window is None:
            raise ValueError(f"Dataset '{self.name}' has no WindowConfig.")
        wc = self.window

        frames = []
        for rel_path, tag in wc.raw_sources:
            path = os.path.join(data_dir, rel_path)
            if wc.raw_format == "tsv":
                df = pd.read_csv(path, sep="\t", low_memory=False)
            elif wc.raw_format == "jsonl":
                df = pd.read_json(path, lines=True)
            else:
                raise ValueError(f"Unknown raw_format {wc.raw_format!r}")
            df["original_split"] = tag
            frames.append(df)
        data = pd.concat(frames, ignore_index=True)

        if wc.row_filter is not None:
            data = data[data[wc.row_filter] == True].copy()

        col = wc.timestamp_col
        if wc.timestamp_kind == "unix_s":
            # Coerce to numeric first so non-numeric timestamps become NaN and
            # can be dropped before the unit="s" conversion.
            secs = pd.to_numeric(data[col], errors="coerce")
            data = data.assign(_secs=secs).dropna(subset=["_secs"])
            data["created_dt"] = pd.to_datetime(data["_secs"], unit="s", utc=True)
            data = data.drop(columns="_secs")
        elif wc.timestamp_kind == "iso":
            dt = pd.to_datetime(data[col], errors="coerce", utc=True)
            data = data.assign(created_dt=dt).dropna(subset=["created_dt"])
        else:
            raise ValueError(f"Unknown timestamp_kind {wc.timestamp_kind!r}")

        return data.sort_values("created_dt").reset_index(drop=True)


@dataclass(frozen=True)
class LabelScheme:
    """
    Describes how to derive an integer target column from a raw label column.

    column        : source column name.
    num_classes   : K — number of output classes.
    str_map       : optional {lowercased-string: int} mapping for string labels.
    numeric_remap : optional {int: int} applied when the source column is already
                    numeric.
    """
    column: str
    num_classes: int
    str_map: Optional[Dict[str, int]] = None
    numeric_remap: Optional[Dict[int, int]] = None


@dataclass(frozen=True)
class WindowConfig:
    """
    Everything the temporal windowing step needs that is dataset-specific.

    window_days      : calendar width of each window.
    min_samples      : a window must have at least this many raw rows to qualify.
    stratify_col     : label column used as the stratification key when
                       subsampling each window to a common size.
    timestamp_col    : column holding the row timestamp.
    timestamp_kind   : how to parse timestamp_col into a datetime:
                         "unix_s" -> integer/float seconds since epoch
                         "iso"    -> ISO-8601 string / anything pd.to_datetime
                                     handles directly
    raw_sources      : list of (relative_path, tag) files to concatenate as the
                       raw corpus, relative to --data_dir. tag is stored in an
                       "original_split" column. For a single-file dataset this is
                       one entry.
    raw_format       : "tsv" or "jsonl" for the raw_sources.
    row_filter       : optional column that must be truthy for a row to be kept.
                       None means keep all rows.
    """
    window_days: int
    min_samples: int
    stratify_col: str
    timestamp_col: str
    timestamp_kind: str
    raw_sources: Tuple[Tuple[str, str], ...]
    raw_format: str = "tsv"
    row_filter: Optional[str] = None


# Fakeddit.

_FAKEDDIT_6WAY = {
    "true":                0,
    "satire":              1,
    "false connection":    2,
    "imposter content":    3,
    "manipulated content": 4,
    "misleading content":  5,
}
_FAKEDDIT_2WAY = {"true": 1, "fake": 0}

FAKEDDIT = DatasetSpec(
    name="fakeddit",
    file_format="tsv",
    modalities=("text", "image"),
    text_col="clean_title",
    id_col="id",
    image_dir="data/images/public_image_set",
    embedding_dir="data/embeddings/fakeddit",
    label_schemes={
        # TSV encodes 0=True,1=Fake; remap to 0=Fake,1=True.
        2: LabelScheme("2_way_label", 2, str_map=_FAKEDDIT_2WAY,
                       numeric_remap={0: 1, 1: 0}),
        6: LabelScheme("6_way_label", 6, str_map=_FAKEDDIT_6WAY,
                       numeric_remap=None),
    },
    window=WindowConfig(
        window_days=60,
        min_samples=9_000,
        stratify_col="6_way_label",
        timestamp_col="created_utc",
        timestamp_kind="unix_s",
        raw_sources=(
            ("multimodal_train.tsv",       "train"),
            ("multimodal_validate.tsv",    "validate"),
            ("multimodal_test_public.tsv", "test"),
        ),
        raw_format="tsv",
        row_filter="hasImage",
    ),
)


# Yelp sentiment.
# JSONL rows: {"id", "timestamp", "sentiment": <1-5 star>, "text"}.
# Star ratings 1..5 map to classes 0..4.
YELP = DatasetSpec(
    name="yelp",
    file_format="jsonl",
    modalities=("text",),
    text_col="text",
    id_col="id",
    image_dir=None,
    embedding_dir="data/embeddings/yelp",
    label_schemes={
        5: LabelScheme("sentiment", 5,
                       numeric_remap={1: 0, 2: 1, 3: 2, 4: 3, 5: 4}),
        # Convenience 2-way: 1-2 stars -> negative(0), 4-5 -> positive(1);
        # 3 stars dropped (mapped to None by absence from the remap).
        2: LabelScheme("sentiment", 2,
                       numeric_remap={1: 0, 2: 0, 4: 1, 5: 1}),
    },
    window=WindowConfig(
        window_days=90,
        min_samples=10_000,
        stratify_col="sentiment",     # stratify on the 5-star rating
        timestamp_col="timestamp",
        timestamp_kind="iso",
        raw_sources=(("yelp_sentiment.jsonl", "all"),),
        raw_format="jsonl",
        row_filter=None,              # keep every row
    ),
)


# Registry.

_REGISTRY: Dict[str, DatasetSpec] = {
    FAKEDDIT.name: FAKEDDIT,
    YELP.name: YELP,
}


def get_spec(name: str) -> DatasetSpec:
    """Look up a registered DatasetSpec by name (case-insensitive)."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def register(spec: DatasetSpec) -> None:
    """Add or overwrite a DatasetSpec in the registry."""
    _REGISTRY[spec.name] = spec