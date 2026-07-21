"""
src/precompute_yelp_embeddings.py

Precompute text embeddings for Yelp splits so training reads a fast .npy cache
instead of encoding on the fly. Text-only (Yelp has no image modality).

Output: for each split file <name>.tsv, writes
    {embedding_dir}/{name}_text.npy    shape (n_rows, 768), row-aligned to the
                                       ORIGINAL split file (before label filtering).

The row alignment matters: EmbeddingDataset slices this cache by the positions of
rows that survive label filtering, so the cache MUST cover every row of the split
in original order. Do not filter here.

Usage:
    python src/preprocess/precompute_yelp_embeddings.py \
        --splits data/splits/yelp/*.tsv \
        --model sentence-transformers/all-distilroberta-v1 \
        --batch_size 256
"""

from __future__ import annotations

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets.registry import get_spec


def parse_args():
    """Parse command-line arguments for the precompute run."""
    p = argparse.ArgumentParser(description="Precompute Yelp text embeddings.")
    p.add_argument("--splits", nargs="+", required=True,
                   help="Split .tsv files (globs allowed).")
    p.add_argument("--model", default="sentence-transformers/all-distilroberta-v1",
                   help="sentence-transformers model producing 768-dim vectors.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", default=None,
                   help="Override embedding_dir from the yelp spec.")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if a cache already exists.")
    return p.parse_args()


def main():
    args = parse_args()
    spec = get_spec("yelp")
    out_dir = args.output_dir or spec.embedding_dir
    os.makedirs(out_dir, exist_ok=True)

    # Expand any glob patterns in --splits into concrete file paths.
    paths = []
    for pat in args.splits:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        print("No split files matched.")
        return

    from sentence_transformers import SentenceTransformer
    print(f"Loading encoder: {args.model}")
    encoder = SentenceTransformer(args.model, device=args.device)
    dim = encoder.get_sentence_embedding_dimension()
    assert dim == spec.embedding_dim("text"), (
        f"Encoder dim {dim} != spec text dim {spec.embedding_dim('text')}. "
        f"Update TEXT_DIM in datasets/registry.py or pick a matching model."
    )

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(out_dir, f"{name}_text.npy")
        if os.path.exists(out) and not args.force:
            print(f"  [{name}] cache exists, skipping (use --force to redo).")
            continue

        df = spec.read_split(path).reset_index(drop=True)   # NO filtering
        texts = df[spec.text_col].fillna("").astype(str).tolist()
        print(f"  [{name}] encoding {len(texts):,} rows...")
        embs = encoder.encode(
            texts, batch_size=args.batch_size,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)

        assert len(embs) == len(df), (len(embs), len(df))

        # Write to a temp file then atomically replace, so an interrupted write
        # cannot leave a truncated cache in place.
        tmp = out + ".tmp.npy"
        np.save(tmp, embs)
        os.replace(tmp, out)
        print(f"  [{name}] wrote {out}  {embs.shape}")

    print("Done.")


if __name__ == "__main__":
    main()