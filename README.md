# Latent States in Neural Networks

Code accompanying the paper *Latent States in Neural Networks: Recovering the Temporal Structure of Drifting Data from Model Weights*.

The pipeline trains an independent classifier on each of a series of consecutive time windows, aligns and PCA-reduces the resulting weight vectors, fits an HMM to the chronological trajectory, and tests whether classifiers transfer better within a recovered state than across a state boundary. It runs on two datasets: Fakeddit (text + image, 6-way) and Yelp (text-only, 5-way).

## Design

Everything dataset-specific lives in a single `DatasetSpec` in `datasets/registry.py`: modalities, file format, column names, label schemes, embedding dimensions, and windowing config. The model, loader, trainer, weight extractor, and every analysis read from the spec, so adding a dataset means adding a spec, not editing the pipeline. The model's input dimension is derived from the spec's modalities rather than hardcoded, so one `MultimodalMLP` serves both datasets.

Heavier steps (per-window training, cross-window eval) are written to run as SLURM array jobs — one element per window — but each also runs as a local loop.

## Directory layout

```
src/
├── model.py                       Model, seeding, metrics, structured weight I/O
├── dataset.py                     EmbeddingDataset: split loading + embedding cache
├── generate_all_figures.py        Regenerate paper figures from saved outputs
│
├── datasets/
│   └── registry.py                DatasetSpec / LabelScheme / WindowConfig + specs
│
├── preprocess/
│   ├── prepare_window_splits.py            Temporal windowing of the raw corpus
│   ├── precompute_fakeddit_embeddings.py   RoBERTa + ResNet-50 caches
│   └── precompute_yelp_embeddings.py       RoBERTa (text-only) caches
│
├── training/
│   ├── make_grid_splits.py         Window-sized grid-search train/val split
│   ├── train_grid.py               Train one hyperparameter cell (early stopping)
│   ├── collect_grid_search.py      Collect grid cells, pick best config
│   └── train_windows.py            Train the per-window ensemble (fixed-epoch)
│
├── weight_extraction/
│   └── extract_weights_pca.py      Align hidden units across seeds, PCA, z-score
│
├── hmm/
│   ├── select_hmm_states.py        Choose K (BIC / AIC / LOO-CV)
│   └── decode_hmm.py               Fit final HMM, Viterbi-decode the state sequence
│
├── eval/
│   ├── cross_window_eval.py        Evaluate one window's models on every window
│   ├── merge_cross_window.py       Assemble per-row files into the F1 matrix
│   └── within_across_states.py     Within- vs. across-state transfer test
│
└── analyses/
    ├── check_equal_windows.py      HMM vs. equal-size-partition baseline
    ├── state_pair_correlation.py   Transfer vs. class-distribution divergence (JSD)
    ├── jsd_stability_analysis.py   Stability of the JSD structure
    ├── partial_jsd_transfer.py     Within-state effect after partialling out JSD + lag
    └── characterize_states.py      Per-state characterization report
```

## File reference

**`model.py`** — Single source of truth for the architecture. Defines `MultimodalMLP` (one hidden layer, variadic `forward(*embs)`), `seed_everything`, the F1 helpers, and the state-dict ↔ structured-weight conversions used by the extractor. The canonical flat-vector weight layout is defined here so the trainer and extractor can't drift apart.

**`dataset.py`** — `EmbeddingDataset` reads a split, derives integer targets from the spec's label scheme, and serves embeddings from an `.npy` cache when present (encoding on the fly otherwise). Caches are aligned to the unfiltered split and sliced to the rows that survive label filtering.

**`datasets/registry.py`** — `DatasetSpec`, `LabelScheme`, and `WindowConfig` dataclasses plus the `FAKEDDIT` and `YELP` specs and `get_spec` / `register`. `input_dim` is a derived property (sum of per-modality embedding widths).

**`preprocess/prepare_window_splits.py`** — Cuts the raw corpus into fixed-width windows, keeps the longest contiguous run meeting a minimum count, stratified-subsamples each to a common size, and writes one TSV per window plus a manifest CSV (dates + per-class counts).

**`preprocess/precompute_*_embeddings.py`** — Encode split text (and Fakeddit images) into `.npy` caches, row-aligned to the unfiltered splits. Fakeddit's version checkpoints image encoding so it can resume.

**`training/make_grid_splits.py`** — Draws disjoint, class-stratified train/val sets, each the size of one window, so tuned hyperparameters match the per-window regime.

**`training/train_grid.py`** — Trains one `(lr, hidden_size)` cell with validation-based early stopping. One invocation = one grid cell.

**`training/collect_grid_search.py`** — Reads every cell's `metrics.json`, writes a combined `grid_results.csv`, and picks the best config (default metric `val_macro_f1`) to `grid_best.json`.

**`training/train_windows.py`** — Trains `--n_seeds` models per window in fixed-epoch, no-validation mode, so each weight vector depends only on the window's data and seed. Writes `runs/.../window_<WWW>/seed_<S>/best_model.pt`.

**`weight_extraction/extract_weights_pca.py`** — Loads every checkpoint, aligns hidden units across seeds by iterated Hungarian matching, PCA-reduces and z-scores the aligned weights, and saves per-window centroids to `weights_pca.npz`. Fits all non-degenerate PCs; how many to use is chosen downstream.

**`hmm/select_hmm_states.py`** — Sweeps K, scoring by BIC, AIC, and leave-one-seed-out CV log-likelihood, with an ARI stability matrix across inits. `--n_pcs` sets how many principal components feed the HMM.

**`hmm/decode_hmm.py`** — Fits the final Gaussian HMM (diagonal covariance; ergodic or left-to-right topology) on the centroid sequence, Viterbi-decodes the state sequence, and saves `<dataset>_decode_k<k>.npz`.

**`eval/cross_window_eval.py`** — HMM-agnostic. For one training window, loads all seed models and records macro-F1 transferring to every valid window, writing one `row_NNN.npz`. Run once per training window (the array job).

**`eval/merge_cross_window.py`** — Assembles the row files into the full N×N F1 matrix, plus a column-centered version (each entry minus its test window's mean incoming F1). Downstream analyses use the column-centered matrix.

**`eval/within_across_states.py`** — The main test. Partitions off-diagonal window pairs into within-/across-state and tests the F1 gap with a lag-stratified label-shuffle permutation test.

**`analyses/check_equal_windows.py`** — Compares the HMM segmentation against a contiguous equal-size partition into the same number of groups, via a permutation test on the difference of pooled gaps.

**`analyses/state_pair_correlation.py`** — Relates transfer to the Jensen-Shannon divergence between window class distributions.

**`analyses/jsd_stability_analysis.py`** — Examines the stability of the JSD structure across the decode.

**`analyses/partial_jsd_transfer.py`** — Tests whether the within-state advantage survives after class divergence and lag are residualized out (Freedman-Lane permutation + cluster-robust SEs).

**`analyses/characterize_states.py`** — Produces a human-readable per-state report (class distribution, category breakdown, centroids).

**`generate_all_figures.py`** — Reads the saved outputs above and regenerates the paper's figures.

## Requirements

Python 3.10+, with `numpy`, `pandas`, `scikit-learn`, `scipy`, `torch`, `torchvision`, `sentence-transformers`, `hmmlearn`, `matplotlib`, `Pillow`, `tqdm`, `joblib`. Each script lists the subset it needs in its docstring.

## Data

The datasets are not included. Obtain [Fakeddit](https://fakeddit.netlify.app/) (Nakamura et al., 2020) and the [Yelp reviews](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) from their original sources and point the preprocessing scripts at them via `--data_dir`. The expected raw layout for each — source files, timestamp field, label column, row filter — is documented in that dataset's `WindowConfig` in `datasets/registry.py`. The pipeline reads and writes under `data/` (splits, caches, HMM outputs, analysis results) and `runs/` (checkpoints); default paths are derived per dataset and overridable on any script's command line.

## Running

Every script takes `--dataset {fakeddit,yelp}` and derives default paths from it. A minimal Fakeddit run, assuming the raw corpus is in place:

```bash
# Windowing + embeddings
python -m src.preprocess.prepare_window_splits --dataset fakeddit \
    --data_dir <raw_dir> --output_dir data/splits/hmm_windows/fakeddit
python -m src.preprocess.precompute_fakeddit_embeddings

# Per-window ensemble (SLURM array over windows, or a local loop)
python -m src.training.train_windows --dataset fakeddit \
    --num_classes 6 --hidden_size 1024 --lr 1e-4 --max_epochs 20 --n_seeds 10

# Align + reduce weights
python -m src.weight_extraction.extract_weights_pca \
    --runs_dir runs/hmm_windows/fakeddit --n_windows 35

# Select K, then fit + decode
python -m src.hmm.select_hmm_states --dataset fakeddit --n_pcs 5 --k_min 2 --k_max 12
python -m src.hmm.decode_hmm --dataset fakeddit --k 11 --n_pcs 5 --topology left_to_right

# Cross-window transfer (array over training windows), then merge
python -m src.eval.cross_window_eval --dataset fakeddit --num_classes 6 \
    --train_window_idx $SLURM_ARRAY_TASK_ID
python -m src.eval.merge_cross_window --dataset fakeddit --num_classes 6

# Main result + controls
python -m src.eval.within_across_states --dataset fakeddit --k 11
python -m src.analyses.check_equal_windows --dataset fakeddit --k 11
python -m src.analyses.partial_jsd_transfer --dataset fakeddit --k 11

# Figures
python -m src.generate_all_figures --dataset fakeddit
```

The Yelp run is identical up to `--dataset yelp`, `--num_classes 5`, `--n_windows 56`, and `--k 16`.