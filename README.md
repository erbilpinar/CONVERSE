# CONVERSE

![Black](https://img.shields.io/badge/Code%20Style-Black-black.svg)
![MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Thesis](https://img.shields.io/badge/MSc%20Thesis-Politecnico%20di%20Milano-blue.svg)
![arXiv](https://img.shields.io/badge/arXiv-2602.01367-b31b1b.svg?logo=arxiv)

A **survival analysis framework** built around **CONVERSE** — a novel deep clustering model for time-to-event prediction — benchmarked against a broad set of classical and deep survival baselines.

> **MSc Thesis** — Politecnico di Milano, Master of Science in Computer Science & Engineering.
> **Pınar Erbil** · Academic Year 2025–2026
>
> A preprint version of this work is available on arXiv: **[arXiv:2602.01367](https://arxiv.org/abs/2602.01367)**

---

## Overview

This repository contains the full implementation of **CONVERSE** (*CONtrastive Variational Ensemble for Risk Stratification and Estimation*), along with a reproducible experimental pipeline for benchmarking survival models.

CONVERSE jointly learns a latent representation, a soft cluster assignment, and a survival function in a single end-to-end framework. Two variants are provided:

- **`CONVERSE_single`** — a single autoencoder with a shared encoder/decoder and a cluster-guided survival head.
- **`CONVERSE_siamese`** — a siamese (dual-encoder) autoencoder where two views of the input are encoded separately, and cross-view contrastive losses enforce cluster consistency.

Both variants support:
- Optional **variational**, VAE-style reparameterization with KL divergence.
- **Student-t kernel** soft cluster assignments.
- **Cluster-specific survival heads** or a **single shared head**.
- Multiple clustering initialisation strategies: `kmeans`, `agglomerative`, `GaussianMixture`, `SpectralClustering`.
- A **DeepHit-style** discrete-time survival head with NLL + pairwise ranking loss.


## Baselines

CONVERSE is benchmarked against the following survival models:

| Category | Models |
|---|---|
| Statistical | CoxPH, AFT, Logistic Hazard, PCHazard |
| Machine Learning | Survival Forest, XGBoost-Cox, DeepSurv, DeepHit |
| Cluster-based | SCA, VaDeSC, DeepCoxMixtures, DVCSurv |


## Datasets

Experiments are run on eight publicly available survival datasets:

`aids`, `breast_cancer`, `gbsg`, `metabric`, `pbc`, `tcga_brca`, `veterans`, `whas`


## Installation

### Prerequisites
- Python **3.12** or newer (but <3.14).
- CUDA-enabled GPU (optional, but recommended for neural models).

### Install with Conda and Poetry

1. Create a new Conda environment (recommended):
   ```bash
   conda env create -f environment.yaml
   conda activate survbase
   ```

2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```

This will create a virtual environment and install all required dependencies.



## Usage

The main entrypoint is `main.py`, which orchestrates experiments with Optuna-based hyperparameter optimisation (TPE or NSGA-II sampler).

### Run with YAML config

```yaml
# config.yaml
datasets: ["aids", "breast_cancer", "gbsg", "metabric", "pbc", "tcga_brca", "veterans", "whas"]
models: ["converse_single", "converse_siamese"]
seed: 42
n_bootstrap_iters: 20
val_size: 0.2
test_size: 0.2
n_trials: 300
n_startup_trials: 30
optimization_metrics: ["cindex", "ibs"]
pruning: true
n_jobs: -1
space_dir: "spaces"
output: "results_{dataset}_{time}.csv"
log_level: "INFO"
save_final_model: false
```

```bash
python main.py --config config.yaml
```

### Key Configuration Options

| Parameter | Description | Default |
|---|---|---|
| `datasets` | List of datasets to run | — |
| `models` | Models to benchmark | — |
| `seed` | Base random seed | `42` |
| `n_bootstrap_iters` | Bootstrap iterations for test evaluation | `20` |
| `val_size` / `test_size` | Validation / test split fractions | `0.2` |
| `n_trials` | Number of Optuna trials | `300` |
| `optimization_metrics` | Multi-objective targets (`cindex`, `ibs`) | — |
| `pruning` | Enable Optuna median pruner | `true` |
| `space_dir` | Directory with per-model YAML search spaces | `spaces/` |
| `save_final_model` | Persist the best model to `saved_models/` as `.joblib` | `false` |

Per-model hyperparameter search spaces are defined in `spaces/<model>.yaml`. Best found parameters per dataset are stored under `spaces_best_param/<dataset>/`.

## Outputs

Running experiments produces:

- **CSV summary** (`results_{dataset}_{time}.csv`) — one row per trial with C-index, IBS, and timings.
- **Detailed CSV** (`results_{dataset}_{time}_detailed.csv`) — per-bootstrap-iteration breakdown.
- **Saved models** (`saved_models/`) — fitted model objects (`.joblib`) for the best trial per dataset.
- **Intermediate checkpoints** — results are persisted incrementally, so progress is not lost on interruption.

## Visualization & Interpretability

Two Jupyter notebooks are provided under `notebooks/`:

- `visualize_results.ipynb` — aggregate results across datasets, filter invalid metrics, plot C-index and IBS comparisons across models.
- `interpretability_results.ipynb` — inspect learned cluster assignments, latent representations, and cluster-specific survival curves.

```bash
jupyter notebook notebooks/visualize_results.ipynb
```
