# GP_benchmark: A Multi-Model Benchmark for Gene–Disease Association Prediction

A reproducible benchmarking framework that evaluates five machine learning models for **gene–disease association (GDA) prediction** under a strict **temporal holdout** setting. Models are trained on associations published up to a cutoff year and evaluated on associations first reported afterward — simulating prospective discovery.

<p align="center">
  <img src="images/pipline3.png" alt="GP_benchmark overall pipeline" width="860"/>
</p>

**Figure:** Gene prioritization benchmark pipeline. (a) Features are collected from three source types: PPI networks, biomedical literature, and sequence; disease-gene associations are split by a 2019 cutoff into training and test sets. During training, unlabeled genes are handled by either one-class ranking, which learns a boundary around known positives, or bagging binary classification, which repeatedly samples unlabeled genes as negatives. Five model families are evaluated: SVMs, RFs, DNNs, GNNs, and MF. For multi-source integration, three fusion strategies are compared: early-level fusion concatenates features before training, mid-level fusion integrates features within the model, and late-level fusion aggregates predictions from separately trained models. Gene rankings are assessed using ranking-based metrics (Recall@k, AUROC, BEDROC@k) and gene enrichment metric BioSim@k. (b) The bagging binary strategy in detail: unlabeled genes are repeatedly sampled as negatives across bootstrap iterations, producing one classifier per iteration, and predictions are aggregated into a final ranking score.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Requirements](#3-requirements)
4. [Installation](#4-installation)
5. [Data](#5-data)
6. [Quick Start](#6-quick-start)
7. [Running the Full Benchmark](#7-running-the-full-benchmark)
8. [Models](#8-models)
9. [Input Features](#9-input-features)
10. [Evaluation Protocol](#10-evaluation-protocol)
11. [Output Format](#11-output-format)
12. [Reproducing Results](#12-reproducing-results)
13. [Citation](#13-citation)

---

## 1. Overview

### Task

Given a set of known gene–disease associations (GDAs) up to a cutoff year (e.g. 2019), rank all candidate genes for each disease so that genuinely novel associations (published after the cutoff) appear at the top of the ranked list.

### Approach

Each gene is represented by a multi-view feature vector assembled from:
- Protein–protein interaction (PPI) network embeddings
- Protein sequence embeddings
- Bioconcept (biomedical literature) embeddings

Five model families are benchmarked:

| Model | Description |
|-------|-------------|
| **Kernel method** | Kernel method with negative bagging, allow early/lid/late fusion |
| **Random Forest (RF)** | Random forest with negative bagging |
| **Matrix Factorization (MF)** | Bayesian matrix factorisation (SMURFF) with multi-view side information |
| **Neural Network (NN)** | Multi-layer perceptron with early, mid, and late feature fusion |
| **Graph Neural Network (GNN)** | GCN and GraphSAGE operating on the STRING PPI graph |

### Temporal Validation

Diseases included in evaluation must satisfy all three conditions:
- At least 15 known GDAs in total
- At least 5 associations published **before** the cutoff year (training positives)
- At least 1 association published **after** the cutoff year (test positives)

This ensures the ranking task is non-trivial and the evaluation is prospective.

---

## 2. Repository Structure

```
GP_benchmark/
├── data/                          # Input data (see §5)
│   ├── bioconcept/
│   │   └── uniport_bio_emb.csv        # Biomedical concept embeddings
│   ├── diffusion/                     # Raw inputs for diffusion kernel computation
│   │   ├── ppi_full_2019.txt          # STRING PPI edge list (required by pre_calculate_diffusion_kernels.py)
│   │   └── 2019_map.pkl               # ENSP → UniProt ID mapping + merge/delete lists
│   ├── diffusion_2019.csv             # Pre-computed diffusion node features (full)
│   ├── diffusion_2019_2.csv           # Diffusion node features (variant 2)
│   ├── diffusion_2019_pcs.csv         # PCA-reduced diffusion node features
│   ├── disgent_2020/
│   │   └── timecut/
│   │       └── dga_time_uniport.csv   # DisGeNET GDAs with publication year
│   ├── esmfold/
│   │   └── uniport_esm2.csv           # ESM-2 protein language model embeddings
│   ├── opentarget/
│   │   └── ot_dga_time_uni.csv        # OpenTargets GDAs (optional)
│   ├── ppi_full_2019_dw_emb_40.csv    # DeepWalk PPI embeddings (dim=40)
│   ├── pre_processed_features/
│   │   └── seq_emb/
│   │       └── uniport_emb.csv        # Sequence-based embeddings
│   ├── stringdb/
│   │   ├── edge_2019.csv              # STRING PPI edges (2019)
│   │   └── uniport_ppi_2019.csv       # STRING PPI node features (2019)
│   └── uniport_id/
│       └── uni2name.pkl               # UniProt ID → gene name mapping
│
├── results/                       # Model outputs (created at runtime)
│   └── dw_auc_norm/
│       ├── 2019/                      # Cached per-feature SVM kernel matrices
│       └── df/
│           └── 2019/                  # Cached diffusion kernel matrices (K and log K per beta)
│
├── logs/                          # Run logs (created at runtime)
│
├── src/                           # All source code
│   ├── config.py                  # Centralised REPO_ROOT path
│   ├── features_reindex.py        # Feature loading utilities
│   ├── main_diffusion.py          # Kernel method entry point
│   ├── main_gnn.py                # GNN entry point
│   ├── main_mf.py                 # Matrix factorisation entry point
│   ├── main_nn_non_para.py        # Neural network entry point
│   ├── main_rf.py                 # Random forest entry point
│   ├── model_diffusion.py         # Kernel method implementation
│   ├── model_gnn.py               # GNN implementation (GCN / GraphSAGE)
│   ├── model_mf.py                # MF implementation (SMURFF)
│   ├── model_nn_non_para.py       # NN implementation
│   ├── model_rf_uni_inductive.py  # RF implementation
│   ├── pre_calculate_diffusion_kernels.py  # Computes diffusion kernel matrices from the PPI graph
│   ├── run_all.sh                 # Full benchmark pipeline
│   ├── run_one_disease.sh         # One-disease test (specify an ICD10 ID)
│   └── run_smoke.sh               # Deprecated alias → run_one_disease.sh
│
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 3. Requirements

### Python version

Python ≥ 3.9 is recommended.

### Core dependencies

```
numpy
pandas
scipy
scikit-learn
rdkit          # BEDROC metric
smurff         # Bayesian matrix factorisation (MF model)
gseapy         # Gene set enrichment
```

### Deep-learning dependencies (required for NN and GNN models)

```
torch
torch-geometric
torch-sparse
```

PyTorch must be installed separately following the [official instructions](https://pytorch.org/get-started/locally/) for your CUDA version. `torch-geometric` and `torch-sparse` have corresponding version requirements — see the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Two conda/venv environments

The benchmark uses **two separate environments** due to conflicting dependency requirements:

| Environment | Models | Key packages |
|-------------|--------|--------------|
| `.venv` | Kernel method, RF, MF | `smurff`, `rdkit`, `gseapy`, `scikit-learn` |
| `new_esm_env` | NN, GNN | `torch`, `torch-geometric`, `torch-sparse` |

`run_all.sh` and `run_one_disease.sh` activate each environment automatically.

---

## 4. Installation

```bash
# Clone the repository
git clone <repo-url> GP_benchmark
cd GP_benchmark
```

**Environment 1** (Kernel method / RF / MF):

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn rdkit smurff gseapy
```

**Environment 2** (NN / GNN — install PyTorch first):

```bash
python -m venv new_esm_env
source new_esm_env/bin/activate
# Install PyTorch for your CUDA version, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-sparse
pip install numpy pandas scipy scikit-learn rdkit gseapy
```

Then update the `source` lines in `src/run_all.sh` and `src/run_one_disease.sh` to point to your environment paths.

---

## 5. Data

### Required files

All data files must be placed inside the `data/` directory. The table below lists every file required by the default `run_all.sh` configuration.

| File path (relative to repo root) | Description | Source |
|------------------------------------|-------------|--------|
| `data/ppi_full_2019_dw_emb_40.csv` | DeepWalk PPI graph embeddings (dim=40, 2019) | STRING + DeepWalk |
| `data/stringdb/uniport_ppi_2019.csv` | STRING PPI adjacency/node features (2019) | STRING DB |
| `data/stringdb/edge_2019.csv` | STRING PPI edge list (2019) — required for GNN | STRING DB |
| `data/esmfold/uniport_esm2.csv` | ESM-2 protein language model embeddings | ESMFold / Meta AI |
| `data/pre_processed_features/seq_emb/uniport_emb.csv` | Sequence-derived embeddings | Computed |
| `data/bioconcept/uniport_bio_emb.csv` | Biomedical concept embeddings | Computed |
| `data/diffusion_2019.csv` | Network diffusion node features (full) | Computed from STRING |
| `data/diffusion_2019_2.csv` | Network diffusion node features (variant 2) | Computed from STRING |
| `data/diffusion_2019_pcs.csv` | PCA-reduced diffusion node features | Computed |
| `data/disgent_2020/timecut/dga_time_uniport.csv` | DisGeNET 2020 GDAs with `first_pub_year` | [DisGeNET](https://www.disgenet.org/) |
| `data/opentarget/ot_dga_time_uni.csv` | OpenTargets GDAs (used only with `dga=opentarget`) | [Open Targets](https://www.opentargets.org/) |
| `data/uniport_id/uni2name.pkl` | UniProt ID → gene name dictionary | [UniProt](https://www.uniprot.org/) |

The following two files are required specifically by the diffusion kernel pre-computation script:

| File path (relative to repo root) | Description | Source |
|------------------------------------|-------------|--------|
| `data/diffusion/ppi_full_2019.txt` | STRING PPI edge list used to build the diffusion graph | STRING DB |
| `data/diffusion/2019_map.pkl` | Pickle of `[merge_groups, delete_ensp, map_dict_aligned]` — maps ENSP node IDs to UniProt IDs | Computed |

### Diffusion kernel cache

The kernel method (`main_diffusion.py`) uses pre-computed diffusion kernel matrices stored under `results/dw_auc_norm/df/2019/`. These are computed once from the raw PPI graph using `src/pre_calculate_diffusion_kernels.py`.

**Automatic computation:** if the cache directory is empty or missing when the kernel method runs, `model_diffusion.py` will automatically invoke `pre_calculate_diffusion_kernels.run(setting='2019')` and populate the cache before continuing. This step is compute-intensive (6 beta values × eigendecomposition of the full PPI adjacency matrix) and only needs to run once.

**Manual pre-computation** (recommended before a full benchmark run):

```bash
cd src
source /path/to/.venv/bin/activate
python pre_calculate_diffusion_kernels.py
```

The script writes six pairs of files to `results/dw_auc_norm/df/2019/` — one kernel matrix `K` and its matrix logarithm `log K` for each beta value in `{0.1, 0.2, 0.5, 0.8, 1, 2}`.

### Per-feature SVM kernel cache

`results/dw_auc_norm/2019/` stores precomputed RBF kernel matrices (one per non-diffusion feature) used by the SVM. These are computed and cached automatically on first run and reused across diseases.

### Expected CSV columns

| File | Required columns |
|------|-----------------|
| `dga_time_uniport.csv` | `disease_id`, `string_id`, `first_pub_year` |
| `ot_dga_time_uni.csv` | `disease_id`, `string_id`, `first_pub_year`, `score` |
| `edge_2019.csv` | `p1`, `p2` (STRING protein IDs) |
| All feature CSVs | `string_id`, `feature_*` (one or more feature columns) |

---

## 6. Quick Start

### Verify data paths (read-only, ~1 second)

```bash
cd GP_benchmark
python -c "
import sys; sys.path.insert(0, 'src')
from config import REPO_ROOT
required = [
    'data/ppi_full_2019_dw_emb_40.csv',
    'data/stringdb/uniport_ppi_2019.csv',
    'data/esmfold/uniport_esm2.csv',
    'data/pre_processed_features/seq_emb/uniport_emb.csv',
    'data/bioconcept/uniport_bio_emb.csv',
    'data/diffusion_2019.csv',
    'data/diffusion_2019_2.csv',
    'data/diffusion_2019_pcs.csv',
    'data/disgent_2020/timecut/dga_time_uniport.csv',
    'data/opentarget/ot_dga_time_uni.csv',
    'data/stringdb/edge_2019.csv',
    'data/uniport_id/uni2name.pkl',
    'data/diffusion/ppi_full_2019.txt',
    'data/diffusion/2019_map.pkl',
]
for f in required:
    p = REPO_ROOT / f
    print('OK     ' if p.exists() else 'MISSING', p)
"
```

### One-disease test

Verifies that every model can run end-to-end without errors, on a single disease you specify:

```bash
./src/run_one_disease.sh ICD10_M41
```

Replace `ICD10_M41` with any ICD10 disease ID present in the dataset that meets the temporal split criteria. Results are written to `results/one_disease/<disease_id>/<model>/`, logs to `logs/one_disease/`.

---

## 7. Running the Full Benchmark

```bash
./src/run_all.sh
```

The script runs all five models sequentially, switching environments automatically. Output directories and log files are created automatically.

### What `run_all.sh` executes

```
Env 1 (.venv):
  main_diffusion.py  → results/2019_df_oob_mid/        (all features)
  main_diffusion.py  → results/2019_df_oob_ppi_mid/    (PPI features only)
  main_mf.py         → results/2019_mf/
  main_rf.py         → results/2019_rf/

Env 2 (new_esm_env):
  main_nn_non_para.py → results/2019_nn_oob/           (all features)
  main_nn_non_para.py → results/2019_nn_oob_ppi/       (PPI features only)
  main_gnn.py         → results/2019_gnn/
```

Logs are written to `logs/`.

### Running a single model

Each script can be invoked directly. Arguments follow the pattern:

```bash
cd src
source /path/to/env/bin/activate

python main_diffusion.py \
    '<comma_separated_features>' \
    'results/<output_dir>' \
    <time_cutoff_year> \
    '<dga_source>'

# Example: kernel model, all features, 2019 cutoff, DisGeNET
python main_diffusion.py \
    'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019,uniport_bio,uniport_seq,uniport_esm' \
    'results/my_run' \
    2019 \
    'disgenet'
```

For the MF model two additional parameters control SMURFF sampling:

```bash
python main_mf.py '<features>' '<output_dir>' <year> '<dga>' <burnin> <num_samples>
# e.g. 200 burn-in + 500 samples (production) or 20+50 (debug)
```

---

## 8. Models

### 8.1 Kernel method (`main_diffusion.py`, `model_diffusion.py`)

For diffusion features, the kernel is derived from the network diffusion matrix via a matrix-logarithm transformation (`K_log = log(K)`). For all other features, an RBF kernel is computed from the raw feature matrix. All kernel matrices are pre-computed once and cached.

**Diffusion kernel pre-computation** (`pre_calculate_diffusion_kernels.py`): builds the normalised diffusion kernel `K = exp(-β L)` for six values of `β` (0.1, 0.2, 0.5, 0.8, 1, 2), remaps node IDs from ENSP to UniProt, and saves both `K` and `log K` to `results/dw_auc_norm/df/{year}/`. If the cache is absent when `model_diffusion.py` runs, it calls this script automatically. Manual pre-computation is recommended before a full benchmark run (see §5).

Hyperparameters (gamma ratio, regularisation `C`) are selected per feature by 3-fold stratified cross-validation using BEDROC as the primary criterion. Predictions are aggregated from **20 bootstrap bags** (random negative sampling, 5× oversampling of negatives). Overlapping train/test indices in each bag are masked before averaging.

Per-feature RBF kernel matrices are cached in `results/dw_auc_norm/{year}/` and reused across diseases.

### 8.2 Random Forest (`main_rf.py`, `model_rf_uni_inductive.py`)

RF hyperparameters are selected from 5 candidate combinations by 3-fold CV. Features are MinMax-scaled before concatenation.

### 8.3 Matrix Factorisation (`main_mf.py`, `model_mf.py`)

Bayesian matrix factorisation via [SMURFF](https://github.com/ExaScience/smurff), treating the disease–gene association matrix as a sparse tensor. Each gene's feature block is used as side information. One MF model is trained per feature type, and predictions are aggregated over **20 parallel MCMC chains** (with distinct random seeds).

Default SMURFF hyperparameters: 16 latent factors, 200 burn-in samples, 500 posterior samples.

### 8.4 Neural Network (`main_nn_non_para.py`, `model_nn_non_para.py`)

A multi-layer perceptron (PyTorch) with three fusion modes:
- **Early fusion**: all features concatenated at input
- **Mid fusion**: features concatenated at an intermediate layer
- **Late fusion**: separate forward passes per feature block, predictions averaged

Trained with 20 bootstrap bags and 5× negative oversampling. Masked-mean aggregation is used to handle variable-length masked positions across bags.

### 8.5 Graph Neural Network (`main_gnn.py`, `model_gnn.py`)

Operates directly on the STRING PPI graph (2019 snapshot). Node features are any of the supported feature sets. Graph convolution is implemented with:
- **GCN** (`GCNConv`) — spectral convolution
- **GraphSAGE** (`SAGEConv`) — inductive neighbourhood aggregation

By default, `run_all.sh` uses GCN. Edges are converted to undirected, self-loops are added, and the adjacency is stored as a `SparseTensor` for efficient message passing. 15 bootstrap bags are used.

---

## 9. Input Features

All features are loaded via `src/features_reindex.py::get_feature(root, feature_name)` and MinMax-scaled to [0, 1] per-column before use (except diffusion features, which are passed through unscaled).

| Feature name | Description | Columns |
|--------------|-------------|---------|
| `uniport_ppi_2019` | STRING PPI node features (2019) | `feature_*` |
| `ppi_2019_dw_40` | DeepWalk PPI embeddings, dim=40 (2019) | `feature_0`…`feature_39` |
| `ppi_2019_dw_10` | DeepWalk PPI embeddings, dim=10 | `feature_0`…`feature_9` |
| `ppi_2019_dw_80` | DeepWalk PPI embeddings, dim=80 | `feature_0`…`feature_79` |
| `uniport_esm` | ESM-2 protein language model embeddings | `feature_*` |
| `uniport_seq` | Sequence-derived protein embeddings | `feature_*` |
| `uniport_bio` | Biomedical concept embeddings | `feature_*` |
| `diffusion_2019` | Network diffusion feature matrix (full) | `feature_*` |
| `diffusion_2019_2` | Diffusion features (variant 2) | `feature_*` |
| `diffusion_2019_pca` | PCA-compressed diffusion features | `feature_*` |

All feature CSVs must contain a `string_id` column (STRING protein ID) used to align multiple feature sources via inner join.

---

## 10. Evaluation Protocol

### Temporal split

For a cutoff year `T` (default 2019):

```
Training positives:  GDAs with first_pub_year ≤ T
Test positives:      GDAs with first_pub_year  > T
Training negatives:  randomly sampled from genes with no known GDA
```

### Negative sampling

For each bootstrap bag, 5 × |train positives| negatives are drawn **with replacement** from all non-positive genes. This is repeated across seeds and predictions are averaged using masked mean (excluding train/test overlap).

### Metrics

All metrics are computed in `eval_bagging()` and reported per disease and per method:

| Metric | Description |
|--------|-------------|
| `top_recall_25` | Recall among the top 25 ranked genes |
| `top_recall_300` | Recall among the top 300 ranked genes |
| `top_recall_10%` | Recall within the top 10% of all genes |
| `top_precision_10%` | Precision within the top 10% of all genes |
| `max_precision_10%` | Maximum achievable precision at top-10% cutoff |
| `top_recall_30%` | Recall within the top 30% of all genes |
| `top_precision_30%` | Precision within the top 30% of all genes |
| `max_precision_30%` | Maximum achievable precision at top-30% cutoff |
| `pm_0.5%`…`pm_30%` | Early-recognition ER_n: TPR_n / (TPR_n + FPR_n) at 0.5%–30% thresholds |
| `auroc` | Area under the ROC curve |
| `rank_ratio` | Average rank of true positives / total number of genes |
| `bedroc_1`…`bedroc_30` | BEDROC at α = 160.9, 32.2, 16.1, 5.3 (emphasises early enrichment) |

BEDROC (Boltzmann-Enhanced Discrimination of ROC) strongly penalises models that rank true positives poorly in the top of the list, making it a stringent early-enrichment metric for prospective discovery.

### Disease inclusion criteria

A disease is included if and only if:

```python
len(all_gda) >= 15
len(train_positives) >= 5          # known before cutoff
len(test_positives) >= 1           # discovered after cutoff
```

---

## 11. Output Format

```
results/
└── {experiment_name}/             # e.g. results/2019_df_oob_mid
    ├── all_disease.csv            # Mean metrics across all diseases
    ├── {disease_id}.csv           # Per-disease metrics, one row per method×fold
    └── {experiment_name}_pred/
        └── {disease_id}_pred.pkl  # Prediction details (labels, gene IDs, scores)
```

### `all_disease.csv` columns

`method`, `top_recall_25`, `top_recall_300`, `top_recall_10%`, `top_precision_10%`, `max_precision_10%`, `top_recall_30%`, `top_precision_30%`, `max_precision_30%`, `pm_0.5%`, `pm_1%`, `pm_5%`, `pm_10%`, `pm_15%`, `pm_20%`, `pm_25%`, `pm_30%`, `auroc`, `rank_ratio`, `bedroc_1`, `bedroc_5`, `bedroc_10`, `bedroc_30`, `disease`

### `{disease_id}_pred.pkl` structure

```python
{
    'true_label':       np.ndarray,   # binary labels for test genes
    'test_genes':       np.ndarray,   # STRING IDs of test set genes
    'train_pos_genes':  np.ndarray,   # STRING IDs of training positive genes
    '<method_name>':    np.ndarray,   # predicted scores, one per test gene
    ...
}
```

---

## 12. Reproducing Results

### Step 1: Clone and install

```bash
git clone <repo-url> GP_benchmark
cd GP_benchmark
# Install environments (see §4)
```

### Step 2: Obtain data

Obtain the required data files listed in §5 and place them under `data/` following the directory structure shown in §2.

### Step 3: Update environment paths

Edit the `source` lines in `src/run_all.sh` and `src/run_one_disease.sh` to point to your virtual environment locations.

### Step 4: Validate paths

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from config import REPO_ROOT
files = [
    'data/ppi_full_2019_dw_emb_40.csv',
    'data/stringdb/uniport_ppi_2019.csv',
    'data/esmfold/uniport_esm2.csv',
    'data/pre_processed_features/seq_emb/uniport_emb.csv',
    'data/bioconcept/uniport_bio_emb.csv',
    'data/diffusion_2019.csv', 'data/diffusion_2019_2.csv', 'data/diffusion_2019_pcs.csv',
    'data/disgent_2020/timecut/dga_time_uniport.csv',
    'data/opentarget/ot_dga_time_uni.csv',
    'data/stringdb/edge_2019.csv',
    'data/uniport_id/uni2name.pkl',
    'data/diffusion/ppi_full_2019.txt',
    'data/diffusion/2019_map.pkl',
]
for f in files:
    p = REPO_ROOT / f
    print('OK' if p.exists() else 'MISSING', p)
"
```

### Step 5: One-disease test

```bash
./src/run_one_disease.sh ICD10_M41
```

Check `logs/one_disease/` for any `[FAIL]` entries and resolve before running the full benchmark.

### Step 6: Full benchmark

```bash
./src/run_all.sh
```

Results are written to `results/2019_*/`. Compare `all_disease.csv` across experiment directories to reproduce Table comparisons.

### Randomness and reproducibility

All bootstrap bags use fixed seeds (`base_seed + i` for `i` in `range(num_bags)`). The default `base_seed = 42`. To change the seed, modify `base_seed` in the relevant `main_*.py` file.

---

## 13. Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{GP_benchmark,
  title        = {{GP\_benchmark}: A Multi-Model Benchmark for Gene--Disease Association Prediction},
  author       = {},
  year         = {2025},
  howpublished = {\url{<repo-url>}},
}
```

---

## License

To be specified.
