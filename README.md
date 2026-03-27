# GP-Benchmark

A unified benchmarking framework for **gene prioritisation (GP)** — predicting which genes are causally associated with a given disease.

## Overview

This repository provides a reproducible evaluation pipeline that:

- Uses **time-based data splitting**: train on disease–gene associations published before year *T*, test on associations published after *T*
- Applies **ranking-aware metrics** (BEDROC, AUROC, rank ratio, top-recall) to reflect realistic discovery scenarios
- Benchmarks five complementary ML paradigms across multiple gene-feature representations

![Pipeline](images/pipline3.png)

## Repository layout

```
GP_benchmark/
├── data/                              # Input features and disease–gene associations
│   ├── disgent_2020/timecut/          # DisGeNET associations with publication years
│   ├── stringdb/                      # PPI network edges and node embeddings
│   ├── esmfold/                       # ESM-2 protein language-model embeddings
│   ├── bioconcept/                    # Text-mining (BioConcept) embeddings
│   └── pre_processed_features/        # Sequence and expression embeddings
├── src/                               # Source code
│   ├── features_reindex.py            # Feature loader — maps feature names to CSV files
│   ├── main_kernel.py                 # Diffusion SVM pipeline
│   ├── main_rf.py                     # Random Forest pipeline
│   ├── main_nn.py                     # Neural Network pipeline
│   ├── main_gnn.py                    # Graph Neural Network pipeline (GCN / GraphSAGE)
│   ├── main_mf.py                     # Matrix Factorisation pipeline (SMURFF)
│   ├── model_diffusion.py             # Kernel computation and SVM training
│   ├── model_rf_uni_inductive.py      # RF early / late fusion implementations
│   ├── model_nn_non_para.py           # MLP early / mid / late fusion implementations
│   ├── model_gnn.py                   # GCN and GraphSAGE implementations
│   ├── model_mf.py                    # Bayesian matrix factorisation (SMURFF) wrapper
│   └── run_all.sh                     # Example script to run all pipelines
├── results/                           # Output directory (created at runtime)
├── images/
├── requirements.txt
└── README.md
```

## Methods

| Script | Method | Fusion strategies |
|---|---|---|
| `main_kernel.py` | Diffusion SVM | — |
| `main_rf.py` | Random Forest | early, later |
| `main_nn.py` | MLP | early, mid, later |
| `main_gnn.py` | GCN / GraphSAGE | per feature |
| `main_mf.py` | Bayesian MF (SMURFF) | per feature |

## Features

Feature types are identified by a short name passed via the command line:

| Name | Description |
|---|---|
| `uniport_ppi_2019` | PPI node2vec embeddings (StringDB 2019) |
| `ppi_2019_dw_40` | PPI DeepWalk embeddings, dim 40 |
| `diffusion_2019` / `diffusion_2019_2` | Network diffusion scores |
| `uniport_esm` | ESM-2 protein language model |
| `uniport_seq` | UniProt sequence embeddings |
| `uniport_bio` | BioConcept text-mining embeddings |
| `uniport_exp` | Gene2Vec expression embeddings |
| `scgpt` | scGPT cell-type embeddings |

See `src/features_reindex.py` for the full list of supported features.

## Evaluation metrics

For each disease, held-out genes (published after year *T*) are ranked against all other genes:

| Metric | Description |
|---|---|
| `auroc` | Area under the ROC curve |
| `rank_ratio` | Mean rank of true positives / total genes |
| `bedroc_1/5/10/30` | BEDROC at α for 1 %, 5 %, 10 %, 30 % early-recovery |
| `top_recall_25/300` | Recall in the top 25 / 300 predictions |
| `top_recall_10%` / `top_recall_30%` | Recall in the top 10 % / 30 % of the ranked list |
| `pm_X%` | Proportion of positives matched in the top X % |

## Installation

```bash
git clone <repo-url>
cd GP_benchmark
pip install -r requirements.txt
```

Two execution environments are required:

- **Main env** — `scikit-learn`, `smurff`, `scipy`, `pandas`, `rdkit`, `gseapy`
- **Deep-learning env** — `torch`, `torch-geometric`, `torch-sparse` (for NN and GNN scripts)

## Usage

All scripts are run from the **project root**:

```bash
python src/main_kernel.py '<features>' '<output_dir>' <year>
python src/main_rf.py     '<features>' '<output_dir>' <year>
python src/main_nn.py     '<features>' '<output_dir>' <year>
python src/main_gnn.py    '<features>' '<output_dir>' <year>
python src/main_mf.py     '<features>' '<output_dir>' <year> <burnin> <nsamples>
```

### Arguments

| Argument | Description | Example |
|---|---|---|
| `features` | Comma-separated feature names | `uniport_ppi_2019,uniport_esm` |
| `output_dir` | Results directory (relative to project root) | `results/2019_rf` |
| `year` | Time cutoff for train/test split | `2019` |
| `burnin` | MF burn-in iterations *(MF only)* | `200` |
| `nsamples` | MF sampling iterations *(MF only)* | `500` |

### Examples

```bash
# Random Forest — PPI + sequence + text features, 2019 time cutoff
python src/main_rf.py \
    'uniport_ppi_2019,ppi_2019_dw_40,uniport_bio,uniport_seq,uniport_esm,diffusion_2019_2' \
    results/2019_rf 2019

# Kernel SVM — PPI features only
python src/main_kernel.py \
    'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019' \
    results/2019_kernel 2019

# Matrix Factorisation
python src/main_mf.py \
    'uniport_ppi_2019,ppi_2019_dw_40,uniport_bio,uniport_seq,uniport_esm,diffusion_2019_pca' \
    results/2019_mf 2019 200 500
```

See `src/run_all.sh` for a complete example running all pipelines.

## Output

Each pipeline writes to `<output_dir>/`:

- `<disease_id>.csv` — per-method metric table for each evaluated disease
- `all_disease.csv` — aggregated results across all diseases
- `<output_dir>_pred/<disease_id>_pred.pkl` — raw prediction scores for downstream analysis

## Data

Place data files under `data/` following the structure expected by `src/features_reindex.py`.

Required files:

| Path | Description |
|---|---|
| `data/disgent_2020/timecut/dga_time_uniport.csv` | DisGeNET disease–gene associations with publication years |
| `data/stringdb/edge_2019.csv` | PPI edges used by the GNN pipeline |
| `data/uniport_id/uni2name.pkl` | UniProt ID → gene name mapping |

Feature CSV files as referenced in `src/features_reindex.py`.

## Disease selection criteria

A disease is included in the evaluation if:

1. ≥ 15 known associated genes overall
2. ≥ 5 known genes published *before* the time cutoff (training positives)
3. ≥ 1 gene published *after* the cutoff (test positives)

## Citation

*(Add paper reference here)*
