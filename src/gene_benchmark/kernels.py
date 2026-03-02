from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors


def is_spd(mat: np.ndarray, tol: float = 1e-8) -> bool:
    return np.allclose(mat, mat.T, atol=tol) and np.all(np.linalg.eigvalsh(mat) > tol)


def make_psd(mat: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    mat = (mat + mat.T) / 2
    eigvals = np.linalg.eigvalsh(mat)
    if np.min(eigvals) < min_eig:
        mat += np.eye(mat.shape[0]) * (min_eig - np.min(eigvals))
    return mat


def normalize_kernel(k: np.ndarray) -> np.ndarray:
    diag = np.sqrt(np.diag(k))
    diag[diag == 0] = 1e-8
    return k / (diag[:, None] * diag[None, :])


def process_kernel(mat: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, tol, None)
    k_log = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T
    return 0.5 * (k_log + k_log.T)


def compute_rbf_kernels(
    x_feature: np.ndarray, feature_id: str, save_dir: Path, compute_log: bool, ratios=(2, 4, 8)
) -> Tuple[str, Dict[int, Tuple[Path, Path]]]:
    save_dir.mkdir(parents=True, exist_ok=True)
    nbrs = NearestNeighbors(n_neighbors=2).fit(x_feature)
    distances, _ = nbrs.kneighbors(x_feature)
    avg_nn_dist = np.mean(distances[:, 1])

    paths: Dict[int, Tuple[Path, Path]] = {}
    for ratio in ratios:
        gamma = 1 / (ratio * avg_nn_dist**2)
        k_full = rbf_kernel(x_feature, x_feature, gamma=gamma)
        k_full = 0.5 * (k_full + k_full.T)

        k_path = save_dir / f"{feature_id}_K_{ratio}_{gamma}.pkl"
        with open(k_path, "wb") as f:
            pickle.dump(k_full, f)

        log_path = None
        if compute_log:
            logm_k = process_kernel(k_full)
            log_path = save_dir / f"{feature_id}_logK_{ratio}_{gamma}.pkl"
            with open(log_path, "wb") as f:
                pickle.dump(logm_k, f)
        paths[ratio] = (k_path, log_path)
    return feature_id, paths
