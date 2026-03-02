from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from sklearn.metrics import roc_auc_score


def average_rank_ratio(y_scores: np.ndarray, y_test: np.ndarray) -> float:
    """
    Average predicted rank of true positives normalized by list length.
    """
    y_scores = np.asarray(y_scores)
    y_test = np.asarray(y_test)

    sorted_indices = np.argsort(-y_scores)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(y_scores) + 1)

    true_positive_indices = np.where(y_test == 1)[0]
    if true_positive_indices.size == 0:
        return 0.0

    average_rank = np.mean(ranks[true_positive_indices])
    return round(average_rank / y_test.shape[0], 4)


def top_recall_precision(frac: float, y_scores: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float]:
    positives = np.sum(y_test == 1)
    if positives == 0:
        return 0.0, 0.0, 0.0

    cut = max(1, int(len(y_scores) * frac))
    top_indices = np.argsort(y_scores)[-cut:][::-1]
    top_labels = y_test[top_indices]

    tp = np.sum(top_labels == 1)
    recall = tp / positives
    precision = tp / len(top_indices)
    max_precision = positives / len(top_indices)
    return recall, precision, max_precision


def calculate_er_n(scores: np.ndarray, y_test: np.ndarray, n: int) -> float:
    """
    ER_n where the top n predictions are considered positive.
    """
    n = min(n, len(scores))
    if n == 0:
        return 0.0

    top_n_labels = scores[:n, 0]
    tp_n = np.sum(top_n_labels)

    total_positives = np.sum(y_test)
    total_negatives = len(y_test) - total_positives

    tpr_n = tp_n / total_positives if total_positives > 0 else 0.0
    fpr_n = (n - tp_n) / total_negatives if total_negatives > 0 else 0.0
    return tpr_n / (tpr_n + fpr_n) if (tpr_n + fpr_n) > 0 else 0.0


def eval_bagging(y_scores: np.ndarray, y_test: np.ndarray):
    rank_ratio = average_rank_ratio(y_scores, y_test)
    try:
        auroc = roc_auc_score(y_test, y_scores)
    except Exception:
        auroc = float("nan")

    scores = np.column_stack((y_test, y_scores))
    scores = scores[scores[:, 1].argsort()[::-1]]

    top_recall_10, top_precision_10, max_precision_10 = top_recall_precision(0.1, y_scores, y_test)
    top_recall_30, top_precision_30, max_precision_30 = top_recall_precision(0.3, y_scores, y_test)

    total_positives = np.sum(y_test)
    top_25_recall = np.sum(scores[:25, 0]) / total_positives if total_positives > 0 else 0.0
    top_300_recall = np.sum(scores[:300, 0]) / total_positives if total_positives > 0 else 0.0

    return np.argsort(y_scores)[::-1], (
        top_25_recall,
        top_300_recall,
        top_recall_10,
        top_precision_10,
        max_precision_10,
        top_recall_30,
        top_precision_30,
        max_precision_30,
        calculate_er_n(scores, y_test, int(0.005 * len(y_test))),
        calculate_er_n(scores, y_test, int(0.01 * len(y_test))),
        calculate_er_n(scores, y_test, int(0.05 * len(y_test))),
        calculate_er_n(scores, y_test, int(0.1 * len(y_test))),
        calculate_er_n(scores, y_test, int(0.15 * len(y_test))),
        calculate_er_n(scores, y_test, int(0.20 * len(y_test))),
        calculate_er_n(scores, y_test, int(0.25 * len(y_test))),
        calculate_er_n(scores, y_test, int(0.30 * len(y_test))),
        auroc,
        rank_ratio,
        CalcBEDROC(scores, col=0, alpha=160.9),
        CalcBEDROC(scores, col=0, alpha=32.2),
        CalcBEDROC(scores, col=0, alpha=16.1),
        CalcBEDROC(scores, col=0, alpha=5.3),
    )


def mask_mean(all_preds: Iterable[Tuple[np.ndarray, Iterable[int]]]) -> np.ndarray:
    arrays = np.stack([arr for arr, _ in all_preds])
    mask = np.zeros_like(arrays, dtype=bool)
    for i, (_, masked_positions) in enumerate(all_preds):
        if masked_positions:
            mask[i, masked_positions] = True

    keep = ~mask
    sum_arr = np.where(keep, arrays, 0).sum(axis=0)
    count_arr = keep.sum(axis=0)
    return np.divide(sum_arr, count_arr, out=np.zeros_like(sum_arr, dtype=float), where=count_arr != 0)
