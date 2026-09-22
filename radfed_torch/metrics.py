from __future__ import annotations

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("accuracy inputs must be nonempty and equally sized")
    return float(np.mean(y_true == y_pred))


def weighted_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("F1 inputs must be nonempty and equally sized")
    weighted = 0.0
    for label in np.unique(y_true):
        true_positive = np.sum((y_true == label) & (y_pred == label))
        false_positive = np.sum((y_true != label) & (y_pred == label))
        false_negative = np.sum((y_true == label) & (y_pred != label))
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else 2.0 * true_positive / denominator
        weighted += np.sum(y_true == label) * f1
    return float(weighted / len(y_true))


def binary_auc_score(y_true: np.ndarray, positive_score: np.ndarray) -> float:
    """Compute ROC AUC using average ranks, including score ties."""

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    scores = np.asarray(positive_score, dtype=float).reshape(-1)
    if len(y_true) == 0 or len(y_true) != len(scores):
        raise ValueError("AUC inputs must be nonempty and equally sized")
    positives = y_true == 1
    num_positive = int(np.sum(positives))
    num_negative = len(y_true) - num_positive
    if num_positive == 0 or num_negative == 0:
        raise ValueError("AUC requires both positive and negative examples")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(np.sum(ranks[positives]))
    return (
        positive_rank_sum - num_positive * (num_positive + 1) / 2.0
    ) / (num_positive * num_negative)
