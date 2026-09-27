"""V2 evaluation and threshold-selection utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score


def f05(y_true, y_pred) -> float:
    """Binary F0.5 score."""
    return float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))


def threshold_sweep(
    y_true,
    probabilities,
    *,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate a probability threshold grid using precision-heavy F0.5."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.951, 0.01), 2)

    rows = []
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(
                    precision_score(y_true, pred, zero_division=0)
                ),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f0_5": f05(y_true, pred),
                "predicted_positive_pairs": int(pred.sum()),
            }
        )

    return pd.DataFrame(rows)


def select_threshold(results: pd.DataFrame) -> float:
    """Select the threshold with maximum validation F0.5.

    Ties are resolved toward the higher threshold to prefer precision.
    """
    ordered = results.sort_values(
        ["f0_5", "threshold"], ascending=[False, False]
    )
    return float(ordered.iloc[0]["threshold"])


def apply_threshold(probabilities, threshold: float) -> np.ndarray:
    """Convert pair probabilities into binary match decisions."""
    return (np.asarray(probabilities) >= threshold).astype(int)


def macro_f05_by_entity(
    truth: dict[str, set[str]],
    predictions: dict[str, set[str]],
) -> float:
    """Compute macro F0.5 across S1 entities.

    An empty prediction is valid; an empty truth with an empty prediction
    receives 1.0, matching the challenge's stated no-match convention.
    """
    scores = []
    for entity_id, true_ids in truth.items():
        pred_ids = predictions.get(entity_id, set())
        if not true_ids and not pred_ids:
            scores.append(1.0)
            continue

        tp = len(true_ids & pred_ids)
        fp = len(pred_ids - true_ids)
        fn = len(true_ids - pred_ids)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        if precision == 0.0 and recall == 0.0:
            scores.append(0.0)
        else:
            scores.append(
                (1.25 * precision * recall) / (0.25 * precision + recall)
            )

    return float(np.mean(scores)) if scores else 0.0
