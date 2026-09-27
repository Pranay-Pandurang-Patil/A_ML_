import numpy as np


def f05_single(true_ids, predicted_ids):
    true_set = set(true_ids)
    pred_set = set(predicted_ids)

    if not true_set and not pred_set:
        return 1.0
    if not pred_set:
        return 0.0

    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set) if true_set else 0.0

    if precision == 0 and recall == 0:
        return 0.0

    beta2 = 0.25
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def macro_f05(y_true, y_pred):
    return float(np.mean([f05_single(t, p) for t, p in zip(y_true, y_pred)]))


def parse_ground_truth(df):
    lookup = {}
    for row in df.itertuples(index=False):
        value = row.matched_entity_ids or ""
        lookup[row.source1_entity_id] = [x for x in value.split(",") if x]
    return lookup
