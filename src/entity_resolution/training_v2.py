"""V2 training utilities: GT parsing, hard-negative construction, and S1-group split."""
from __future__ import annotations

from collections import defaultdict
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def parse_matched_entity_ids(value) -> list[str]:
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def build_training_labels(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    s1_id_col: str = "source1_entity_id",
    matched_col: str = "matched_entity_ids",
    source_id_col: str = "source_entity_id",
) -> pd.DataFrame:
    """Label only generated candidate pairs; all other candidates are hard negatives."""
    gt = {
        str(row[s1_id_col]): set(parse_matched_entity_ids(row[matched_col]))
        for _, row in ground_truth.iterrows()
    }
    out = candidates.copy()
    out["label"] = [
        int(str(s1_id) in gt and str(source_id) in gt[str(s1_id)])
        for s1_id, source_id in zip(out[s1_id_col], out[source_id_col])
    ]
    return out


def split_by_s1_entity(
    pairs: pd.DataFrame,
    *,
    group_col: str = "source1_entity_id",
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split candidate pairs so no S1 entity appears in both partitions."""
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=validation_size, random_state=random_state
    )
    train_idx, valid_idx = next(splitter.split(pairs, groups=pairs[group_col]))
    return pairs.iloc[train_idx].reset_index(drop=True), pairs.iloc[valid_idx].reset_index(drop=True)

def select_hard_negatives(
    pairs: pd.DataFrame,
    *,
    label_col: str = "label",
    negative_ratio: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Keep every positive and sample negatives per S1 group."""
    positives = pairs[pairs[label_col] == 1]
    negatives = pairs[pairs[label_col] == 0]

    if positives.empty:
        return pairs.copy()

    chosen = []
    for s1_id, pos_group in positives.groupby("source1_entity_id"):
        n = max(1, len(pos_group) * negative_ratio)
        neg_group = negatives[negatives["source1_entity_id"] == s1_id]

        if len(neg_group) > n:
            neg_group = neg_group.sample(
                n=n,
                random_state=random_state,
            )

        chosen.append(neg_group)

    selected_negatives = (
        pd.concat(chosen)
        if chosen
        else negatives.iloc[0:0]
    )

    return (
        pd.concat([positives, selected_negatives], ignore_index=True)
        .drop_duplicates()
    )