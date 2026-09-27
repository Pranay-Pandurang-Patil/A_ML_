"""V2 multi-pass candidate generation using the V2 blocking indexes."""
from __future__ import annotations

from collections import Counter
import pandas as pd

from .blocking_v2 import make_indexes_v2, candidates_for_row_v2


def generate_candidates(s1: pd.DataFrame, source: pd.DataFrame):
    """Generate the union of all V2 blocks and retain blocking reasons."""
    indexes = make_indexes_v2(s1)
    rows = []
    reason_rows = []
    for row in source.itertuples(index=False):
        candidate_ids, reasons = candidates_for_row_v2(row, indexes)
        for s1_id in candidate_ids:
            source_id = str(row.entity_id)
            rows.append((str(s1_id), source_id))
            for reason in reasons.get(s1_id, ()):
                reason_rows.append((str(s1_id), source_id, reason))
    pairs = pd.DataFrame(rows, columns=["source1_entity_id", "source_entity_id"]).drop_duplicates()
    reasons = pd.DataFrame(
        reason_rows, columns=["source1_entity_id", "source_entity_id", "block"]
    ).drop_duplicates()
    return pairs.reset_index(drop=True), reasons.reset_index(drop=True)


def candidate_summary(candidates: pd.DataFrame, s1_size: int) -> dict[str, float | int]:
    counts = candidates.groupby("source1_entity_id").size() if not candidates.empty else pd.Series(dtype=int)
    return {
        "s1_rows": int(s1_size),
        "s1_with_candidates": int(counts.size),
        "candidate_pairs": int(len(candidates)),
        "avg_candidates_per_s1": float(counts.mean()) if len(counts) else 0.0,
        "max_candidates_per_s1": int(counts.max()) if len(counts) else 0,
    }


def block_summary(reasons: pd.DataFrame) -> pd.DataFrame:
    if reasons.empty:
        return pd.DataFrame(columns=["block", "candidate_pairs", "s1_with_candidates"])
    return (
        reasons.groupby("block")
        .agg(candidate_pairs=("source_entity_id", "count"), s1_with_candidates=("source1_entity_id", "nunique"))
        .reset_index()
    )
