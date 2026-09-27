import sys

sys.path.insert(0, "src")

from entity_resolution.data_loader import load_source, load_ground_truth
from entity_resolution.preprocessing_v2 import prepare_dataframe_v2
from entity_resolution.blocking_v2 import (
    make_indexes_v2,
    candidates_for_row_v2,
)


S1_PATH = "data/train/train_source1.tsv"
S2_PATH = "data/train/train_source2.tsv"
GT_PATH = "data/train/train_ground_truth.tsv"

S1_SAMPLE_SIZE = 25000
S2_SAMPLE_SIZE = 1000


def main():
    print("Loading ground truth...")

    gt = load_ground_truth(GT_PATH)

    # Use only GT rows that have at least one matched entity.
    gt_rows = []

    for _, row in gt.iterrows():
        matched_ids = {
            x.strip()
            for x in str(row["matched_entity_ids"]).split(",")
            if x.strip()
        }

        if matched_ids:
            gt_rows.append(
                (
                    str(row["source1_entity_id"]),
                    matched_ids,
                )
            )

    if not gt_rows:
        raise RuntimeError(
            "No matched GT pairs were found."
        )

    # Take a deterministic sample of GT-linked S1 entities.
    sample_count = min(
        S1_SAMPLE_SIZE,
        len(gt_rows),
    )

    sampled_gt = gt_rows[:sample_count]

    target_s1_ids = {
        s1_id
        for s1_id, _ in sampled_gt
    }

    print(
        f"GT-linked S1 entities sampled: "
        f"{len(target_s1_ids):,}"
    )

    print("Loading S1...")

    # Read S1 in chunks so we can retrieve the exact
    # GT-referenced entities without loading duplicate
    # full copies unnecessarily.
    import pandas as pd

    s1_parts = []

    for chunk in pd.read_csv(
        S1_PATH,
        sep="\t",
        chunksize=100000,
        dtype=str,
        keep_default_na=False,
    ):
        if "entity_id" not in chunk.columns:
            raise RuntimeError(
                "S1 file does not contain entity_id."
            )

        matched = chunk[
            chunk["entity_id"].astype(str).isin(
                target_s1_ids
            )
        ]

        if not matched.empty:
            s1_parts.append(matched)

        found = sum(
            len(part)
            for part in s1_parts
        )

        if found >= len(target_s1_ids):
            break

    if not s1_parts:
        raise RuntimeError(
            "None of the sampled GT S1 entities "
            "were found in S1."
        )

    s1_raw = pd.concat(
        s1_parts,
        ignore_index=True,
    )

    s1 = prepare_dataframe_v2(
        s1_raw
    )

    print(
        f"S1 sampled rows loaded: "
        f"{len(s1):,}"
    )

    print("Building S1 indexes...")

    indexes = make_indexes_v2(s1)

    print("Loading S2 sample...")

    s2 = prepare_dataframe_v2(
        load_source(
            S2_PATH,
            nrows=S2_SAMPLE_SIZE,
        )
    )

    s2_map = {
        str(row.entity_id): row
        for row in s2.itertuples(index=False)
    }

    checked = 0
    captured = 0

    for s1_id, matched_ids in sampled_gt:

        # Only check GT matches whose S2 entity is
        # present in the bounded S2 sample.
        for matched_id in matched_ids:

            if matched_id not in s2_map:
                continue

            checked += 1

            candidates, reasons = candidates_for_row_v2(
                s2_map[matched_id],
                indexes,
            )

            if s1_id in candidates:
                captured += 1

                print(
                    f"CAPTURED: {s1_id} -> {matched_id}"
                )
            else:
                print(
                    f"MISSED:   {s1_id} -> {matched_id}"
                )

    recall = (
        captured / checked
        if checked
        else 0.0
    )

    print()
    print(
        "GT PAIRS IN SAMPLE:",
        checked,
    )
    print(
        "CAPTURED:",
        captured,
    )
    print(
        "RECALL:",
        recall,
    )


if __name__ == "__main__":
    main()