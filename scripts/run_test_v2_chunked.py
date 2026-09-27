"""Memory-safe V2 test inference using source chunks."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from entity_resolution.data_loader import load_source
from entity_resolution.preprocessing_v2 import prepare_dataframe_v2
from entity_resolution.blocking_v2 import (
    make_indexes_v2,
    candidates_for_row_v2,
)
from entity_resolution.features_v2 import pair_features_v2
from entity_resolution.model_v2 import (
    load_model,
    predict_probabilities,
)
from entity_resolution.evaluation_v2 import apply_threshold


def process_chunk(
    source,
    source_name,
    s1_lookup,
    indexes,
    model,
    threshold,
):
    pairs = []
    features = []

    for row in source.itertuples(index=False):
        candidate_ids, _ = candidates_for_row_v2(
            row,
            indexes,
        )

        source_id = str(row.entity_id)

        for s1_id in candidate_ids:
            left = s1_lookup.get(
                str(s1_id)
            )

            if left is None:
                continue

            pairs.append(
                (
                    str(s1_id),
                    source_id,
                    source_name,
                )
            )

            features.append(
                pair_features_v2(
                    left._asdict(),
                    row._asdict(),
                )
            )

    if not pairs:
        return pd.DataFrame(), {}

    pair_df = pd.DataFrame(
        pairs,
        columns=[
            "source1_entity_id",
            "source_entity_id",
            "source",
        ],
    )

    feature_df = pd.DataFrame(
        features
    )

    probabilities = predict_probabilities(
        model,
        feature_df,
    )

    matches = apply_threshold(
        probabilities,
        threshold,
    )

    pair_df["probability"] = probabilities
    pair_df["is_match"] = matches

    matched = {}

    for row in pair_df.loc[
        pair_df["is_match"] == 1
    ].itertuples(index=False):

        matched.setdefault(
            str(row.source1_entity_id),
            set(),
        ).add(
            str(row.source_entity_id)
        )

    return pair_df, matched


def append_tsv(
    df: pd.DataFrame,
    path: Path,
    *,
    header: bool,
):
    if df.empty:
        return

    df.to_csv(
        path,
        sep="\t",
        index=False,
        mode="a",
        header=header,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--s1",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--s2",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--s3",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--model",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--threshold",
        type=float,
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_v2"),
    )

    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--s1-limit", type=int, default=None)
    ap.add_argument("--s2-limit", type=int, default=None)
    ap.add_argument("--s3-limit", type=int, default=None)

    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_path = (
        out / "candidate_pairs.tsv"
    )

    score_path = (
        out / "test_pair_scores.tsv"
    )

    matching_path = (
        out / "matching_results.tsv"
    )

    # Remove old inference artifacts so the new
    # run starts cleanly.
    for path in [
        candidate_path,
        score_path,
        matching_path,
    ]:
        if path.exists():
            path.unlink()

    print("Loading S1...")

    s1 = prepare_dataframe_v2(
        load_source(args.s1, nrows=args.s1_limit)
    )

    print(
        f"S1 rows: {len(s1):,}"
    )

    print(
        "Building S1 indexes..."
    )

    indexes = make_indexes_v2(
        s1
    )

    s1_lookup = {
        str(row.entity_id): row
        for row in s1.itertuples(
            index=False
        )
    }

    print("Loading model...")

    model = load_model(
        args.model
    )

    all_matches = {}

    candidate_header_written = False
    score_header_written = False

    for source_name, source_path in [
        ("s2", args.s2),
        ("s3", args.s3),
    ]:

        print()
        print(
            f"Processing {source_name.upper()}..."
        )

        source_reader = pd.read_csv(
            source_path,
            sep="\t",
            chunksize=args.chunk_size,
            nrows=(args.s2_limit if source_name == "s2" else args.s3_limit),
            dtype=str,
            keep_default_na=False,
        )

        for chunk_no, raw_chunk in enumerate(
            source_reader,
            start=1,
        ):

            print(
                f"{source_name.upper()} chunk "
                f"{chunk_no}: "
                f"{len(raw_chunk):,} rows"
            )

            source = prepare_dataframe_v2(
                raw_chunk
            )

            pair_df, matched = process_chunk(
                source,
                source_name,
                s1_lookup,
                indexes,
                model,
                args.threshold,
            )

            for s1_id, ids in matched.items():
                all_matches.setdefault(
                    s1_id,
                    set(),
                ).update(ids)

            if not pair_df.empty:

                candidate_df = pair_df[
                    [
                        "source1_entity_id",
                        "source_entity_id",
                        "source",
                    ]
                ]

                append_tsv(
                    candidate_df,
                    candidate_path,
                    header=not candidate_header_written,
                )

                append_tsv(
                    pair_df,
                    score_path,
                    header=not score_header_written,
                )

                candidate_header_written = True
                score_header_written = True

            del source
            del raw_chunk
            del pair_df
            del matched

    print()
    print(
        "Writing submission..."
    )

    # Exactly one S1 row for every S1 entity.
    rows = []

    for s1_id in s1.entity_id.astype(str):

        ids = sorted(
            all_matches.get(
                s1_id,
                set(),
            )
        )

        rows.append(
            {
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(ids),
            }
        )

    matching_results = pd.DataFrame(
        rows
    )

    matching_results.to_csv(
        matching_path,
        sep="\t",
        index=False,
    )

    print(
        f"Wrote {len(matching_results):,} S1 rows to "
        f"{matching_path}"
    )

    if candidate_path.exists():
        print(
            f"Candidate pairs written to "
            f"{candidate_path}"
        )

    if score_path.exists():
        print(
            f"Pair scores written to "
            f"{score_path}"
        )


if __name__ == "__main__":
    main()