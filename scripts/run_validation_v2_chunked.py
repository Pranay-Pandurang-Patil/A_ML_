"""Memory-safe V2 validation runner.

Processes S2/S3 in bounded chunks instead of loading all sources
and all candidate features into RAM at once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from entity_resolution.data_loader import (
    load_source,
    load_source_chunks,
    load_ground_truth,
)
from entity_resolution.preprocessing_v2 import prepare_dataframe_v2
from entity_resolution.blocking_v2 import (
    make_indexes_v2,
    candidates_for_row_v2,
)
from entity_resolution.features_v2 import pair_features_v2
from entity_resolution.training_v2 import (
    build_training_labels,
    split_by_s1_entity,
    select_hard_negatives,
)
from entity_resolution.model_v2 import (
    train_model,
    predict_probabilities,
    save_model,
)
from entity_resolution.evaluation_v2 import (
    threshold_sweep,
    select_threshold,
    macro_f05_by_entity,
)


RANDOM_STATE = 42
CHUNK_SIZE = 25000
MAX_TRAIN_ROWS = 200000


def prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    return prepare_dataframe_v2(chunk)


def build_candidate_features(
    source_chunk: pd.DataFrame,
    s1_map: dict,
    indexes: dict,
    max_pairs: int | None = None,
):
    pair_rows = []
    feature_rows = []

    for row in source_chunk.itertuples(index=False):
        candidate_ids, _ = candidates_for_row_v2(row, indexes)

        for s1_id in candidate_ids:
            left = s1_map.get(str(s1_id))

            if left is None:
                continue

            pair_rows.append(
                (
                    str(s1_id),
                    str(row.entity_id),
                )
            )

            feature_rows.append(
                pair_features_v2(
                    left._asdict(),
                    row._asdict(),
                )
            )

            if max_pairs is not None and len(pair_rows) >= max_pairs:
                return (
                    pd.DataFrame(
                        pair_rows,
                        columns=[
                            "source1_entity_id",
                            "source_entity_id",
                        ],
                    ),
                    pd.DataFrame(feature_rows),
                )

    return (
        pd.DataFrame(
            pair_rows,
            columns=[
                "source1_entity_id",
                "source_entity_id",
            ],
        ),
        pd.DataFrame(feature_rows),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--s1",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--s2",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--s3",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_v2"),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
    )

    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=MAX_TRAIN_ROWS,
    )

    args = parser.parse_args()

    out = args.output_dir
    (out / "model").mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading S1...")

    s1 = prepare_dataframe_v2(
        load_source(args.s1)
    )

    print(
        f"S1 rows: {len(s1):,}"
    )

    print("Building S1 blocking indexes...")

    indexes = make_indexes_v2(s1)

    s1_map = {
        str(row.entity_id): row
        for row in s1.itertuples(index=False)
    }

    print("Loading ground truth...")

    gt = load_ground_truth(
        args.ground_truth
    )

    gt_map = {
        str(row.source1_entity_id): {
            x.strip()
            for x in str(
                row.matched_entity_ids
            ).split(",")
            if x.strip()
        }
        for row in gt.itertuples(index=False)
    }

    sampled_pairs = []
    sampled_features = []

    source_paths = [
        ("s2", args.s2),
        ("s3", args.s3),
    ]

    blocking_metrics = []

    for source_name, source_path in source_paths:

        print()
        print(
            f"Processing {source_name.upper()} in chunks..."
        )

        total_candidates = 0
        total_source_rows = 0
        total_s1_with_candidates = set()

        for chunk_no, chunk in enumerate(
            load_source_chunks(
                source_path,
                chunksize=args.chunk_size,
            ),
            start=1,
        ):

            print(
                f"{source_name.upper()} chunk {chunk_no}: "
                f"{len(chunk):,} rows"
            )

            source_chunk = prepare_chunk(chunk)

            pairs, features = build_candidate_features(
                source_chunk,
                s1_map,
                indexes,
            )

            if pairs.empty:
                total_source_rows += len(chunk)
                continue

            pairs["source"] = source_name

            total_candidates += len(pairs)
            total_source_rows += len(chunk)

            total_s1_with_candidates.update(
                pairs["source1_entity_id"].astype(str)
            )

            labeled = build_training_labels(
                pairs[
                    [
                        "source1_entity_id",
                        "source_entity_id",
                    ]
                ],
                gt,
            )

            labeled = labeled.reset_index(
                drop=True
            )

            features = features.reset_index(
                drop=True
            )

            combined = pd.concat(
                [
                    labeled,
                    features,
                ],
                axis=1,
            )

            # Keep every positive and a bounded amount
            # of hard negatives.
            positives = combined[
                combined["label"] == 1
            ]

            negatives = combined[
                combined["label"] == 0
            ]

            if len(negatives) > len(positives) * 5:
                negatives = negatives.sample(
                    n=max(
                        1,
                        len(positives) * 5,
                    ),
                    random_state=RANDOM_STATE,
                )

            selected = pd.concat(
                [
                    positives,
                    negatives,
                ],
                ignore_index=True,
            ).drop_duplicates()

            remaining = (
                args.max_train_rows
                - sum(
                    len(x)
                    for x in sampled_pairs
                )
            )

            if remaining <= 0:
                break

            selected = selected.head(
                remaining
            )

            sampled_pairs.append(
                selected[
                    [
                        "source1_entity_id",
                        "source_entity_id",
                        "label",
                    ]
                ]
            )

            sampled_features.append(
                selected[
                    [
                        c
                        for c in features.columns
                    ]
                ]
            )

            del source_chunk
            del pairs
            del features
            del combined
            del labeled
            del positives
            del negatives
            del selected

        blocking_metrics.append(
            {
                "source": source_name,
                "source_rows": total_source_rows,
                "candidate_pairs": total_candidates,
                "s1_with_candidates": len(
                    total_s1_with_candidates
                ),
                "avg_candidates_per_source_row": (
                    total_candidates
                    / total_source_rows
                    if total_source_rows
                    else 0.0
                ),
            }
        )

        if (
            sum(
                len(x)
                for x in sampled_pairs
            )
            >= args.max_train_rows
        ):
            break

    if not sampled_pairs:
        raise RuntimeError(
            "No candidate pairs were generated."
        )

    print()
    print(
        "Combining bounded training sample..."
    )

    pairs = pd.concat(
        sampled_pairs,
        ignore_index=True,
    )

    features = pd.concat(
        sampled_features,
        ignore_index=True,
    )

    print(
        "Training candidate rows retained: "
        f"{len(pairs):,}"
    )

    labeled_features = pd.concat(
        [
            pairs[
                [
                    "source1_entity_id",
                    "source_entity_id",
                    "label",
                ]
            ],
            features,
        ],
        axis=1,
    )

    print("Splitting by S1 entity...")

    train_pairs, valid_pairs = split_by_s1_entity(
        labeled_features,
        random_state=RANDOM_STATE,
    )

    train_pairs = select_hard_negatives(
        train_pairs,
        negative_ratio=5,
        random_state=RANDOM_STATE,
    )

    feature_cols = list(
        features.columns
    )

    X_train = train_pairs[
        feature_cols
    ]

    y_train = train_pairs[
        "label"
    ]

    X_valid = valid_pairs[
        feature_cols
    ]

    y_valid = valid_pairs[
        "label"
    ]

    print(
        f"Train rows: {len(X_train):,}"
    )

    print(
        f"Validation rows: {len(X_valid):,}"
    )

    if y_train.nunique() < 2:
        raise RuntimeError(
            "Training data contains only one class."
        )

    print("Training XGBoost...")

    model = train_model(
        X_train,
        y_train,
        random_state=RANDOM_STATE,
        model_type="xgboost",
    )

    print(
        "Predicting validation probabilities..."
    )

    valid_prob = predict_probabilities(
        model,
        X_valid,
    )

    print("Running threshold sweep...")

    sweep = threshold_sweep(
        y_valid,
        valid_prob,
    )

    threshold = select_threshold(
        sweep
    )

    valid_pred = (
        valid_prob >= threshold
    ).astype(int)

    predictions = {}

    for (
        s1_id,
        source_id,
    ), pred in zip(
        valid_pairs[
            [
                "source1_entity_id",
                "source_entity_id",
            ]
        ].itertuples(
            index=False,
            name=None,
        ),
        valid_pred,
    ):

        if pred:
            predictions.setdefault(
                str(s1_id),
                set(),
            ).add(
                str(source_id)
            )

    macro = macro_f05_by_entity(
        gt_map,
        predictions,
    )

    metrics = {
        "macro_f0_5": macro,
        "selected_threshold": threshold,
        "training_rows": int(
            len(X_train)
        ),
        "validation_rows": int(
            len(X_valid)
        ),
        "candidate_sample_rows": int(
            len(pairs)
        ),
        "random_state": RANDOM_STATE,
        "chunk_size": args.chunk_size,
        "candidate_metrics": blocking_metrics,
        "model_type": "xgboost",
    }

    sweep.to_csv(
        out / "threshold_results.csv",
        index=False,
    )

    pd.DataFrame(
        blocking_metrics
    ).to_csv(
        out / "blocking_metrics.csv",
        index=False,
    )

    (out / "validation_metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_model(
        model,
        out / "model" / "model.joblib",
    )

    (out / "model" / "threshold.txt").write_text(
        str(threshold),
        encoding="utf-8",
    )

    # Save feature importance for XGBoost.
    # Keep Logistic Regression compatibility as fallback.
    if hasattr(
        model,
        "feature_importances_",
    ):
        importance = model.feature_importances_

        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": importance,
            }
        ).sort_values(
            "importance",
            ascending=False,
        ).to_csv(
            out / "feature_importance.csv",
            index=False,
        )

    else:
        coef = model.named_steps[
            "classifier"
        ].coef_[0]

        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": abs(coef),
                "coefficient": coef,
            }
        ).sort_values(
            "importance",
            ascending=False,
        ).to_csv(
            out / "feature_importance.csv",
            index=False,
        )

    print()

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()