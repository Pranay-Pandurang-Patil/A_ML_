"""Run the complete V2 validation pipeline on real TSV datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from entity_resolution.data_loader import load_ground_truth, load_source
from entity_resolution.preprocessing_v2 import prepare_dataframe_v2
from entity_resolution.candidate_generation_v2 import generate_candidates, candidate_summary, block_summary
from entity_resolution.features_v2 import pair_features_v2
from entity_resolution.training_v2 import build_training_labels, split_by_s1_entity, select_hard_negatives
from entity_resolution.model_v2 import train_model, predict_probabilities, save_model
from entity_resolution.evaluation_v2 import threshold_sweep, select_threshold, macro_f05_by_entity

RANDOM_STATE = 42


def _record_maps(s1, source):
    left = {str(r.entity_id): r for r in s1.itertuples(index=False)}
    right = {str(r.entity_id): r for r in source.itertuples(index=False)}
    return left, right


def build_feature_pairs(candidates, s1, source):
    left, right = _record_maps(s1, source)
    rows = []
    valid = []
    for row in candidates.itertuples(index=False):
        a = left.get(str(row.source1_entity_id))
        b = right.get(str(row.source_entity_id))
        if a is None or b is None:
            continue
        rows.append(pair_features_v2(a._asdict(), b._asdict()))
        valid.append((str(row.source1_entity_id), str(row.source_entity_id)))
    features = pd.DataFrame(rows)
    pairs = pd.DataFrame(valid, columns=["source1_entity_id", "source_entity_id"])
    return pairs, features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", type=Path, required=True)
    ap.add_argument("--s2", type=Path, required=True)
    ap.add_argument("--s3", type=Path, required=True)
    ap.add_argument("--ground-truth", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("output_v2"))
    args = ap.parse_args()

    out = args.output_dir
    (out / "model").mkdir(parents=True, exist_ok=True)

    s1 = prepare_dataframe_v2(load_source(args.s1))
    s2 = prepare_dataframe_v2(load_source(args.s2))
    s3 = prepare_dataframe_v2(load_source(args.s3))
    gt = load_ground_truth(args.ground_truth)

    all_pairs = []
    all_features = []
    all_reasons = []
    summaries = []
    recalls = {}

    gt_map = {str(r.source1_entity_id): set(str(r.matched_entity_ids).split(",")) if str(r.matched_entity_ids).strip() else set() for r in gt.itertuples(index=False)}

    for source_name, source in (("s2", s2), ("s3", s3)):
        candidates, reasons = generate_candidates(s1, source)
        pairs, features = build_feature_pairs(candidates, s1, source)
        pairs["source"] = source_name
        all_pairs.append(pairs)
        all_features.append(features)
        all_reasons.append(reasons.assign(source=source_name))
        summaries.append({"source": source_name, **candidate_summary(candidates, len(s1))})

        covered = 0
        total = 0
        candidate_set = set(zip(candidates.source1_entity_id, candidates.source_entity_id))
        for s1_id, ids in gt_map.items():
            for source_id in ids:
                # GT does not expose the originating source; test membership in this source.
                if source_id in set(source.entity_id.astype(str)):
                    total += 1
                    covered += int((s1_id, source_id) in candidate_set)
        recalls[source_name] = {"covered_true_pairs": covered, "true_pairs": total, "candidate_recall": covered / total if total else 1.0}

    pairs = pd.concat(all_pairs, ignore_index=True)
    features = pd.concat(all_features, ignore_index=True)
    reasons = pd.concat(all_reasons, ignore_index=True)

    labeled = build_training_labels(pairs, gt)
    labeled = labeled.reset_index(drop=True)
    features = features.reset_index(drop=True)
    labeled_features = pd.concat([labeled, features], axis=1)

    train_pairs, valid_pairs = split_by_s1_entity(labeled_features, random_state=RANDOM_STATE)
    feature_cols = list(features.columns)
    train_pairs = select_hard_negatives(train_pairs, negative_ratio=5, random_state=RANDOM_STATE)

    X_train = train_pairs[feature_cols]
    y_train = train_pairs["label"]
    X_valid = valid_pairs[feature_cols]
    y_valid = valid_pairs["label"]

    if y_train.nunique() < 2:
        raise RuntimeError("Training candidates contain only one class; real candidate coverage must be inspected before training.")

    model = train_model(
    X_train,
    y_train,
    random_state=RANDOM_STATE,
    model_type="xgboost",
)
    valid_prob = predict_probabilities(model, X_valid)
    sweep = threshold_sweep(y_valid, valid_prob)
    threshold = select_threshold(sweep)
    valid_pred = (valid_prob >= threshold).astype(int)

    predictions = {}
    for (s1_id, source_id), pred in zip(valid_pairs[["source1_entity_id", "source_entity_id"]].itertuples(index=False, name=None), valid_pred):
        if pred:
            predictions.setdefault(str(s1_id), set()).add(str(source_id))
    macro = macro_f05_by_entity(gt_map, predictions)

    counts = pd.Series(list(predictions.values())).map(len) if predictions else pd.Series(dtype=int)
    metrics = {
        "macro_f0_5": macro,
        "validation_pair_precision": float((valid_pred & y_valid.to_numpy()).sum() / valid_pred.sum()) if valid_pred.sum() else 0.0,
        "validation_pair_recall": float((valid_pred & y_valid.to_numpy()).sum() / y_valid.sum()) if y_valid.sum() else 0.0,
        "selected_threshold": threshold,
        "predicted_links": int(valid_pred.sum()),
        "singleton_rate": float((counts == 1).mean()) if len(counts) else 0.0,
        "avg_links_per_s1_with_prediction": float(counts.mean()) if len(counts) else 0.0,
        "avg_candidates_per_s1": float(pairs.groupby("source1_entity_id").size().mean()) if not pairs.empty else 0.0,
        "max_candidates_per_s1": int(pairs.groupby("source1_entity_id").size().max()) if not pairs.empty else 0,
        "candidate_recall": recalls,
    }

    pairs.to_csv(out / "candidate_pairs.tsv", sep="\t", index=False)
    reasons.to_csv(out / "candidate_block_reasons.tsv", sep="\t", index=False)
    pd.DataFrame(summaries).to_csv(out / "blocking_metrics.csv", index=False)
    sweep.to_csv(out / "threshold_results.csv", index=False)
    (out / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
    save_model(model, out / "model" / "model.joblib")
    (out / "model" / "threshold.txt").write_text(str(threshold))

    coef = model.named_steps["classifier"].coef_[0]
    pd.DataFrame({"feature": feature_cols, "importance": abs(coef), "coefficient": coef}).sort_values("importance", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
