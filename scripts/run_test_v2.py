"""Run V2 test inference and write the competition submission."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from entity_resolution.data_loader import load_source
from entity_resolution.preprocessing_v2 import prepare_dataframe_v2
from entity_resolution.candidate_generation_v2 import generate_candidates
from entity_resolution.features_v2 import pair_features_v2
from entity_resolution.model_v2 import load_model, predict_probabilities
from entity_resolution.evaluation_v2 import apply_threshold


def build_feature_pairs(candidates, s1, source):
    left = {str(r.entity_id): r for r in s1.itertuples(index=False)}
    right = {str(r.entity_id): r for r in source.itertuples(index=False)}
    valid = []
    features = []
    for row in candidates.itertuples(index=False):
        a, b = left.get(str(row.source1_entity_id)), right.get(str(row.source_entity_id))
        if a is not None and b is not None:
            valid.append((str(row.source1_entity_id), str(row.source_entity_id)))
            features.append(pair_features_v2(a._asdict(), b._asdict()))
    return pd.DataFrame(valid, columns=["source1_entity_id", "source_entity_id"]), pd.DataFrame(features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", type=Path, required=True)
    ap.add_argument("--s2", type=Path, required=True)
    ap.add_argument("--s3", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("output_v2"))
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    s1 = prepare_dataframe_v2(load_source(args.s1))
    sources = [("s2", prepare_dataframe_v2(load_source(args.s2))), ("s3", prepare_dataframe_v2(load_source(args.s3)))]
    model = load_model(args.model)

    all_pairs, all_features = [], []
    for source_name, source in sources:
        candidates, reasons = generate_candidates(s1, source)
        pairs, features = build_feature_pairs(candidates, s1, source)
        pairs["source"] = source_name
        all_pairs.append(pairs)
        all_features.append(features)

    pairs = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame(columns=["source1_entity_id", "source_entity_id", "source"])
    features = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    probabilities = predict_probabilities(model, features)
    matches = apply_threshold(probabilities, args.threshold)
    pairs["probability"] = probabilities
    pairs["is_match"] = matches

    rows = []
    for s1_id in s1.entity_id.astype(str):
        group = pairs[(pairs.source1_entity_id == s1_id) & (pairs.is_match == 1)]
        ids = group.source_entity_id.astype(str).drop_duplicates().tolist()
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ",".join(ids)})

    pairs.drop(columns=["probability", "is_match"]).to_csv(out / "candidate_pairs.tsv", sep="\t", index=False)
    pairs.to_csv(out / "test_pair_scores.tsv", sep="\t", index=False)
    pd.DataFrame(rows).to_csv(out / "matching_results.tsv", sep="\t", index=False)
    print(f"Wrote {len(rows)} S1 rows to {out / 'matching_results.tsv'}")


if __name__ == "__main__":
    main()
