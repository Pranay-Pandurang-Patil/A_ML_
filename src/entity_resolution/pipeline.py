import time
from pathlib import Path
import pandas as pd

from .config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, GROUND_TRUTH,
    TEST_S1, TEST_S2, TEST_S3,
    OUTPUT_DIR, MATCHING_OUTPUT, CANDIDATE_OUTPUT,
)
from .data_loader import load_ground_truth
from .preprocessing import prepare_source
from .blocking import make_indexes
from .evaluation import parse_ground_truth, macro_f05
from .inference import predict_for_s1


def build_maps(df):
    return df.set_index("entity_id").to_dict("index")


def validate_outputs(s1, s2, s3, matching_df, candidate_df):
    problems = []
    expected = set(s1["entity_id"])
    predicted = set(matching_df["source1_entity_id"])
    if expected != predicted:
        problems.append(f"S1 coverage mismatch: expected {len(expected)}, got {len(predicted)}")
    if matching_df["source1_entity_id"].duplicated().any():
        problems.append("Duplicate source1_entity_id in matching output")

    valid_ids = set(s2["entity_id"]) | set(s3["entity_id"])
    candidate_map = {
        r.source1_entity_id: set(r.candidate_entity_ids.split(",")) if r.candidate_entity_ids else set()
        for r in candidate_df.itertuples(index=False)
    }

    for r in matching_df.itertuples(index=False):
        mids = set(r.matched_entity_ids.split(",")) if r.matched_entity_ids else set()
        if len(mids) != len(r.matched_entity_ids.split(",")) if r.matched_entity_ids else False:
            problems.append(f"Duplicate IDs for {r.source1_entity_id}")
        bad = mids - valid_ids
        if bad:
            problems.append(f"Invalid IDs for {r.source1_entity_id}: {sorted(bad)[:3]}")
        if not mids.issubset(candidate_map.get(r.source1_entity_id, set())):
            problems.append(f"Match outside candidate set: {r.source1_entity_id}")

    return problems


def run_validation(s1_limit=2000):
    print(f"Loading validation S1: first {s1_limit:,} rows")
    s1 = prepare_source(TRAIN_S1, nrows=s1_limit)
    s2 = prepare_source(TRAIN_S2)
    s3 = prepare_source(TRAIN_S3)
    gt = parse_ground_truth(load_ground_truth(GROUND_TRUTH))

    idx2, idx3 = make_indexes(s2), make_indexes(s3)
    by2, by3 = build_maps(s2), build_maps(s3)

    predictions = {}
    candidates = {}
    start = time.time()
    for row in s1.itertuples(index=False):
        pred, cand = predict_for_s1(row, idx2, idx3, by2, by3)
        predictions[row.entity_id] = pred
        candidates[row.entity_id] = cand

    eval_ids = [eid for eid in s1.entity_id if eid in gt]
    y_true = [gt[eid] for eid in eval_ids]
    y_pred = [predictions.get(eid, []) for eid in eval_ids]
    score = macro_f05(y_true, y_pred)

    print(f"Validation entities: {len(eval_ids):,}")
    print(f"Elapsed: {time.time()-start:.2f}s")
    print(f"Macro F0.5: {score:.6f}")
    print(f"Predicted links: {sum(map(len, y_pred)):,}")
    print(f"Predicted singletons: {sum(len(x)==0 for x in y_pred):,}")
    print(f"Average candidates/S1: {sum(map(len,candidates.values()))/len(candidates):.4f}")
    return score


def run_test(s1_limit=None, s2_limit=None, s3_limit=None):
    print("Loading test data...")
    s1 = prepare_source(TEST_S1, nrows=s1_limit)
    s2 = prepare_source(TEST_S2, nrows=s2_limit)
    s3 = prepare_source(TEST_S3, nrows=s3_limit)

    print(f"S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}")
    idx2, idx3 = make_indexes(s2), make_indexes(s3)
    by2, by3 = build_maps(s2), build_maps(s3)

    candidate_rows, matching_rows = [], []
    start = time.time()
    for i, row in enumerate(s1.itertuples(index=False), 1):
        matches, cands = predict_for_s1(row, idx2, idx3, by2, by3)
        candidate_rows.append({
            "source1_entity_id": row.entity_id,
            "candidate_entity_ids": ",".join(cands),
        })
        matching_rows.append({
            "source1_entity_id": row.entity_id,
            "matched_entity_ids": ",".join(matches),
        })
        if i % 10000 == 0:
            print(f"Processed {i:,}/{len(s1):,}")

    candidate_df = pd.DataFrame(candidate_rows)
    matching_df = pd.DataFrame(matching_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(CANDIDATE_OUTPUT, sep="\t", index=False)
    matching_df.to_csv(MATCHING_OUTPUT, sep="\t", index=False)

    problems = validate_outputs(s1, s2, s3, matching_df, candidate_df)
    print(f"Elapsed: {time.time()-start:.2f}s")
    print(f"Candidate output: {CANDIDATE_OUTPUT}")
    print(f"Matching output: {MATCHING_OUTPUT}")
    if problems:
        print("Validation problems:")
        for p in problems[:20]:
            print(" -", p)
    else:
        print("Local structural validation: PASS")

    return matching_df, candidate_df
