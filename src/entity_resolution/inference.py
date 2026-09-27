from .blocking import candidates_for_row
from .features import pair_features
from .matcher import is_match


def predict_for_s1(row, idx2, idx3, by_id2, by_id3, name_threshold=0.96, address_threshold=0.88):
    candidates = set()
    candidates.update(candidates_for_row(row, idx2))
    candidates.update(candidates_for_row(row, idx3))

    predictions = []
    for cid in candidates:
        candidate = by_id2.get(cid) if cid.startswith("S2-") else by_id3.get(cid)
        if candidate is None:
            continue
        if is_match(pair_features(row._asdict(), candidate), name_threshold, address_threshold):
            predictions.append(cid)
    return sorted(set(predictions)), sorted(candidates)
