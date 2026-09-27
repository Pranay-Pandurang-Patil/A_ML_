from rapidfuzz import fuzz


def _tokens(value):
    return set(str(value).split()) if value else set()


def _jaccard(a, b):
    a = _tokens(a)
    b = _tokens(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _overlap(a, b):
    a = _tokens(a)
    b = _tokens(b)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _intersection_count(a, b):
    return float(len(_tokens(a) & _tokens(b)))


def _safe_ratio(a, b, scorer):
    if not a or not b:
        return 0.0
    return scorer(a, b) / 100.0


def pair_features_v2(a, b):
    """Build ID-independent pair features for two prepared V2 records."""
    name_norm_a, name_norm_b = a["name_norm"], b["name_norm"]
    name_no_suffix_a, name_no_suffix_b = a["name_no_suffix"], b["name_no_suffix"]
    address_a, address_b = a["address_norm"], b["address_norm"]

    return {
        # Name
        "name_exact": int(bool(name_norm_a) and name_norm_a == name_norm_b),
        "name_no_suffix_exact": int(bool(name_no_suffix_a) and name_no_suffix_a == name_no_suffix_b),
        "name_alnum_exact": int(a["name_alnum"] == b["name_alnum"] and bool(a["name_alnum"])),
        "name_jaccard": _jaccard(name_norm_a, name_norm_b),
        "name_overlap": _overlap(name_norm_a, name_norm_b),
        "name_token_intersection": _intersection_count(name_norm_a, name_norm_b),
        "name_fuzz_ratio": _safe_ratio(name_norm_a, name_norm_b, fuzz.ratio),
        "name_fuzz_partial": _safe_ratio(name_norm_a, name_norm_b, fuzz.partial_ratio),
        "name_fuzz_token_sort": _safe_ratio(name_norm_a, name_norm_b, fuzz.token_sort_ratio),
        "name_fuzz_token_set": _safe_ratio(name_norm_a, name_norm_b, fuzz.token_set_ratio),
        "name_no_suffix_fuzz": _safe_ratio(name_no_suffix_a, name_no_suffix_b, fuzz.token_set_ratio),
        "name_length_diff": float(abs(len(a["name_alnum"]) - len(b["name_alnum"]))),

        # Address
        "address_exact": int(bool(address_a) and address_a == address_b),
        "address_alnum_exact": int(a["address_alnum"] == b["address_alnum"] and bool(a["address_alnum"])),
        "address_jaccard": _jaccard(address_a, address_b),
        "address_overlap": _overlap(address_a, address_b),
        "address_token_intersection": _intersection_count(address_a, address_b),
        "address_fuzz_ratio": _safe_ratio(address_a, address_b, fuzz.ratio),
        "address_fuzz_partial": _safe_ratio(address_a, address_b, fuzz.partial_ratio),
        "address_fuzz_token_sort": _safe_ratio(address_a, address_b, fuzz.token_sort_ratio),
        "address_fuzz_token_set": _safe_ratio(address_a, address_b, fuzz.token_set_ratio),

        # Structured address fields
        "house_number_exact": int(bool(a["house_number"]) and a["house_number"] == b["house_number"]),
        "postal_exact": int(bool(a["postal_code"]) and a["postal_code"] == b["postal_code"]),
        "address_numbers_jaccard": _jaccard(a["address_numbers"], b["address_numbers"]),

        # Country / missingness
        "country_exact": int(bool(a["country_norm"]) and a["country_norm"] == b["country_norm"]),
        "name_missing_either": int(not name_norm_a or not name_norm_b),
        "address_missing_either": int(not address_a or not address_b),
    }
