from rapidfuzz import fuzz


def jaccard_tokens(a, b):
    A, B = set(a.split()), set(b.split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def pair_features(a, b):
    name_a = a["compact_name"]
    name_b = b["compact_name"]
    addr_a = a["compact_address"]
    addr_b = b["compact_address"]

    name_exact = int(bool(name_a) and name_a == name_b)
    addr_exact = int(bool(addr_a) and addr_a == addr_b)

    return {
        "name_exact": name_exact,
        "addr_exact": addr_exact,
        "name_token": jaccard_tokens(a["norm_name"], b["norm_name"]),
        "addr_token": jaccard_tokens(a["norm_address"], b["norm_address"]),
        "name_fuzzy": fuzz.token_set_ratio(a["norm_name"], b["norm_name"]) / 100.0,
        "addr_fuzzy": fuzz.token_set_ratio(a["norm_address"], b["norm_address"]) / 100.0,
        "country_exact": int(a["country"] == b["country"]),
    }
