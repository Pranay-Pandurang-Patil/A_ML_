def is_match(features, name_threshold=0.96, address_threshold=0.88):
    if features["name_exact"] and features["addr_exact"]:
        return True

    if features["name_exact"] and features["addr_fuzzy"] >= 0.90:
        return True

    if features["addr_exact"] and features["name_fuzzy"] >= 0.92:
        return True

    if (
        features["name_fuzzy"] >= name_threshold
        and features["addr_fuzzy"] >= address_threshold
    ):
        return True

    return False
