from collections import defaultdict, Counter


BLOCKS = (
    "exact_name",
    "suffix_name",
    "exact_address",
    "address_number_name_token",
    "postal_name",
    "rare_name_token",
    "char_ngram",
    "address_number",
)


def _index_values(df, value_fn):
    index = defaultdict(list)

    for row in df.itertuples(index=False):
        values = value_fn(row)

        if isinstance(values, str):
            values = (values,) if values else ()

        for value in values:
            if value:
                index[value].append(row.entity_id)

    return index


def _char_ngrams(text, n=3):
    if not text or len(text) < n:
        return ()

    return {
        text[i:i + n]
        for i in range(len(text) - n + 1)
    }


def make_indexes_v2(
    df,
    rare_token_min_count=1,
    rare_ngram_max_count=50,
    address_number_max_count=200,
):
    token_counts = defaultdict(int)
    ngram_counts = Counter()
    address_number_counts = Counter()

    for row in df.itertuples(index=False):
        # Name-token frequency
        for token in set(row.name_no_suffix.split()):
            if token:
                token_counts[token] += 1

        # Character n-gram frequency
        for gram in _char_ngrams(row.name_no_suffix_alnum):
            ngram_counts[gram] += 1

        # Address-number frequency
        for number in set(row.address_numbers.split()):
            if number:
                address_number_counts[number] += 1

    indexes = {}

    # ---------------------------------------------------------
    # 1. Exact normalized name
    # ---------------------------------------------------------
    indexes["exact_name"] = _index_values(
        df,
        lambda r: (
            f"{r.country_norm}|{r.name_norm}"
            if r.name_norm else ""
        ),
    )

    # ---------------------------------------------------------
    # 2. Name with legal suffixes removed
    # ---------------------------------------------------------
    indexes["suffix_name"] = _index_values(
        df,
        lambda r: (
            f"{r.country_norm}|{r.name_no_suffix}"
            if r.name_no_suffix else ""
        ),
    )

    # ---------------------------------------------------------
    # 3. Exact normalized address
    # ---------------------------------------------------------
    indexes["exact_address"] = _index_values(
        df,
        lambda r: (
            f"{r.country_norm}|{r.address_norm}"
            if r.address_norm else ""
        ),
    )

    # ---------------------------------------------------------
    # 4. Address number + name token
    # ---------------------------------------------------------
    indexes["address_number_name_token"] = _index_values(
        df,
        lambda r: (
            f"{r.country_norm}|{r.house_number}|{token}"
            for token in set(r.name_no_suffix.split())
            if r.house_number and token
        ),
    )

    # ---------------------------------------------------------
    # 5. Postal code + name token
    # ---------------------------------------------------------
    indexes["postal_name"] = _index_values(
        df,
        lambda r: (
            f"{r.country_norm}|{r.postal_code}|{token}"
            for token in set(r.name_no_suffix.split())
            if r.postal_code and token
        ),
    )

    # ---------------------------------------------------------
    # 6. Rare name token
    # ---------------------------------------------------------
    indexes["rare_name_token"] = _index_values(
        df,
        lambda r: (
            f"{r.country_norm}|{token}"
            for token in set(r.name_no_suffix.split())
            if token
            and token_counts[token] <= rare_token_min_count
        ),
    )

    # ---------------------------------------------------------
    # 7. Rare character 3-gram
    # ---------------------------------------------------------
    indexes["char_ngram"] = _index_values(
        df,
        lambda r: (
            f"{r.country_norm}|{gram}"
            for gram in _char_ngrams(r.name_no_suffix_alnum)
            if ngram_counts[gram] <= rare_ngram_max_count
        ),
    )

    # ---------------------------------------------------------
    # 8. Selective address-number block
    #
    # Only index numbers occurring <= address_number_max_count
    # times in S1 to avoid candidate explosion.
    # ---------------------------------------------------------
    indexes["address_number"] = _index_values(
        df,
        lambda r: (
            f"{r.country_norm}|{number}"
            for number in set(r.address_numbers.split())
            if number
            and address_number_counts[number] <= address_number_max_count
        ),
    )

    return indexes


def candidates_for_row_v2(row, indexes):
    candidates = set()
    reasons = defaultdict(set)

    keys = {
        # -----------------------------------------------------
        # Exact normalized name
        # -----------------------------------------------------
        "exact_name": (
            f"{row.country_norm}|{row.name_norm}"
            if row.name_norm else "",
        ),

        # -----------------------------------------------------
        # Suffix-stripped name
        # -----------------------------------------------------
        "suffix_name": (
            f"{row.country_norm}|{row.name_no_suffix}"
            if row.name_no_suffix else "",
        ),

        # -----------------------------------------------------
        # Exact normalized address
        # -----------------------------------------------------
        "exact_address": (
            f"{row.country_norm}|{row.address_norm}"
            if row.address_norm else "",
        ),

        # -----------------------------------------------------
        # Address number + name token
        # -----------------------------------------------------
        "address_number_name_token": tuple(
            f"{row.country_norm}|{row.house_number}|{token}"
            for token in set(row.name_no_suffix.split())
            if row.house_number and token
        ),

        # -----------------------------------------------------
        # Postal + name token
        # -----------------------------------------------------
        "postal_name": tuple(
            f"{row.country_norm}|{row.postal_code}|{token}"
            for token in set(row.name_no_suffix.split())
            if row.postal_code and token
        ),

        # -----------------------------------------------------
        # Rare name token
        # -----------------------------------------------------
        "rare_name_token": tuple(
            f"{row.country_norm}|{token}"
            for token in set(row.name_no_suffix.split())
            if token
        ),

        # -----------------------------------------------------
        # Character 3-gram
        # -----------------------------------------------------
        "char_ngram": tuple(
            f"{row.country_norm}|{gram}"
            for gram in _char_ngrams(row.name_no_suffix_alnum)
        ),

        # -----------------------------------------------------
        # Address numbers
        # -----------------------------------------------------
        "address_number": tuple(
            f"{row.country_norm}|{number}"
            for number in set(row.address_numbers.split())
            if number
        ),
    }

    for block_name, block_keys in keys.items():
        index = indexes[block_name]

        for key in block_keys:
            if not key:
                continue

            for entity_id in index.get(key, []):
                candidates.add(entity_id)
                reasons[entity_id].add(block_name)

    return candidates, reasons