from collections import defaultdict

BLOCK_COLUMNS = {
    "name": "country_name_key",
    "addr": "country_addr_key",
    "name_addr": "country_name_addr_key",
}


def build_index(df, column):
    index = defaultdict(list)
    for row in df.itertuples(index=False):
        key = getattr(row, column)
        if key:
            index[key].append(row.entity_id)
    return index


def make_indexes(df):
    return {
        name: build_index(df, column)
        for name, column in BLOCK_COLUMNS.items()
    }


def candidates_for_row(row, indexes):
    candidates = set()
    for name, column in BLOCK_COLUMNS.items():
        key = getattr(row, column)
        if key:
            candidates.update(indexes[name].get(key, []))
    return candidates
