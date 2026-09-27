from pathlib import Path
import pandas as pd

CORE_COLS = ["entity_id", "business_name", "business_address", "country"]


def load_source(path: Path, nrows=None) -> pd.DataFrame:
    kwargs = {
        "sep": "\t",
        "usecols": CORE_COLS,
        "dtype": "string",
    }
    if nrows is not None:
        kwargs["nrows"] = nrows
    return pd.read_csv(path, **kwargs).fillna("")


def load_source_chunks(path: Path, chunksize=25000):
    """Yield source records in bounded-memory chunks."""
    reader = pd.read_csv(
        path,
        sep="\t",
        usecols=CORE_COLS,
        dtype="string",
        chunksize=chunksize,
    )

    for chunk in reader:
        yield chunk.fillna("")


def load_ground_truth(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype="string").fillna("")