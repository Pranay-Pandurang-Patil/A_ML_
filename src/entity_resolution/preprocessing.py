import re
import unicodedata
import pandas as pd

LEGAL_SUFFIXES = {
    "incorporated", "inc", "corporation", "corp", "limited", "ltd",
    "llc", "llp", "private", "pvt", "company", "co",
    "limitedliabilitycompany",
}


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value).lower())
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(value) -> str:
    text = normalize_text(value)
    return " ".join(t for t in text.split() if t not in LEGAL_SUFFIXES)


def normalize_address(value) -> str:
    return normalize_text(value)


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def token_signature(text: str) -> str:
    return " ".join(sorted(set(text.split())))


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().fillna("")
    out["norm_name"] = out["business_name"].map(normalize_name)
    out["norm_address"] = out["business_address"].map(normalize_address)
    out["compact_name"] = out["norm_name"].map(compact)
    out["compact_address"] = out["norm_address"].map(compact)
    out["name_signature"] = out["norm_name"].map(token_signature)
    out["address_signature"] = out["norm_address"].map(token_signature)
    return out


def add_block_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    country = out["country"].astype(str)
    out["country_name_key"] = country + "|" + out["compact_name"]
    out["country_addr_key"] = country + "|" + out["compact_address"]
    out["country_name_addr_key"] = (
        country + "|" + out["compact_name"] + "|" + out["compact_address"]
    )
    return out


def prepare_source(path, nrows=None) -> pd.DataFrame:
    from .data_loader import load_source
    return add_block_keys(prepare_dataframe(load_source(path, nrows=nrows)))
