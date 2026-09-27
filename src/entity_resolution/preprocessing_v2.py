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

def strip_legal_suffixes(text: str) -> str:
    return " ".join(t for t in text.split() if t not in LEGAL_SUFFIXES)

def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)

def sorted_tokens(text: str) -> str:
    return " ".join(sorted(set(text.split())))

def alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text)

def numeric_tokens(text: str):
    return re.findall(r"\d+", text)

def extract_postal(value: str) -> str:
    nums = numeric_tokens(value)
    for token in nums:
        if 3 <= len(token) <= 10:
            return token
    return ""

def extract_house_number(value: str) -> str:
    match = re.match(r"\s*(\d+[a-z]?)\b", value, flags=re.I)
    return match.group(1).lower() if match else ""

def prepare_dataframe_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().fillna("")
    name_raw = out["business_name"].astype(str)
    addr_raw = out["business_address"].astype(str)

    out["name_norm"] = name_raw.map(normalize_text)
    out["name_alnum"] = out["name_norm"].map(alnum)
    out["name_tokens"] = out["name_norm"].map(lambda x: " ".join(x.split()))
    out["name_sorted_tokens"] = out["name_norm"].map(sorted_tokens)
    out["name_compact"] = out["name_norm"].map(compact)

    out["name_no_suffix"] = out["name_norm"].map(strip_legal_suffixes)
    out["name_no_suffix_alnum"] = out["name_no_suffix"].map(alnum)
    out["name_no_suffix_compact"] = out["name_no_suffix"].map(compact)

    out["address_norm"] = addr_raw.map(normalize_text)
    out["address_alnum"] = out["address_norm"].map(alnum)
    out["address_tokens"] = out["address_norm"].map(lambda x: " ".join(x.split()))
    out["address_sorted_tokens"] = out["address_norm"].map(sorted_tokens)
    out["address_numbers"] = out["address_norm"].map(lambda x: " ".join(numeric_tokens(x)))
    out["house_number"] = out["address_norm"].map(extract_house_number)
    out["postal_code"] = out["address_norm"].map(extract_postal)

    out["country_norm"] = out["country"].astype(str).map(normalize_text)
    return out

def prepare_source_v2(path, nrows=None) -> pd.DataFrame:
    from .data_loader import load_source
    return prepare_dataframe_v2(load_source(path, nrows=nrows))
