import argparse, sys
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("file")
p.add_argument("--sample", type=int, default=None)
args = p.parse_args()

kwargs = {"sep":"\t", "dtype":"string", "usecols":["entity_id","business_name","business_address","country"]}
if args.sample:
    kwargs["nrows"] = args.sample

df = pd.read_csv(args.file, **kwargs).fillna("")
print("Rows:", len(df))
print("Columns:", list(df.columns))
print("\nMissing:")
print(df.isna().sum())
print("\nUnique:")
for c in df.columns:
    print(f"{c}: {df[c].nunique():,}")
print("\nCountry:")
print(df["country"].value_counts(dropna=False))
for c in ["business_name","business_address"]:
    s=df[c].astype(str)
    print(f"\n{c}")
    print("mean length:", round(s.str.len().mean(),2))
    print("median length:", s.str.len().median())
    print("duplicates:", int(s.duplicated().sum()))
print("\nSample:")
print(df.head(10).to_string(index=False))
