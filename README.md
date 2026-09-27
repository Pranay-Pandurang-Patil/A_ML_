# Amazon ML Challenge 2026 — Python Entity Resolution Baseline

This project converts the supplied reference notebook into a command-line Python pipeline. It keeps the notebook's deterministic baseline: text normalization, country-aware blocking, RapidFuzz pair scoring, conservative matching rules, F0.5 validation, and generation of `matching_results.tsv` and `candidate_pairs.tsv`.

## 1. Put the challenge data here

```text
data/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

## 2. Create the environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

## 3. Profile a large file

```bash
python scripts/profile_dataset.py data/train/train_source1.tsv
```

For a quick sample:

```bash
python scripts/profile_dataset.py data/train/train_source1.tsv --sample 100000
```

## 4. Run the baseline validation

```bash
python scripts/run_validation.py --s1-limit 2000
```

The validation uses the first N Source 1 records while loading the full S2/S3 reference pools, matching the supplied notebook's validation style.

## 5. Run a development test

Start small:

```bash
python scripts/run_test.py --s1-limit 25000 --s2-limit 100000 --s3-limit 100000
```

It should generate:

```text
output/candidate_pairs.tsv
output/matching_results.tsv
```

## 6. Run the full test

Only after the development run works:

```bash
python scripts/run_test.py
```

## Important

This is the deterministic baseline, not the final intended competition model. The next stage is to strengthen blocking and train a pairwise classifier on blocked candidate pairs. The challenge is precision-heavy, so every improvement must be measured using macro F0.5 rather than accuracy.
