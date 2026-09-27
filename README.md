# Amazon ML Challenge 2026 – Business Entity Resolution

## Overview

This project implements an ML-based Business Entity Resolution system for matching noisy business records from Source 2 and Source 3 with reference entities from Source 1.

The system combines data normalization, multi-pass blocking, fuzzy matching, candidate generation, and XGBoost-based classification to identify potential entity matches efficiently.

---

## Key Features

- V2 Multi-Pass Blocking for efficient candidate generation
- Business name and address normalization
- Fuzzy name and address matching
- Hard-negative candidate generation
- XGBoost-based ML matcher
- F0.5-based threshold selection
- Support for multiple matches
- Support for no-match cases
- Candidate-pair audit output
- CPU-friendly chunked processing
- Designed to handle large datasets efficiently

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- RapidFuzz
- XGBoost
- Joblib

---

## Project Structure

```text
.
├── src/
│   └── entity_resolution/
│       ├── preprocessing_v2.py
│       ├── blocking_v2.py
│       ├── candidate_generation_v2.py
│       ├── features_v2.py
│       ├── training_v2.py
│       ├── model_v2.py
│       ├── evaluation_v2.py
│       └── artifacts_v2.py
│
├── scripts/
│   ├── train_model_v2.py
│   ├── run_validation_v2.py
│   ├── run_validation_v2_chunked.py
│   ├── run_test_v2.py
│   └── run_test_v2_chunked.py
│
├── data/
│   ├── train/
│   └── test/
│
├── output_v2/
├── output_v2_test/
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create or Activate the Virtual Environment

If a virtual environment already exists:

```cmd
.venv\Scripts\activate
```

### 2. Install Dependencies

```cmd
python -m pip install -r requirements.txt
```

### 3. Set the Python Path

For Windows CMD:

```cmd
set PYTHONPATH=%CD%\src
```

---

## Training and Validation

Run memory-conscious training and validation using:

```cmd
python scripts\train_model_v2.py ^
  --s1 data\train\train_source1.tsv ^
  --s2 data\train\train_source2.tsv ^
  --s3 data\train\train_source3.tsv ^
  --ground-truth data\train\train_ground_truth.tsv ^
  --output-dir output_v2 ^
  --chunk-size 5000 ^
  --max-train-rows 5000
```

### Parameters

| Parameter | Description |
|---|---|
| `--s1` | Source 1 reference dataset |
| `--s2` | Source 2 dataset |
| `--s3` | Source 3 dataset |
| `--ground-truth` | Ground-truth matching file |
| `--output-dir` | Directory for generated model and artifacts |
| `--chunk-size` | Number of rows processed per chunk |
| `--max-train-rows` | Maximum number of training rows |

---

## Full Test

Run the test pipeline on the complete test datasets:

```cmd
python scripts\run_test_v2_chunked.py ^
  --s1 data\test\test_source1.tsv ^
  --s2 data\test\test_source2.tsv ^
  --s3 data\test\test_source3.tsv ^
  --model output_v2\model\model.joblib ^
  --threshold 0.95 ^
  --output-dir output_v2_test ^
  --chunk-size 500
```

---

## Generated Test Files

After successful execution, the following files are generated:

```text
output_v2_test/
├── matching_results.tsv
├── candidate_pairs.tsv
└── test_pair_scores.tsv
```

### File Descriptions

| File | Description |
|---|---|
| `matching_results.tsv` | Final entity matching results |
| `candidate_pairs.tsv` | Candidate pairs generated during blocking |
| `test_pair_scores.tsv` | ML-generated matching scores for candidate pairs |

---

## Submission Format

The final submission file is:

```text
matching_results.tsv
```

It contains the following columns:

```text
source1_entity_id    matched_entity_ids
```

Each Source 1 entity appears exactly once.

The system supports:

- Single matches
- Multiple matches
- Empty matches / no-match cases

Example:

```text
source1_entity_id    matched_entity_ids
2206821              S1-773889195
2206822              S1-925783039,S1-884562311
2206823
```

---

## Matching Pipeline

The overall entity resolution pipeline follows these stages:

```text
Raw Business Records
        |
        v
Data Preprocessing
        |
        v
Name & Address Normalization
        |
        v
Multi-Pass Blocking
        |
        v
Candidate Generation
        |
        v
Fuzzy Matching & Feature Extraction
        |
        v
Hard-Negative Generation
        |
        v
XGBoost ML Matcher
        |
        v
Threshold Selection
        |
        v
Final Entity Matches
```

This approach avoids unrestricted Cartesian matching and reduces computational cost when working with large datasets.

---

## Evaluation

The primary evaluation metric used by the system is:

**Macro F0.5**

The matching threshold is selected based on F0.5 performance to balance precision and recall while giving greater importance to precision.

---

## Notes

- Random State: `42`
- Evaluation Metric: Macro F0.5
- ML Model: XGBoost
- Fuzzy Matching: RapidFuzz
- Candidate Generation: Multiple blocking strategies
- Processing: Chunked and memory-conscious
- Matching: Supports multiple matches and no-match cases
- Candidate generation uses blocking instead of unrestricted Cartesian matching

---

## Running the Complete Pipeline

A typical workflow is:

```text
1. Install dependencies
        |
        v
2. Activate virtual environment
        |
        v
3. Set PYTHONPATH
        |
        v
4. Train the XGBoost matcher
        |
        v
5. Validate the model
        |
        v
6. Select the matching threshold
        |
        v
7. Run the full test pipeline
        |
        v
8. Generate matching_results.tsv
        |
        v
9. Submit the final results
```

---

## Amazon ML Challenge 2026

**Project:** Business Entity Resolution

**Approach:** Multi-Pass Blocking + Fuzzy Matching + XGBoost

**Primary Evaluation Metric:** Macro F0.5
