# Business Entity Resolution Challenge

## Methodology Documentation

**Amazon ML Challenge 2026**


## 1. Problem Understanding

The task is a multi-source business entity resolution problem. Source 1 is the deduplicated reference source, while Sources 2 and 3 contain noisy business records. For every Source 1 entity, the system must identify all corresponding records from Source 2 and/or Source 3, including the possibility of zero, one, or multiple matches.

The challenge data contains business names, addresses, and country labels. The data can contain abbreviations, legal-suffix differences, typos, transliterations, punctuation changes, word-order changes, incomplete addresses, landmark-based addresses, and other format variations. The test data also contains France, so country is treated as an open-set string field rather than being restricted to the countries present in training.

## 2. Overall Methodology

The implemented solution follows a precision-oriented, candidate-based entity-resolution pipeline. Instead of comparing every Source 2/Source 3 record against every Source 1 record, the pipeline first generates a manageable set of plausible candidates using multiple blocking rules. Pairwise features are then calculated for the surviving candidate pairs and passed to a supervised machine-learning model. A probability threshold is finally applied to determine accepted matches.

### Pipeline Steps

1. Input and normalize business records.
2. Build searchable indexes over the Source 1 reference entities.
3. Generate candidates using multiple complementary blocking strategies.
4. Compute name, address, country, numeric and token-level pair features.
5. Score candidate pairs with the trained XGBoost model.
6. Apply the selected probability threshold.
7. Aggregate accepted Source 2/Source 3 IDs by Source 1 entity.
8. Write the final matching results and candidate/audit artifacts.

## 3. Data Preprocessing and Normalization

Normalization is designed to reduce superficial differences while retaining information that can distinguish businesses. Separate normalized representations are maintained rather than relying on a single transformed string.

- Case and whitespace normalization.
- Punctuation and non-alphanumeric normalization.
- Normalized/alphanumeric business-name representations.
- Tokenization and token-set/sorted-token representations.
- Handling of legal-name and suffix variation through suffix-aware representations.
- Address normalization and compact/alphanumeric address representations.
- Extraction and comparison of house/building numbers.
- Extraction and comparison of postal-code information.
- Extraction of numeric address tokens.
- Country normalization while preserving arbitrary country labels.

The preprocessing stage does not depend on external business databases, geocoding APIs, or internet-based identity lookup. The approach uses only the supplied challenge data, consistent with the challenge's fair-play requirements.

## 4. Candidate Generation / Blocking Strategy

Candidate generation is the recall-oriented stage of the pipeline. Its purpose is to reduce the search space while retaining plausible true matches. Multiple blocking rules are used because no single key is reliable under the challenge's noise patterns.

The implemented V2 blocking/candidate-generation pipeline uses complementary signals such as:

- Exact normalized business-name matches.
- Suffix-aware / suffix-stripped business-name matches.
- Exact normalized address matches.
- Name-token and address-number combinations.
- Postal-code and name-token combinations when available.
- Rare or informative name-token based retrieval.
- Character n-gram based candidate retrieval.
- Address-number based retrieval.

Candidates from the applicable blocking strategies are combined so that a pair found by any valid blocking rule can proceed to feature generation and model scoring. This is important because an entity may have a damaged name but a useful address, or a damaged address but a useful name.

The final candidate set used for inference is intended to represent the pairs actually fed to the matching model. This is consistent with the challenge definition of `candidate_pairs.tsv` as the last candidate-generation stage before model scoring.

## 5. Pairwise Feature Engineering

For every Source 1–Source 2/Source 3 candidate pair, the pipeline constructs numerical features describing name, address, country, and structural agreement.

### 5.1 Business Name Features

- Exact equality of normalized names.
- Exact equality of alphanumeric/compact representations.
- Suffix-aware and suffix-stripped equality.
- Token overlap and token-intersection measures.
- Jaccard-style token similarity.
- RapidFuzz ratio.
- RapidFuzz partial-ratio similarity.
- Token-sort similarity.
- Token-set similarity.
- Similarity between suffix-stripped names.
- Name-length difference and related structural indicators.

### 5.2 Address Features

- Exact equality of normalized addresses.
- Alphanumeric/compact address equality.
- Address token overlap.
- Address token-intersection and Jaccard-style measures.
- Fuzzy address similarity measures.
- House/building-number agreement.
- Postal-code agreement.
- Numeric-address-token similarity.

### 5.3 Other Features

- Country equality.
- Missing-value indicators.
- Structural numeric/address agreement.

The feature design intentionally combines exact, token-based, fuzzy, and structural signals. Exact features are useful for clean records, while fuzzy and token-level features provide robustness to the noisy variations described in the challenge.

## 6. Model Architecture

The V2 matching stage uses a supervised pair-classification approach. Each candidate pair is represented by the engineered feature vector and classified as a likely match or non-match.

### 6.1 Primary Model

The primary model is **XGBoost**. It is used because the feature space combines heterogeneous numeric similarity signals, exact-match indicators, missingness indicators, and interaction effects between name and address evidence. A tree-based gradient-boosting model can learn non-linear combinations of these signals without requiring a single manually defined similarity formula.

### 6.2 Model Artifact

The trained model is serialized as a **Joblib** artifact and loaded during test inference. The inference script accepts the model path and the decision threshold as command-line arguments, allowing the same pipeline to be reused without changing source code.

## 7. Training Strategy

Training pairs are derived from the labelled training data and the candidate-generation process. Positive examples correspond to known Source 1-to-Source 2/Source 3 matches, while hard negatives are generated from plausible candidate pairs that are not labelled as true matches. Hard negatives are important because random negatives are generally too easy for an entity-resolution classifier.

The V2 training/validation workflow supports chunked processing and a configurable maximum number of training rows so that experiments can be performed under constrained memory. A fixed random state of **42** is used for reproducibility in the implemented workflow.

## 8. Threshold Selection and Evaluation

The challenge evaluates predictions using **macro F0.5**. F0.5 is precision-heavy, meaning false merges are penalized more strongly than missed links. The implemented validation workflow therefore treats the probability threshold as an important model-selection parameter rather than simply using the default classifier threshold.

The F0.5 score is computed per Source 1 entity and macro-averaged. Singleton entities with no true matches are included in the metric, so the system must be able to leave an entity unmatched rather than forcing a prediction.

The final inference threshold used in the documented test command is **0.95**. This reflects the precision-oriented design of the solution and is configurable through the command-line interface.

## 9. Inference and Output Generation

Inference is performed in chunks to control memory consumption. Source 1 is loaded and indexed, while Source 2 and Source 3 are processed chunk by chunk. Candidate pairs are generated, scored, filtered by the configured threshold, and aggregated into one result row per Source 1 entity.

The final `matching_results.tsv` contains `source1_entity_id` and a comma-separated list of accepted Source 2 and/or Source 3 entity IDs. Empty lists are retained for Source 1 entities for which no match is accepted.

## 10. Candidate and Audit Outputs

In addition to the final matching results, the pipeline produces candidate-pair and pair-score artifacts during the test workflow. These artifacts make it possible to inspect which candidate pairs were considered and how the model scored them.

| Output File | Purpose |
|---|---|
| `matching_results.tsv` | Final entity-resolution predictions |
| `candidate_pairs.tsv` | Candidate pairs generated for model inference |
| `test_pair_scores.tsv` | Candidate-pair probabilities and match decisions |

## 11. Scalability and Resource Considerations

A major implementation consideration was memory usage. Full pairwise comparison across all sources would be prohibitively expensive, so the solution uses blocking before feature calculation and processes test records in configurable chunks. The chunk size can be changed from the command line to adapt the pipeline to available RAM.

The implementation also separates preprocessing, blocking, feature engineering, model training, evaluation, and inference into independent modules. This makes individual stages easier to test and allows the inference pipeline to reuse the trained model artifact.

## 12. Reproducibility

The repository contains the V2 source modules, training/validation scripts, test inference script, requirements file, and README run instructions. The pipeline is intended to be reproduced using the supplied challenge data and the declared Python dependencies.

The test inference command uses `run_test_v2_chunked.py` with the trained model artifact, a configurable threshold, and chunked processing. No external identity-resolution service, commercial entity-resolution API, geocoder, or external business database is required.

## 13. Submission and Compliance Considerations

The challenge requires every Source 1 test entity to appear exactly once in `matching_results.tsv`, with matched IDs restricted to existing Source 2/Source 3 test entities and no duplicate IDs within an ID list. The challenge also requires the final submission package to contain the final output files, runnable code, requirements, and the methodology document.

The methodology described here uses only the provided challenge data and local model processing. No external data augmentation or business-identity lookup is part of the approach.

## 13A. Project Architecture

The solution follows a modular, two-stage entity-resolution architecture: **candidate generation followed by pairwise matching and decision-making**.

### End-to-End Architecture

```text
Input Sources
  Source 1 | Source 2 | Source 3
            |
            v
   Data Preprocessing
            |
            v
 Candidate Generation
      / Blocking
            |
            v
    Candidate Pairs
            |
            v
   Feature Engineering
            |
            v
      XGBoost Model
            |
            v
   Decision / Threshold
            |
            v
    Final Output Files
```

### 1. Input Layer

The system accepts three independent tab-separated datasets: Source 1, Source 2, and Source 3. Source 1 is the deduplicated reference source, while Source 2 and Source 3 contain records that may correspond to Source 1 entities. Each record contains an entity identifier, business name, business address, and country.

### 2. Preprocessing Layer

The preprocessing layer transforms raw text fields into normalized representations suitable for entity matching. It reduces superficial differences in capitalization, punctuation, formatting, and other textual variations while retaining information useful for matching.

Implemented in:

```text
src/entity_resolution/preprocessing_v2.py
```

### 3. Candidate Generation / Blocking Layer

Instead of comparing every Source 1 record against every Source 2/3 record, the system first generates a smaller candidate set using multiple blocking and retrieval strategies. This reduces computational cost while maintaining a recall-oriented candidate pool.

Relevant components:

```text
src/entity_resolution/blocking_v2.py
src/entity_resolution/candidate_generation_v2.py
```

### 4. Feature Engineering Layer

Each generated candidate pair is converted into a numerical feature vector. Features represent similarities and useful signals from business names, addresses, numeric/address information, country information, and normalized text representations.

Implemented in:

```text
src/entity_resolution/features_v2.py
```

### 5. Machine Learning Layer

The engineered candidate-pair features are supplied to an XGBoost classifier, which produces a match score for each candidate pair. Training and model management are handled by:

```text
src/entity_resolution/training_v2.py
src/entity_resolution/model_v2.py
```

The trained model is persisted as a Joblib artifact.

### 6. Decision Layer

The model score is converted into a final matching decision using the configured inference threshold. The current test command uses a threshold of **0.95**. The decision stage also handles final match selection, duplicate prevention, and empty-match cases.

### 7. Output Layer

The inference pipeline produces:

- `matching_results.tsv` — final prediction file.
- `candidate_pairs.tsv` — final candidate set supplied to the matching stage.
- `test_pair_scores.tsv` — pair-level scoring/audit artifact.

### Repository-Level Architecture

```text
business_entity_resolution/
|
+-- data/
|   +-- train/
|   +-- test/
|
+-- src/
|   +-- entity_resolution/
|       +-- preprocessing_v2.py
|       +-- blocking_v2.py
|       +-- candidate_generation_v2.py
|       +-- features_v2.py
|       +-- training_v2.py
|       +-- model_v2.py
|       +-- evaluation_v2.py
|       +-- artifacts_v2.py
|
+-- scripts/
|   +-- train_model_v2.py
|   +-- run_validation_v2.py
|   +-- run_validation_v2_chunked.py
|   +-- run_test_v2.py
|   +-- run_test_v2_chunked.py
|
+-- output_v2/
|   +-- model/
|       +-- model.joblib
|
+-- output_v2_test/
|   +-- matching_results.tsv
|   +-- candidate_pairs.tsv
|   +-- test_pair_scores.tsv
|
+-- requirements.txt
+-- README.md
```

## End-to-End Training and Inference Flow

### Training

```text
Source 1 + Source 2 + Source 3
              |
              v
       Preprocessing
              |
              v
   Candidate Generation
              |
              v
    Feature Engineering
              |
              v
          XGBoost
              |
              v
       Trained Model
              |
              v
         model.joblib
```

### Inference

```text
Test Source Data
       |
       v
Preprocessing
       |
       v
Candidate Generation
       |
       v
Feature Engineering
       |
       v
XGBoost Scoring
       |
       v
Threshold / Decisions
      / \
     v   v
candidate_pairs   matching_results
```

## 14. Summary of the Approach

The complete approach can be summarized as:

1. Normalize noisy business records.
2. Generate high-recall candidates with multiple blocking rules.
3. Compute complementary exact, token, fuzzy, and structural pair features.
4. Classify candidate pairs using XGBoost.
5. Apply a precision-oriented probability threshold.
6. Aggregate accepted links by Source 1 entity.
7. Produce the required submission and audit artifacts.
