from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = ROOT_DIR / "output"

TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"
TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

MATCHING_OUTPUT = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUTPUT = OUTPUT_DIR / "candidate_pairs.tsv"

CORE_COLS = ["entity_id", "business_name", "business_address", "country"]
