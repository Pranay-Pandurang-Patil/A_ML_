import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entity_resolution.pipeline import run_validation

p = argparse.ArgumentParser()
p.add_argument("--s1-limit", type=int, default=2000)
args = p.parse_args()
run_validation(args.s1_limit)
