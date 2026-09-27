import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entity_resolution.pipeline import run_test

p = argparse.ArgumentParser()
p.add_argument("--s1-limit", type=int, default=None)
p.add_argument("--s2-limit", type=int, default=None)
p.add_argument("--s3-limit", type=int, default=None)
args = p.parse_args()
run_test(args.s1_limit, args.s2_limit, args.s3_limit)
