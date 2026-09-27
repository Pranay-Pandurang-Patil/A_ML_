"""V2 model training entry point.

This script exposes the memory-conscious V2 training pipeline
through the required train_model_v2.py CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Make the project source directory importable when this script
# is executed directly with: python scripts/train_model_v2.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from run_validation_v2_chunked import main


if __name__ == "__main__":
    main()