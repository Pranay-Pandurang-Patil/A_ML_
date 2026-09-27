"""V2 output/config helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def ensure_output_layout(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    (output / "model").mkdir(parents=True, exist_ok=True)
    return output
