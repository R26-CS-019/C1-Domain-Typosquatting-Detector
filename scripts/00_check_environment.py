"""Check that Python, packages, paths, and the CSV are ready."""

from __future__ import annotations

import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DATASET_PATH  # noqa: E402


def main() -> int:
    print("IT22131010 environment check")
    print(f"Python:       {platform.python_version()}")
    print(f"Executable:   {sys.executable}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset:      {DEFAULT_DATASET_PATH}")

    if sys.version_info < (3, 12):
        print("ERROR: Python 3.12 or newer is required.")
        return 1

    if not DEFAULT_DATASET_PATH.is_file():
        print("ERROR: Dataset CSV was not found at the expected path.")
        print("Use --dataset with the audit script if your location is different.")
        return 1

    import pandas  # noqa: F401
    import sklearn  # noqa: F401
    import tldextract  # noqa: F401

    print("Status: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

