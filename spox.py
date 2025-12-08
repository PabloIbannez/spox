"""Compatibility launcher for the legacy SPOX viewer."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spox import run_legacy_main

if __name__ == "__main__":
    run_legacy_main()
