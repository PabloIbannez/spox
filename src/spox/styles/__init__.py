"""Color palette and material helpers."""
from pathlib import Path
from typing import Dict, Any
import json

# Locate the repository root regardless of installation mode.
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "config"


def load_palette(name: str) -> Dict[str, Any]:
    """Load a palette definition from ``config/palettes``.

    Palettes are defined as JSON files to keep additions easy without
    touching the Python sources.
    """
    palette_file = CONFIG_DIR / "palettes" / f"{name}.json"
    with palette_file.open() as handle:
        return json.load(handle)
