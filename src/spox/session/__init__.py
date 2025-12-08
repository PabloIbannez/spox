"""Session and configuration management."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any
import json


@dataclass
class SessionState:
    """Serializable state for the SPOX viewer."""

    camera: Dict[str, Any] = field(default_factory=dict)
    playback: Dict[str, Any] = field(default_factory=dict)
    styles: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_file(cls, path: str | Path) -> "SessionState":
        with open(path) as handle:
            payload = json.load(handle)
        return cls(**payload)

    def write(self, path: str | Path) -> None:
        with open(path, "w") as handle:
            json.dump(self.__dict__, handle, indent=2)
