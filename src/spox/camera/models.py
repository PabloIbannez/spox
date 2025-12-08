"""Camera configuration models."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class ProjectionType(str, Enum):
    """Supported projection types."""

    PERSPECTIVE = "perspective"
    ORTHOGRAPHIC = "orthographic"


@dataclass
class CameraConfig:
    """Basic camera configuration shared across render backends."""

    position: Tuple[float, float, float] = (0.0, 0.0, 5.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    projection: ProjectionType = ProjectionType.PERSPECTIVE
    fov: float = 60.0

    def as_dict(self) -> dict:
        return {
            "cam_pos": self.position,
            "cam_target": self.target,
            "cam_up": self.up,
            "fov": self.fov,
        }
