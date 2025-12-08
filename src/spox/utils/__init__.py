"""Utility helpers for SPOX."""
from colorsys import rgb_to_hsv, hsv_to_rgb
from typing import Tuple


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert a ``#RRGGBB`` string to normalized RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    return "#" + "".join(f"{int(c * 255):02x}" for c in rgb)
