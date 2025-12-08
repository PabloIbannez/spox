"""SPOX particle viewer package.

This package provides a modular structure around the legacy single-file
implementation. The legacy viewer is preserved under
:mod:`spox.legacy_viewer` and can be launched via ``python -m spox`` or
``python spox.py`` for backward compatibility.
"""
from __future__ import annotations
from importlib import import_module
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:  # pragma: no cover
    from .legacy_viewer import ParticleViewer


def get_legacy_viewer() -> Type["ParticleViewer"]:
    """Return the legacy :class:`ParticleViewer` class without importing
    it at module import time.
    """
    module = import_module("spox.legacy_viewer")
    return module.ParticleViewer


def run_legacy_main() -> None:
    """Run the legacy entrypoint in :mod:`spox.legacy_viewer`.

    This helper keeps the heavy PlotOptiX dependency from being imported
    automatically when the package is merely inspected.
    """
    module = import_module("spox.legacy_viewer")
    module.main()

__all__ = ["get_legacy_viewer", "run_legacy_main"]
