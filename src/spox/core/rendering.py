"""Rendering adapters for PlotOptiX integration.

These classes are scaffolding for a future refactor that will detach the
rendering engine from the GUI and data loading concerns.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SceneConfig:
    """Configuration for initializing a render scene."""

    background_color: str = "#000000"
    tonemap: str = "filmic"
    samples_per_frame: int = 1
    denoiser: bool = True
    camera: Dict[str, Any] = field(default_factory=dict)


class RendererFactory:
    """Factory responsible for creating renderer instances.

    The current codebase instantiates :class:`plotoptix.TkOptiX` directly.
    Introducing a factory allows swapping implementations (or mocking in
    tests) without touching the caller code.
    """

    def __init__(self, config: SceneConfig | None = None) -> None:
        self.config = config or SceneConfig()

    def create(self) -> Any:
        """Create a renderer instance.

        This method currently serves as a placeholder; the legacy
        implementation constructs the renderer inside the viewer. New
        code can delegate to this method to centralize renderer
        configuration.
        """

        # Deferred import to keep optional dependencies light for tooling
        # and unit tests.
        from plotoptix import TkOptiX  # type: ignore

        renderer = TkOptiX()
        renderer.set_param(**self.config.camera)
        return renderer
