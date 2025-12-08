"""Parser registration utilities for SPOX data formats."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Protocol


class BaseParser(Protocol):
    """Protocol describing a parser capable of loading particle data."""

    format_name: str

    def matches(self, path: str) -> bool:
        """Return ``True`` if the parser can handle the provided path."""

    def load(self, path: str) -> object:
        """Load data from ``path`` and return an in-memory representation."""


@dataclass
class ParserRegistry:
    """Registry for data format parsers.

    The registry allows pluggable parsers to be discovered by file name or
    explicit format keys. It is intentionally lightweight so that it can
    be reused by both GUI and CLI layers.
    """

    parsers: Dict[str, BaseParser]

    def __init__(self) -> None:
        self.parsers = {}

    def register(self, parser: BaseParser) -> None:
        self.parsers[parser.format_name] = parser

    def get(self, format_name: str) -> BaseParser:
        return self.parsers[format_name]

    def detect(self, path: str) -> BaseParser | None:
        for parser in self.parsers.values():
            try:
                if parser.matches(path):
                    return parser
            except Exception:
                continue
        return None

    def available_formats(self) -> Iterable[str]:
        return self.parsers.keys()
