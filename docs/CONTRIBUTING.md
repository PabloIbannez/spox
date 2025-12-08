# Contributing

1. Keep imports lightweight at module import time to avoid pulling heavy
   GUI/rendering dependencies during tests.
2. Prefer configuration files in `config/` for palettes, materials, and
   defaults rather than hardcoding values.
3. Add type hints and docstrings following the Google style.
4. Include unit tests for new modules in `tests/` and keep them free of
   PlotOptiX dependencies where possible.
5. Preserve backward compatibility with the legacy session format and
   command-line interface.
