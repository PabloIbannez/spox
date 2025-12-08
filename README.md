# SPOX Particle Viewer (Refactor Scaffold)

This repository contains an early modularization scaffold for the SPOX
particle viewer. The original single-file implementation now lives in
`src/spox/legacy_viewer.py` and remains runnable through `python -m spox`
or `python spox.py` for backward compatibility.

## Project Layout

```
spox/
├── config/                # Configuration-driven palettes, materials, defaults
├── docs/                  # Architecture and user-facing documentation
├── src/spox/              # New modular package scaffold
│   ├── core/              # Rendering adapters and scene configuration
│   ├── data/              # Parser registry and future data loaders
│   ├── camera/            # Camera models and projection helpers
│   ├── gui/               # GUI components (placeholder)
│   ├── session/           # Session serialization utilities
│   ├── styles/            # Palette and material loaders
│   ├── utils/             # Shared helpers
│   ├── __main__.py        # Entry point invoking the legacy viewer
│   └── legacy_viewer.py   # Original monolithic code
└── tests/                 # Tests for new modular pieces
```

## Running the legacy viewer

```
python -m spox path/to/data.txt
# or
python spox.py path/to/data.txt
```

## Extending the new architecture

- Add palettes as JSON files under `config/palettes` and load them via
  `spox.styles.load_palette`.
- Implement data parsers by conforming to `spox.data.BaseParser` and
  registering them with `ParserRegistry`.
- Use `spox.camera.CameraConfig` and `spox.core.SceneConfig` to centralize
  camera and rendering defaults as the refactor progresses.

## Development status

This refactor is in progress. The new package structure and configuration
artifacts are in place to make further extraction from the legacy file
incremental while keeping the existing workflow intact.
