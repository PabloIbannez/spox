# SPOX Architecture Overview

The refactor introduces a package layout that separates concerns between
rendering, data parsing, GUI composition, styles, and session handling.
The legacy implementation is preserved in `spox/legacy_viewer.py` while
features are gradually migrated into focused modules.

## Module responsibilities

- **spox.core**: Rendering adapters and scene configuration objects that
  abstract PlotOptiX specifics behind factories.
- **spox.data**: Parser registry for handling multiple data formats. New
  formats can be added by implementing `BaseParser` and registering the
  instance.
- **spox.camera**: Declarative camera configuration and projection types
  used by both rendering and GUI layers.
- **spox.gui**: Future home for Tkinter panels and event wiring. Keeping
  this module separate allows swapping GUI frameworks later.
- **spox.session**: Serialization helpers for saving and loading viewer
  state.
- **spox.styles**: Palette and material loading powered by external JSON
  files in `config/`.
- **spox.utils**: Shared helpers such as color conversions.

## Data parser example

```python
from spox.data import BaseParser, ParserRegistry

class SuperPuntoParser:
    format_name = "superpunto"

    def matches(self, path: str) -> bool:
        return path.lower().endswith(".sup")

    def load(self, path: str):
        with open(path) as handle:
            return handle.read()

registry = ParserRegistry()
registry.register(SuperPuntoParser())
parser = registry.detect("sample.sup")
if parser:
    data = parser.load("sample.sup")
```

## Palette example

Create `config/palettes/my_palette.json`:

```json
{
  "name": "my_palette",
  "colors": ["#ff0000", "#00ff00", "#0000ff"]
}
```

Load it:

```python
from spox.styles import load_palette
palette = load_palette("my_palette")
```
