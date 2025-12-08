# User Guide

## Running the viewer

Run the packaged module to start the legacy viewer:

```
python -m spox path/to/simulation.txt
```

## Loading palettes

Palettes are JSON files in `config/palettes`. Add a new palette file and
refer to it by name. The loader API returns the parsed structure:

```python
from spox.styles import load_palette
palette = load_palette("classic")
```

## Adding a new data parser

Implement the `BaseParser` protocol and register it:

```python
from spox.data import BaseParser, ParserRegistry

class CsvParser:
    format_name = "csv"

    def matches(self, path: str) -> bool:
        return path.lower().endswith(".csv")

    def load(self, path: str):
        # Parse CSV data here
        return []

registry = ParserRegistry()
registry.register(CsvParser())
```

## Sessions

Use `SessionState` to persist viewer settings independent of the GUI:

```python
from spox.session import SessionState
state = SessionState(camera={"fov": 45})
state.write("session.json")
restored = SessionState.from_file("session.json")
```
