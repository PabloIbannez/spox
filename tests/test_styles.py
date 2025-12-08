from spox.styles import load_palette


def test_load_palette_classic():
    palette = load_palette("classic")
    assert palette["name"] == "classic"
    assert "colors" in palette
