"""Test the presence and content of the sample MP3 asset."""

from pathlib import Path


def test_sondy_fixed_mp3_exists_and_nonempty():
    """Ensure the sample MP3 asset is present and has some content.

    This is a first step before wiring it into an integration test.
    """
    path = Path("src/data/mp3/Sondy_FIXED.mp3")
    assert path.exists(), f"Expected MP3 at {path}, but file not found."
    size = path.stat().st_size
    assert size > 0, f"File {path} is empty (size={size})."
