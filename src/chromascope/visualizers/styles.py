"""
Shared style preset definitions for kaleidoscope visualizations.

This module loads style presets from a single JSON source so that both the
Python renderer and the web frontend can rely on the same semantic defaults.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict

from importlib import resources


@lru_cache(maxsize=1)
def load_style_presets() -> Dict[str, Any]:
    """Load all style presets from the packaged JSON file."""
    with resources.files("chromascope.visualizers").joinpath("styles.json").open(
        "r", encoding="utf-8"
    ) as f:
        return json.load(f)


def get_kaleidoscope_style(style: str) -> Dict[str, Any] | None:
    """
    Get the kaleidoscope-specific style preset for a given style name.

    Returns a dictionary of KaleidoscopeConfig-compatible overrides, or None
    if the style is unknown.
    """
    data = load_style_presets()
    all_styles = data.get("kaleidoscope", {})
    return all_styles.get(style)

