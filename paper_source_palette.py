#!/usr/bin/env python3
"""Shared source palette for manuscript and presentation figures.

The palette assigns one visual identity to each data source.  The primary
colors are deliberately separated in both hue and perceptual lightness:
Tianji is the darkest source color, Pangu is a mid-light violet, and the
baseline/reference-analysis family is neutral grey.  Marker shape must remain
the secondary accessibility cue whenever sources are compared directly.
"""

from __future__ import annotations


SOURCE_COLORS = {
    "tianji": "#2E5A87",
    "pangu": "#8E6BBE",
    "baseline": "#9A9A9A",
    "era5_reference_analysis": "#9A9A9A",
}

SOURCE_DARK_COLORS = {
    "tianji": "#1E405F",
    "pangu": "#684A9B",
    "baseline": "#666A6D",
    "era5_reference_analysis": "#666A6D",
}

SOURCE_LIGHT_COLORS = {
    "tianji": "#9CB5CC",
    "pangu": "#C7B9DD",
    "baseline": "#D1D3D4",
    "era5_reference_analysis": "#D1D3D4",
}

SOURCE_PALE_COLORS = {
    "tianji": "#EDF2F6",
    "pangu": "#F3EFF8",
    "baseline": "#F2F2F2",
    "era5_reference_analysis": "#F2F2F2",
}

SOURCE_MARKERS = {
    "tianji": "s",
    "pangu": "o",
    "baseline": "D",
    "era5_reference_analysis": "D",
}


def source_color(source: str) -> str:
    """Return the primary manuscript color for a canonical source name."""

    key = source.strip().lower()
    if key not in SOURCE_COLORS:
        raise KeyError(f"Unknown paper source {source!r}; expected {sorted(SOURCE_COLORS)}")
    return SOURCE_COLORS[key]
