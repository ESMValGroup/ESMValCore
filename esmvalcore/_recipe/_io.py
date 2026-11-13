"""Functions for reading recipes."""

from __future__ import annotations

import os.path
from pathlib import Path
from typing import Any

import yaml


def _copy(item):
    """Create copies of mutable objects.

    This avoids accidental changes when a recipe contains the same
    mutable object in multiple places due to the use of YAML anchors.
    """
    if isinstance(item, dict):
        new = {k: _copy(v) for k, v in item.items()}
    elif isinstance(item, list):
        new = [_copy(v) for v in item]
    else:
        new = item
    return new


def _load_recipe(recipe: Path | str | dict[str, Any] | None) -> dict[str, Any]:
    """Load a recipe from a file, string, dict, or create a new recipe."""
    if recipe is None:
        recipe = {
            "diagnostics": {},
        }

    text = _recipe_text(recipe)
    if text is not None:
        recipe = _parse_yaml(text)

    return _copy(recipe)


def _recipe_text(recipe: Path | str | dict[str, Any] | None) -> str | None:
    """Get the YAML text of a recipe given as a file or a string."""
    if isinstance(recipe, Path) or (
        isinstance(recipe, str) and os.path.exists(recipe)
    ):
        return Path(recipe).read_text(encoding="utf-8")
    if isinstance(recipe, str):
        return recipe
    return None


def _parse_yaml(text: str) -> Any:  # noqa: ANN401
    """Parse YAML text."""
    try:
        # Use CSafeLoader for speed, but fall back to SafeLoader for better
        # error messages if parsing fails or if CSafeLoader is not available.
        return yaml.load(text, Loader=yaml.CSafeLoader)
    except (yaml.YAMLError, AttributeError):
        return yaml.safe_load(text)
