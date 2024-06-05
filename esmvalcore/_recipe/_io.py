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
            'diagnostics': {},
        }

    if isinstance(recipe, Path) or (isinstance(recipe, str)
                                    and os.path.exists(recipe)):
        recipe = Path(recipe).read_text(encoding='utf-8')

    if isinstance(recipe, str):
        recipe = yaml.safe_load(recipe)

    recipe = _copy(recipe)

    return recipe  # type: ignore
