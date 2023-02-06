"""Functions for reading recipes."""
from __future__ import annotations

import os.path
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _load_recipe(recipe: Path | str | dict[str, Any] | None) -> dict[str, Any]:
    """Load a recipe from a file, string, dict, or create a new recipe."""
    if recipe is None:
        recipe = {
            'diagnostics': {},
        }
    else:
        recipe = deepcopy(recipe)

    if isinstance(recipe, Path) or (isinstance(recipe, str)
                                    and os.path.exists(recipe)):
        recipe = Path(recipe).read_text(encoding='utf-8')

    if isinstance(recipe, str):
        recipe = yaml.safe_load(recipe)

    return recipe  # type: ignore
