"""Unit tests for the recipe writer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from esmvalcore.recipe import format_recipe

if TYPE_CHECKING:
    from pytest_regressions import FileRegressionFixture


@pytest.mark.parametrize(
    "recipe",
    list(Path(__file__).parent.glob("recipes/*.yml")),
)
def test_writer(
    tmp_path: Path,
    file_regression: FileRegressionFixture,
    recipe: Path,
) -> None:
    """Test that the recipe writer produces the expected output."""
    # Run pytest tests/unit/recipe/writer/test_writer.py --force-regen to
    # regenerate the expected output files.
    load_recipe = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    reference_data_path = Path(__file__).parent / "reference_recipes"
    file_regression.check(
        contents=format_recipe(load_recipe),
        encoding="utf-8",
        fullpath=reference_data_path / recipe.name,
    )


def test_empty_recipe() -> None:
    """Test that the code does not crash on an empty recipe."""
    assert format_recipe({}) == "{}\n"


@pytest.mark.parametrize(
    "recipe",
    list(Path(__file__).parent.glob("recipes/*.yml")),
)
def test_writer_keeps_data(recipe: Path) -> None:
    """Test that the formatted recipe contains the same data."""

    def ignore_dataset_order(data: object, key: object = None) -> object:
        if isinstance(data, dict):
            return {k: ignore_dataset_order(v, k) for k, v in data.items()}
        if isinstance(data, list):
            items = [ignore_dataset_order(v) for v in data]
            if key in ("datasets", "additional_datasets"):
                items.sort(key=repr)
            return items
        return data

    data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    formatted = yaml.safe_load(format_recipe(data))
    assert ignore_dataset_order(formatted) == ignore_dataset_order(data)
