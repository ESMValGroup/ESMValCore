"""Unit tests for the recipe writer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from esmvalcore._recipe.writer import to_yaml

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
        contents=f"{to_yaml(load_recipe)}\n",
        encoding="utf-8",
        fullpath=reference_data_path / recipe.name,
    )


def test_empty_recipe() -> None:
    """Test that the code does not crash on an empty recipe."""
    recipe_text = to_yaml({})
    assert recipe_text == "{}"


def test_recipe_file_endswith_newline(tmp_path: Path) -> None:
    """Test that the recipe ends with a newline."""
    filename = tmp_path / "recipe.yml"
    to_yaml({}, file=filename)
    assert filename.read_text(encoding="utf-8").endswith("\n")
