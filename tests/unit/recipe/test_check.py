"""Tests for :mod:`esmvalcore._recipe.check`."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import pytest
import yaml

from esmvalcore._recipe import check
from esmvalcore.exceptions import RecipeError
from tests.unit.recipe.example_recipes import (
    DUPLICATE_RECIPE,
    INVALID_RECIPE,
    RECIPE,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_recipe_with_schema(tmp_path: Path) -> None:
    recipe = yaml.safe_load(INVALID_RECIPE)
    path = tmp_path / "recipe_test.yml"
    path.write_text(INVALID_RECIPE, encoding="utf-8")
    msg = (
        f"Recipe {path} is invalid:\n"
        "- line 5: datasets > 1: missing required key 'dataset'.\n"
        "- line 5: datasets > 1 > tier: expected at most 3, got 7.\n"
    )
    with pytest.raises(RecipeError, match=re.escape(msg)):
        check.recipe_with_schema(recipe, path)


def test_recipe_with_schema_no_file(tmp_path: Path) -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["diagnostics"]["example"].pop("scripts")
    path = tmp_path / "recipe_test.yml"
    msg = (
        f"Recipe {path} is invalid:\n"
        "- diagnostics > example: missing required key 'scripts'."
    )
    with pytest.raises(RecipeError, match=re.escape(msg)):
        check.recipe_with_schema(recipe, path)


@pytest.mark.usefixtures("esmvaltool")
def test_recipe_with_schema_warnings(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    recipe = yaml.safe_load(RECIPE)
    recipe["documentation"]["authors"] = ["author_100"]
    path = tmp_path / "recipe_test.yml"
    path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    check.recipe_with_schema(recipe, path)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    message = caplog.records[0].getMessage()
    assert message.startswith(f"Recipe {path}: line ")
    assert (
        "documentation > authors > 0: unknown value 'author_100'." in message
    )


def test_recipe_with_schema_duplicate_keys(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    recipe = yaml.safe_load(DUPLICATE_RECIPE)
    path = tmp_path / "recipe_test.yml"
    path.write_text(DUPLICATE_RECIPE, encoding="utf-8")
    check.recipe_with_schema(recipe, path)
    assert [r.levelname for r in caplog.records] == ["WARNING", "WARNING"]
    assert caplog.records[1].getMessage() == (
        f"Recipe {path}: line 16: diagnostics > example > variables > tas: "
        "duplicate key, this definition is ignored because 'tas' is defined "
        "again on line 18."
    )
