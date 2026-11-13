from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from esmvalcore._recipe import _io

if TYPE_CHECKING:
    from pathlib import Path


def test_copy_dict():
    a = {"a": 1}
    b = {"b": a, "c": a}
    result = _io._copy(b)
    assert result["b"] == result["c"]
    assert result["b"] is not result["c"]


def test_copy_list():
    a = ["a"]
    b = {"b": a, "c": a}
    result = _io._copy(b)
    assert result["b"] == result["c"]
    assert result["b"] is not result["c"]


@pytest.mark.parametrize("has_csafeloader", [True, False])
def test_parse_yaml(
    monkeypatch: pytest.MonkeyPatch,
    has_csafeloader: bool,
) -> None:
    if not has_csafeloader:
        monkeypatch.delattr(yaml, "CSafeLoader", raising=False)
    assert _io._parse_yaml("diagnostics: {}\n") == {"diagnostics": {}}


def test_parse_yaml_invalid() -> None:
    with pytest.raises(yaml.YAMLError, match="mapping values are not allowed"):
        _io._parse_yaml("a: b: c\n")


def test_load_recipe_from_file(tmp_path: Path) -> None:
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text("diagnostics: {}\n", encoding="utf-8")
    assert _io._load_recipe(recipe_file) == {"diagnostics": {}}


def test_load_recipe_from_text() -> None:
    assert _io._load_recipe("diagnostics: {}\n") == {"diagnostics": {}}


def test_recipe_text(tmp_path: Path) -> None:
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text("diagnostics: {}\n", encoding="utf-8")
    assert _io._recipe_text(recipe_file) == "diagnostics: {}\n"
    assert _io._recipe_text(str(recipe_file)) == "diagnostics: {}\n"
    assert _io._recipe_text("diagnostics: {}\n") == "diagnostics: {}\n"
    assert _io._recipe_text({"diagnostics": {}}) is None
