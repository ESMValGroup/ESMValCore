"""Test the `Recipe` class implementing the `esmvaltool recipes` command."""

import textwrap

import pytest

import esmvalcore.config._diagnostics
from esmvalcore._main import Recipes
from esmvalcore.exceptions import RecipeError


def test_list(mocker, tmp_path, capsys):
    """Test the command `esmvaltool recipes list`."""
    recipe_dir = tmp_path
    recipe1 = recipe_dir / "recipe_test1.yml"
    recipe2 = recipe_dir / "subdir" / "recipe_test2.yml"
    recipe1.touch()
    recipe2.parent.mkdir()
    recipe2.touch()

    diagnostics = mocker.patch.object(
        esmvalcore.config._diagnostics,
        "DIAGNOSTICS",
        create_autospec=True,
    )
    diagnostics.recipes = recipe_dir

    Recipes().list()

    msg = capsys.readouterr().out
    expected = textwrap.dedent(f"""
    # Installed recipes
    {recipe1.relative_to(recipe_dir)}

    # Subdir
    {recipe2.relative_to(recipe_dir)}
    """)
    print(msg)
    assert msg.endswith(expected)


def test_show(mocker, tmp_path, capsys):
    """Test the command `esmvaltool recipes list`."""
    recipe_dir = tmp_path
    recipe = recipe_dir / "recipe_test.yml"
    recipe.write_text("example")

    diagnostics = mocker.patch.object(
        esmvalcore.config._diagnostics,
        "DIAGNOSTICS",
        create_autospec=True,
    )
    diagnostics.recipes = recipe_dir

    Recipes().show(recipe.name)

    msg = capsys.readouterr().out
    print(msg)
    assert f"Recipe {recipe.name}" in msg
    assert "example" in msg


def test_show_fail():
    """Test the command `esmvaltool recipes show`."""
    msg = r"Recipe invalid_recipe not found."
    with pytest.raises(RecipeError, match=msg):
        Recipes().show("invalid_recipe")


def test_get(mocker, tmp_path, monkeypatch):
    """Test the command `esmvaltool recipes get`."""
    recipe_dir = tmp_path / "subdir"
    recipe = recipe_dir / "recipe_1.yml"
    recipe.parent.mkdir(parents=True, exist_ok=True)
    recipe.touch()
    diagnostics = mocker.patch.object(
        esmvalcore.config._diagnostics,
        "DIAGNOSTICS",
        create_autospec=True,
    )
    diagnostics.recipes = recipe_dir
    monkeypatch.chdir(tmp_path)

    copied_recipe_path = tmp_path / recipe.name
    assert not copied_recipe_path.is_file()

    Recipes().get(str(recipe))

    assert copied_recipe_path.is_file()


def test_get_fail():
    """Test the command `esmvaltool recipes get`."""
    msg = r"Recipe invalid_recipe not found."
    with pytest.raises(RecipeError, match=msg):
        Recipes().get("invalid_recipe")
