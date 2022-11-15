"""Test the `Recipe` class implementing the `esmvaltool recipes` command."""
import textwrap

import esmvalcore.config._diagnostics
from esmvalcore._main import Recipes


def test_list(mocker, tmp_path, capsys):
    """Test the command `esmvaltool recipes list`."""
    recipe_dir = tmp_path
    recipe1 = recipe_dir / 'recipe_test1.yml'
    recipe2 = recipe_dir / 'subdir' / 'recipe_test2.yml'
    recipe1.touch()
    recipe2.parent.mkdir()
    recipe2.touch()

    diagnostics = mocker.patch.object(
        esmvalcore.config._diagnostics,
        'DIAGNOSTICS',
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
    recipe = recipe_dir / 'recipe_test.yml'
    recipe.write_text("example")

    diagnostics = mocker.patch.object(
        esmvalcore.config._diagnostics,
        'DIAGNOSTICS',
        create_autospec=True,
    )
    diagnostics.recipes = recipe_dir

    Recipes().show(recipe.name)

    msg = capsys.readouterr().out
    print(msg)
    assert f"Recipe {recipe.name}" in msg
    assert "example" in msg
