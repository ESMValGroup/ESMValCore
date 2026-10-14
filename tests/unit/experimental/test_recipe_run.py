from textwrap import dedent

import xarray as xr

from esmvalcore.experimental.recipe import Recipe


def test_run_sets_xarray_options(mocker, tmp_path, session):
    """Test that the xarray options are set while loading and running."""
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        dedent("""
            documentation:
              title: Test recipe
              description: This is a test recipe.
              authors:
                - andela_bouwe
            diagnostics: null
        """),
    )
    session["xarray_options"] = {"file_cache_maxsize": 2}
    recipe = Recipe(recipe_file)

    xarray_options = {}

    def record_options(*_, **__):
        xarray_options.update(xr.get_options())
        return mocker.DEFAULT

    mock_load = mocker.patch.object(
        Recipe,
        "_load",
        autospec=True,
        side_effect=record_options,
    )
    mocker.patch.object(Recipe, "get_output", autospec=True)
    original = xr.get_options()

    recipe.run(session=session)

    mock_load.assert_called_once_with(recipe, session=session)
    mock_load.return_value.run.assert_called_once_with()
    assert xarray_options["file_cache_maxsize"] == 2
    assert xr.get_options() == original
