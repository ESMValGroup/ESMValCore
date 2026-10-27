"""Tests for :mod:`esmvalcore.io._woodpecker`."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import dask.array as da
import iris.coords
import iris.cube
import numpy as np
import pytest
import xarray as xr
from iris.mesh import Connectivity, MeshXY

import esmvalcore.io._woodpecker
from esmvalcore.io._woodpecker import (
    PROVENANCE_ATTRIBUTE,
    apply_fixes,
    apply_recipe,
    find_recipe,
    is_fixed_by_woodpecker,
    mask_invalid,
    merge_provenance,
)
from esmvalcore.iris_helpers import dataset_to_iris
from tests.conftest import WOODPECKER_RECIPES

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


@pytest.fixture
def recipe_path(woodpecker_recipe_path: Path) -> Path:
    return woodpecker_recipe_path


def make_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"tas": (("time",), [1.0, 2.0], {"units": "K"})},
        attrs={"title": "test"},
    )


@pytest.mark.usefixtures("recipe_path")
def test_apply_fixes_matching_recipe(caplog: pytest.LogCaptureFixture) -> None:
    dataset = make_dataset()
    with caplog.at_level(logging.DEBUG):
        applied = apply_fixes(dataset, "s3://bucket/TEST-DATA-v1.zarr")
    assert applied == ["esmvaltest.set_fixed_by"]
    assert dataset.attrs["fixed_by"] == "woodpecker"
    provenance = json.loads(dataset.attrs[PROVENANCE_ATTRIBUTE])
    assert provenance["source"] == "s3://bucket/TEST-DATA-v1.zarr"
    assert provenance["applied_fix_ids"] == ["esmvaltest.set_fixed_by"]
    assert provenance["recipe_id"] == "esmvaltest.fix_test_data"
    assert set(provenance) == {
        "run_id",
        "generated_at",
        "source",
        "recipe_id",
        "applied_fix_ids",
    }
    assert (
        "Applied woodpecker recipe esmvaltest.fix_test_data to "
        "s3://bucket/TEST-DATA-v1.zarr, fixes that changed the data: "
        "esmvaltest.set_fixed_by" in caplog.text
    )


@pytest.mark.usefixtures("recipe_path")
def test_apply_fixes_no_matching_recipe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataset = make_dataset()
    with caplog.at_level(logging.DEBUG):
        applied = apply_fixes(dataset, "s3://bucket/OTHER-DATA.zarr")
    assert applied == []
    assert "fixed_by" not in dataset.attrs
    assert "No woodpecker recipe matches" in caplog.text
    # The data is not marked, so ESMValCore applies its own fixes.
    assert PROVENANCE_ATTRIBUTE not in dataset.attrs


def test_apply_fixes_multiple_recipes(recipe_path: Path) -> None:
    second = WOODPECKER_RECIPES.replace("id: esmvaltest.fix", "id: other.fix")
    (recipe_path / "other.yaml").write_text(second, encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=(
            r"Multiple woodpecker recipes match TEST-DATA: "
            r"esmvaltest\.fix_test_data, other\.fix_test_data\."
        ),
    ):
        apply_fixes(make_dataset(), "TEST-DATA")


@pytest.mark.usefixtures("recipe_path")
def test_apply_fixes_already_fixed(caplog: pytest.LogCaptureFixture) -> None:
    dataset = make_dataset()
    dataset.attrs["fixed_by"] = "woodpecker"
    with caplog.at_level(logging.DEBUG):
        assert apply_fixes(dataset, "TEST-DATA") == []
    assert "fixes that changed the data: none" in caplog.text
    # The recipe matches, so woodpecker is responsible for the data, even
    # though it did not change it.
    provenance = json.loads(dataset.attrs[PROVENANCE_ATTRIBUTE])
    assert provenance["recipe_id"] == "esmvaltest.fix_test_data"
    assert provenance["applied_fix_ids"] == []


@pytest.mark.usefixtures("recipe_path")
def test_apply_fixes_keeps_data_lazy() -> None:
    dataset = make_dataset().chunk()
    apply_fixes(dataset, "TEST-DATA")
    assert dataset["tas"].chunks is not None


def test_is_fixed_by_woodpecker() -> None:
    cube = iris.cube.Cube([1.0])
    assert not is_fixed_by_woodpecker(cube)
    cube.attributes.globals[PROVENANCE_ATTRIBUTE] = "{}"
    assert is_fixed_by_woodpecker(cube)


@pytest.mark.usefixtures("recipe_path")
def test_marker_survives_conversion_to_iris() -> None:
    dataset = make_dataset()
    apply_fixes(dataset, "TEST-DATA")
    (cube,) = dataset_to_iris(dataset)
    assert is_fixed_by_woodpecker(cube)


def test_recipe_catalog_reloaded_when_path_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("WOODPECKER_RECIPE_PATH", str(empty))
    dataset = make_dataset()
    assert apply_fixes(dataset, "TEST-DATA") == []
    assert PROVENANCE_ATTRIBUTE not in dataset.attrs

    recipes = tmp_path / "recipes"
    recipes.mkdir()
    (recipes / "esmvaltest.yaml").write_text(
        WOODPECKER_RECIPES,
        encoding="utf-8",
    )
    monkeypatch.setenv("WOODPECKER_RECIPE_PATH", str(recipes))
    assert apply_fixes(dataset, "TEST-DATA") == ["esmvaltest.set_fixed_by"]


@pytest.mark.usefixtures("recipe_path")
def test_find_recipe_by_path() -> None:
    recipe = find_recipe(make_dataset(), "/data/TEST-DATA/tas.nc")
    assert recipe is not None
    assert recipe.id == "esmvaltest.fix_test_data"


@pytest.mark.usefixtures("recipe_path")
def test_find_recipe_by_attributes() -> None:
    dataset = make_dataset()
    dataset.attrs["source_id"] = "TEST-MODEL"
    recipe = find_recipe(dataset, "/data/tas.nc")
    assert recipe is not None
    assert recipe.id == "esmvaltest.fix_test_model"


@pytest.mark.usefixtures("recipe_path")
def test_find_recipe_only_global_attributes() -> None:
    dataset = xr.Dataset(attrs={"source_id": "TEST-MODEL"})
    recipe = find_recipe(dataset, "/data/tas.nc")
    assert recipe is not None
    assert recipe.id == "esmvaltest.fix_test_model"


@pytest.mark.usefixtures("recipe_path")
def test_find_recipe_no_match(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        assert find_recipe(make_dataset(), "/data/tas.nc") is None
    assert "No woodpecker recipe matches /data/tas.nc" in caplog.text


@pytest.mark.usefixtures("recipe_path")
def test_apply_recipe() -> None:
    recipe = find_recipe(make_dataset(), "TEST-DATA")
    assert recipe is not None
    dataset = make_dataset()
    applied = apply_recipe(dataset, recipe, "some/name")
    assert applied == ["esmvaltest.set_fixed_by"]
    assert dataset.attrs["fixed_by"] == "woodpecker"
    provenance = json.loads(dataset.attrs[PROVENANCE_ATTRIBUTE])
    assert provenance["source"] == "some/name"
    assert provenance["recipe_id"] == "esmvaltest.fix_test_data"


@pytest.mark.parametrize("lazy", [True, False])
def test_mask_invalid(lazy: bool) -> None:
    data = np.array([1.0, np.nan, np.inf, 4.0], dtype=np.float32)
    cube = iris.cube.Cube(da.from_array(data) if lazy else data)
    mask_invalid(iris.cube.CubeList([cube]))
    assert cube.has_lazy_data() is lazy
    assert cube.dtype == np.float32
    np.testing.assert_array_equal(
        np.ma.getmaskarray(cube.data),
        [False, True, True, False],
    )


def test_mask_invalid_integers() -> None:
    cube = iris.cube.Cube(np.array([1, 2], dtype=np.int16))
    mask_invalid(iris.cube.CubeList([cube]))
    assert cube.dtype == np.int16
    assert not np.ma.is_masked(cube.data)


def test_mask_invalid_coordinates_and_other_variables() -> None:
    def data(lazy: bool) -> np.ndarray | da.Array:
        array = np.array([1.0, np.nan, 3.0])
        return da.from_array(array) if lazy else array

    cube = iris.cube.Cube(np.zeros(3))
    cube.add_dim_coord(iris.coords.DimCoord([0.0, 1.0, 2.0], var_name="x"), 0)
    aux = iris.coords.AuxCoord(
        data(lazy=True),
        bounds=np.stack([data(lazy=False)] * 2, axis=-1),
        var_name="aux",
    )
    cube.add_aux_coord(aux, 0)
    cube.add_cell_measure(
        iris.coords.CellMeasure(data(lazy=False), var_name="area"),
        0,
    )
    cube.add_ancillary_variable(
        iris.coords.AncillaryVariable(data(lazy=True), var_name="flag"),
        0,
    )

    mask_invalid(iris.cube.CubeList([cube]))

    expected = [False, True, False]
    assert aux.has_lazy_points()
    np.testing.assert_array_equal(np.ma.getmaskarray(aux.points), expected)
    np.testing.assert_array_equal(
        np.ma.getmaskarray(aux.bounds),
        np.stack([expected] * 2, axis=-1),
    )
    for variable in [*cube.cell_measures(), *cube.ancillary_variables()]:
        np.testing.assert_array_equal(
            np.ma.getmaskarray(variable.data),
            expected,
        )
    assert not np.ma.is_masked(cube.coord("x").points)


def test_mask_invalid_real_data_without_invalid_values() -> None:
    data = np.array([1.0, 2.0])
    cube = iris.cube.Cube(data)
    mask_invalid(iris.cube.CubeList([cube]))
    # Nothing to mask, so the data is not replaced by a masked copy.
    assert cube.core_data() is data


def test_mask_invalid_shared_mesh(mocker: MockerFixture) -> None:
    node_x = iris.coords.AuxCoord([0.0, 1.0, np.nan], "longitude")
    node_y = iris.coords.AuxCoord([0.0, 0.0, 1.0], "latitude")
    face_x = iris.coords.AuxCoord([0.5], "longitude")
    face_y = iris.coords.AuxCoord([np.nan], "latitude")
    mesh = MeshXY(
        2,
        ((node_x, "x"), (node_y, "y")),
        Connectivity([[0, 1, 2]], cf_role="face_node_connectivity"),
        face_coords_and_axes=((face_x, "x"), (face_y, "y")),
    )
    cubes = iris.cube.CubeList()
    for name in ("tas", "pr"):
        cube = iris.cube.Cube([1.0], var_name=name)
        for coord in mesh.to_MeshCoords("face"):
            cube.add_aux_coord(coord, 0)
        cubes.append(cube)
    masked_invalid = mocker.spy(esmvalcore.io._woodpecker, "_masked_invalid")

    mask_invalid(cubes)

    # The data of the 2 cubes, and the points of the 4 mesh coordinates
    # once, not once per cube.
    assert masked_invalid.call_count == 2 + 4
    np.testing.assert_array_equal(
        np.ma.getmaskarray(node_x.points),
        [False, False, True],
    )
    for cube in cubes:
        assert np.ma.getmaskarray(cube.coord("latitude").points).all()


def test_merge_provenance() -> None:
    a = json.dumps({"source": "a.nc", "applied_fix_ids": []})
    b = json.dumps({"source": "b.nc", "applied_fix_ids": ["x"]})
    c = json.dumps({"source": "c.nc", "applied_fix_ids": []})

    assert json.loads(merge_provenance([a, a])) == json.loads(a)
    merged = merge_provenance([a, b])
    assert json.loads(merged) == [json.loads(a), json.loads(b)]
    # Merging again, e.g. after a second concatenation, flattens the list.
    assert json.loads(merge_provenance([merged, c, a])) == [
        json.loads(a),
        json.loads(b),
        json.loads(c),
    ]
