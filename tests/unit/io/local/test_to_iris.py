from __future__ import annotations

from typing import TYPE_CHECKING

import iris
import iris.cube
import netCDF4
import numpy as np
import pytest
import xarray as xr
from iris.coords import AuxCoord
from iris.mesh import Connectivity, MeshXY

import esmvalcore.io.local
from esmvalcore.io._woodpecker import is_fixed_by_woodpecker
from esmvalcore.io.local import LocalFile, _get_attr_from_field_coord

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


@pytest.fixture
def local_file(tmp_path: Path) -> LocalFile:
    cube = iris.cube.Cube([0])
    cube.attributes.globals["attribute"] = "value"
    file = tmp_path / "test.nc"
    iris.save(cube, file)
    return LocalFile(file)


def test_to_iris(local_file: LocalFile) -> None:
    cubes = local_file.to_iris()
    assert len(cubes) == 1


def test_attributes(local_file: LocalFile) -> None:
    local_file.to_iris()  # Load the file to populate attributes
    attrs = local_file.attributes
    assert attrs["attribute"] == "value"


def test_attributes_without_loading(local_file: LocalFile) -> None:
    """Test that accessing attributes without loading the file first raises."""
    with pytest.raises(
        ValueError,
        match=r"Attributes have not been read yet.*",
    ):
        local_file.attributes  # noqa: B018


def test_get_attr_from_field_coord_none(mocker: MockerFixture) -> None:
    """Test ``_get_attr_from_field_coord``."""
    attr = _get_attr_from_field_coord(mocker.sentinel.ncfield, None, "attr")
    assert attr is None


def _write_netcdf(path: Path, **attributes: str) -> LocalFile:
    dataset = xr.Dataset(
        {
            "tas": (
                ("time", "lat"),
                np.array([[280.0, np.nan], [281.0, 282.0]], dtype=np.float32),
                {"standard_name": "air_temperature", "units": "K"},
            ),
        },
        coords={
            "time": ("time", [0.5, 1.5], {"units": "days since 2000-01-01"}),
            "lat": (
                "lat",
                [-45.0, 45.0],
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
        },
        attrs=attributes,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)
    return LocalFile(path)


@pytest.mark.usefixtures("woodpecker_recipe_path")
def test_to_iris_woodpecker(tmp_path: Path) -> None:
    local_file = _write_netcdf(tmp_path / "TEST-DATA" / "tas.nc", title="t")

    cubes = local_file.to_iris()

    (cube,) = cubes.extract("air_temperature")
    assert is_fixed_by_woodpecker(cube)
    assert cube.attributes.globals["fixed_by"] == "woodpecker"
    assert cube.attributes.globals["source_file"] == str(local_file)
    assert cube.has_lazy_data()
    # Woodpecker closes the dataset after fixing it, the data must still be
    # readable.
    np.testing.assert_array_equal(
        np.ma.masked_invalid(cube.data),
        np.ma.masked_invalid([[280.0, np.nan], [281.0, 282.0]]),
    )
    assert cube.coord("time").units == "days since 2000-01-01"
    assert local_file.attributes["fixed_by"] == "woodpecker"
    assert local_file.attributes["source_file"] == str(local_file)


@pytest.mark.usefixtures("woodpecker_recipe_path")
def test_to_iris_no_woodpecker_recipe(tmp_path: Path) -> None:
    local_file = _write_netcdf(tmp_path / "tas.nc", title="t")

    cubes = local_file.to_iris()

    (cube,) = cubes.extract("air_temperature")
    assert not is_fixed_by_woodpecker(cube)
    assert "fixed_by" not in cube.attributes.globals
    assert local_file.attributes["title"] == "t"


@pytest.mark.usefixtures("woodpecker_recipe_path")
def test_woodpecker_recipe_by_attributes(tmp_path: Path) -> None:
    local_file = _write_netcdf(tmp_path / "tas.nc", source_id="TEST-MODEL")
    recipe = local_file._woodpecker_recipe()
    assert recipe is not None
    assert recipe.id == "esmvaltest.fix_test_model"


@pytest.mark.usefixtures("woodpecker_recipe_path")
def test_woodpecker_recipe_by_full_path(tmp_path: Path) -> None:
    local_file = _write_netcdf(tmp_path / "TEST-DATA" / "tas.nc")
    recipe = local_file._woodpecker_recipe()
    assert recipe is not None
    assert recipe.id == "esmvaltest.fix_test_data"


@pytest.mark.usefixtures("woodpecker_recipe_path")
@pytest.mark.parametrize("filename", ["TEST-DATA.grib", "TEST-DATA.txt"])
def test_woodpecker_recipe_not_netcdf(
    tmp_path: Path,
    mocker: MockerFixture,
    filename: str,
) -> None:
    netcdf = mocker.patch.object(esmvalcore.io.local, "Dataset")
    local_file = LocalFile(tmp_path / filename)
    assert local_file._woodpecker_recipe() is None
    netcdf.assert_not_called()


@pytest.mark.usefixtures("woodpecker_recipe_path")
def test_woodpecker_recipe_header_read_once(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    local_file = _write_netcdf(tmp_path / "TEST-DATA" / "tas.nc")
    netcdf = mocker.spy(esmvalcore.io.local, "Dataset")
    stat = mocker.spy(LocalFile, "stat")
    local_file._woodpecker_recipe()
    # Another LocalFile for the same path, as created by fix_file and load
    # when they are given a plain Path.
    LocalFile(local_file).to_iris()
    netcdf.assert_called_once()
    stat.assert_not_called()


@pytest.mark.usefixtures("woodpecker_recipe_path")
def test_to_iris_woodpecker_masks_nans(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """NaNs are masked, also if a fix removed the fill value from the encoding."""
    local_file = _write_netcdf(tmp_path / "TEST-DATA" / "tas.nc")

    def apply_recipe(dataset: xr.Dataset, *_: object) -> list[str]:
        # Arithmetic drops the encoding, including the _FillValue.
        dataset["tas"] = dataset["tas"] * 1
        assert "_FillValue" not in dataset["tas"].encoding
        dataset.attrs["woodpecker_provenance"] = "{}"
        return []

    mocker.patch.object(esmvalcore.io.local, "apply_recipe", apply_recipe)

    (cube,) = local_file.to_iris().extract("air_temperature")

    assert cube.has_lazy_data()
    np.testing.assert_array_equal(
        np.ma.getmaskarray(cube.data),
        [[False, True], [False, False]],
    )


@pytest.mark.usefixtures("woodpecker_recipe_path")
def test_to_iris_woodpecker_default_fill_value_coordinate(
    tmp_path: Path,
) -> None:
    """Default fill values of auxiliary coordinates are masked as by Iris."""
    path = tmp_path / "TEST-DATA" / "x.nc"
    path.parent.mkdir()
    fill_value = netCDF4.default_fillvals["f8"]
    xr.Dataset(
        {"x": ("i", np.array([1, 2, 3], dtype="f4"))},
        coords={
            "lat": (
                "i",
                [0.0, fill_value, 10.0],
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
        },
    ).to_netcdf(path, encoding={"lat": {"_FillValue": None}})
    expected = iris.load_cube(path).coord("latitude")

    (cube,) = LocalFile(path).to_iris()

    coord = cube.coord("latitude")
    assert coord.units == "degrees_north"
    np.testing.assert_array_equal(
        np.ma.getmaskarray(coord.points),
        np.ma.getmaskarray(expected.points),
    )
    np.testing.assert_array_equal(
        np.ma.getmaskarray(coord.points),
        [False, True, False],
    )


@pytest.mark.usefixtures("woodpecker_recipe_path")
def test_to_iris_woodpecker_mesh(tmp_path: Path) -> None:
    """UGRID meshes, including masked connectivities, are loaded as by Iris."""
    node_x = AuxCoord([0.0, 1.0, 1.0, 0.0, 2.0], "longitude", units="degrees")
    node_y = AuxCoord([0.0, 0.0, 1.0, 1.0, 0.5], "latitude", units="degrees")
    face_x = AuxCoord([0.5, 1.3], "longitude", units="degrees")
    face_y = AuxCoord([0.5, 0.5], "latitude", units="degrees")
    # A quad and a triangle, so one index of the triangle is masked.
    indices = np.ma.masked_array(
        [[0, 1, 2, 3], [1, 4, 2, -1]],
        mask=[[False, False, False, False], [False, False, False, True]],
        dtype=np.int32,
    )
    mesh = MeshXY(
        2,
        ((node_x, "x"), (node_y, "y")),
        Connectivity(indices, cf_role="face_node_connectivity"),
        face_coords_and_axes=((face_x, "x"), (face_y, "y")),
    )
    cube = iris.cube.Cube(
        np.ma.masked_array([280.0, 0.0], mask=[False, True], dtype=np.float32),
        var_name="tas",
        units="K",
    )
    for coord in mesh.to_MeshCoords("face"):
        cube.add_aux_coord(coord, 0)
    path = tmp_path / "TEST-DATA" / "tas.nc"
    path.parent.mkdir()
    iris.save(cube, path)

    (loaded,) = LocalFile(path).to_iris()

    assert is_fixed_by_woodpecker(loaded)
    assert loaded.mesh is not None
    connectivity = loaded.mesh.connectivity(cf_role="face_node_connectivity")
    assert connectivity.indices.dtype == np.int32
    np.testing.assert_array_equal(
        np.ma.getmaskarray(connectivity.indices),
        np.ma.getmaskarray(indices),
    )
    np.testing.assert_array_equal(
        np.ma.getmaskarray(loaded.data),
        [False, True],
    )


def test_to_iris_mesh(tmp_path: Path) -> None:
    """UGRID files can be loaded without a woodpecker recipe."""
    node_x = AuxCoord([0.0, 1.0, 1.0, 0.0], "longitude", units="degrees")
    node_y = AuxCoord([0.0, 0.0, 1.0, 1.0], "latitude", units="degrees")
    face_x = AuxCoord([0.5], "longitude", units="degrees")
    face_y = AuxCoord([0.5], "latitude", units="degrees")
    mesh = MeshXY(
        2,
        ((node_x, "x"), (node_y, "y")),
        Connectivity([[0, 1, 2, 3]], cf_role="face_node_connectivity"),
        face_coords_and_axes=((face_x, "x"), (face_y, "y")),
    )
    cube = iris.cube.Cube([280.0], var_name="tas", units="K")
    for coord in mesh.to_MeshCoords("face"):
        cube.add_aux_coord(coord, 0)
    path = tmp_path / "tas.nc"
    iris.save(cube, path)

    (loaded,) = LocalFile(path).to_iris()

    assert loaded.mesh is not None


@pytest.mark.usefixtures("woodpecker_recipe_path")
def test_to_iris_woodpecker_masks_like_netcdf4(tmp_path: Path) -> None:
    """Values are masked as by Iris, before woodpecker fixes the data."""
    path = tmp_path / "TEST-DATA" / "x.nc"
    path.parent.mkdir()
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("i", 4)
        variable = dataset.createVariable("x", "i1", ("i",))
        variable.valid_range = np.array([0, 10], dtype="i1")
        variable[:] = np.array([-1, 5, 11, netCDF4.default_fillvals["i1"]])
    expected = iris.load_cube(path)

    (cube,) = LocalFile(path).to_iris()

    assert is_fixed_by_woodpecker(cube)
    assert cube.dtype == np.int8
    assert "valid_range" not in cube.attributes
    np.testing.assert_array_equal(
        np.ma.getmaskarray(cube.data),
        np.ma.getmaskarray(expected.data),
    )
    np.testing.assert_array_equal(
        np.ma.getmaskarray(cube.data),
        [True, False, True, True],
    )
