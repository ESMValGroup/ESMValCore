"""Tests for :mod:`esmvalcore.iris_helpers`."""

import datetime
import warnings
from copy import deepcopy
from itertools import permutations
from pprint import pformat
from unittest import mock

import dask.array as da
import iris.analysis
import ncdata
import numpy as np
import pytest
import xarray as xr
from cf_units import Unit
from iris.coords import (
    AncillaryVariable,
    AuxCoord,
    CellMeasure,
    CellMethod,
    DimCoord,
)
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateMultiDimError, UnitConversionError
from iris.warnings import IrisLoadWarning, IrisUnknownCellMethodWarning

from esmvalcore.iris_helpers import (
    add_leading_dim_to_cube,
    dataset_to_iris,
    date2num,
    has_irregular_grid,
    has_regular_grid,
    has_unstructured_grid,
    ignore_iris_vague_metadata_warnings,
    merge_cube_attributes,
    rechunk_cube,
    safe_convert_units,
)


@pytest.fixture
def cubes():
    """Test cubes."""
    return CubeList(
        [
            Cube(0.0, var_name="a", long_name="a"),
            Cube(0.0, var_name="a", long_name="b"),
            Cube(0.0, var_name="c", long_name="d"),
        ],
    )


@pytest.fixture
def units():
    return Unit("days since 0001-01-01", calendar="proleptic_gregorian")


def test_add_leading_dim_to_cube():
    """Test :func:`esmvalcore.iris_helpers.add_leading_dim_to_cube`."""
    lat_coord = DimCoord(
        [0.0, 1.0, 2.0],
        var_name="lat",
        standard_name="latitude",
        long_name="latitude",
        units="degrees_north",
    )
    lon_coord = DimCoord(
        [0.0, 1.0],
        var_name="lon",
        standard_name="longitude",
        long_name="longitude",
        units="degrees_east",
    )
    height_coord = AuxCoord(
        [2.0, 1.0],
        var_name="height",
        standard_name="height",
        long_name="height",
        units="m",
        attributes={"positive": "up"},
    )
    land_mask = AncillaryVariable(
        [0.5, 0.2],
        var_name="sftlf",
        standard_name=None,
        long_name="Land fraction",
        units="1",
    )
    cell_area = CellMeasure(
        [1.0, 2.0],
        var_name="areacella",
        standard_name="cell_area",
        long_name="Cell Area",
        units="m2",
        measure="area",
    )
    cube = Cube(
        [1, 42],
        var_name="ta",
        standard_name="air_temperature",
        long_name="Air Temperature",
        units="K",
        attributes={"model_name": "ESM"},
        cell_methods=[CellMethod("mean", coords="time")],
        aux_coords_and_dims=[(height_coord, 0)],
        dim_coords_and_dims=[(lon_coord, 0)],
        ancillary_variables_and_dims=[(land_mask, 0)],
        cell_measures_and_dims=[(cell_area, 0)],
    )

    new_cube = add_leading_dim_to_cube(cube, lat_coord)

    np.testing.assert_equal(new_cube.data, [[1, 42], [1, 42], [1, 42]])

    assert new_cube.var_name == "ta"
    assert new_cube.standard_name == "air_temperature"
    assert new_cube.long_name == "Air Temperature"
    assert new_cube.units == "K"
    assert new_cube.cell_methods == (CellMethod("mean", coords="time"),)
    assert new_cube.attributes == {"model_name": "ESM"}

    assert new_cube.coords(lat_coord, dim_coords=True)
    assert new_cube.coords(lon_coord, dim_coords=True)
    assert new_cube.coords(height_coord, dim_coords=False)
    assert new_cube.coord_dims(lat_coord) == (0,)
    assert new_cube.coord_dims(lon_coord) == (1,)
    assert new_cube.coord_dims(height_coord) == (1,)

    assert new_cube.ancillary_variables(land_mask)
    assert new_cube.cell_measures(cell_area)
    assert new_cube.ancillary_variable_dims(land_mask) == (1,)
    assert new_cube.cell_measure_dims(cell_area) == (1,)


def test_add_leading_dim_to_cube_non_1d():
    """Test :func:`esmvalcore.iris_helpers.add_leading_dim_to_cube`."""
    coord_2d = AuxCoord([[0, 1], [2, 3]], var_name="coord_2d")
    msg = "Multi-dimensional coordinate not supported: 'coord_2d'"
    with pytest.raises(CoordinateMultiDimError, match=msg):
        add_leading_dim_to_cube(mock.sentinel.cube, coord_2d)


@pytest.mark.parametrize(
    ("date", "dtype", "expected"),
    [
        (datetime.datetime(1, 1, 1), np.float64, 0.0),
        (datetime.datetime(1, 1, 1), int, 0.0),
        (datetime.datetime(1, 1, 2, 12), np.float64, 1.5),
    ],
)
def test_date2num_scalar(date, dtype, expected, units):
    num = date2num(date, units, dtype=dtype)
    assert num == expected
    assert num.dtype == dtype


def assert_attributes_equal(attrs_1: dict, attrs_2: dict) -> None:
    """Check attributes using :func:`numpy.testing.assert_array_equal`."""
    print(pformat(dict(attrs_1)))
    print(pformat(dict(attrs_2)))
    assert len(attrs_1) == len(attrs_2)
    for attr, val in attrs_1.items():
        assert attr in attrs_2
        np.testing.assert_array_equal(attrs_2[attr], val)


def make_cube_with_attrs(index):
    """Make cube that contains different types of attributes."""
    attributes = {
        # Identical attribute values across cubes
        "int": 42,
        "float": 3.1415,
        "bool": True,
        "str": "Hello, world",
        "list": [1, 1, 2, 3, 5, 8, 13],
        "tuple": (1, 2, 3, 4, 5),
        "nparray": np.arange(42),
        # Differing attribute values across cubes
        "diff_int": index,
        "diff_str": "abc"[index],
        "diff_nparray": np.arange(index),
        "mix": np.arange(3) if index == 0 else index,
        "diff_list": [index, index],
        "diff_tuple": (index, index),
        # Differing attribute keys across cubes
        str(index + 1000): index,
        str(index % 2 + 100): index,
        str(index % 2): index % 2,
    }
    return Cube(0.0, attributes=attributes)


CUBES = [make_cube_with_attrs(i) for i in range(3)]


# Test all permutations of CUBES to test that results do not depend on order
@pytest.mark.parametrize("cubes", list(permutations(CUBES)))
def test_merge_cube_attributes(cubes):
    """Test `merge_cube_attributes`."""
    expected_attributes = {
        "int": 42,
        "float": 3.1415,
        "bool": True,
        "str": "Hello, world",
        "list": [1, 1, 2, 3, 5, 8, 13],
        "tuple": (1, 2, 3, 4, 5),
        "nparray": np.arange(42),
        "diff_int": "0 1 2",
        "diff_str": "a b c",
        "diff_nparray": "[0 1] [0] []",
        "mix": "1 2 [0 1 2]",
        "diff_list": "[0, 0] [1, 1] [2, 2]",
        "diff_tuple": "(0, 0) (1, 1) (2, 2)",
        "1000": 0,
        "1001": 1,
        "1002": 2,
        "100": "0 2",
        "101": 1,
        "0": 0,
        "1": 1,
    }
    cubes = deepcopy(cubes)
    merge_cube_attributes(cubes)
    assert len(cubes) == 3
    for cube in cubes:
        assert_attributes_equal(cube.attributes, expected_attributes)


def test_merge_cube_attributes_0_cubes():
    """Test `merge_cube_attributes` with 0 cubes."""
    merge_cube_attributes([])


def test_merge_cube_attributes_1_cube():
    """Test `merge_cube_attributes` with 1 cube."""
    cubes = CubeList([deepcopy(CUBES[0])])
    expected_attributes = deepcopy(cubes[0].attributes)
    merge_cube_attributes(cubes)
    assert len(cubes) == 1
    assert_attributes_equal(cubes[0].attributes, expected_attributes)


def test_merge_cube_attributes_global_local():
    cube1 = CUBES[0].copy()
    cube2 = CUBES[1].copy()
    cube1.attributes.globals["attr1"] = 1
    cube1.attributes.globals["attr2"] = 1
    cube1.attributes.globals["attr3"] = 1
    cube2.attributes.locals["attr1"] = 1
    merge_cube_attributes([cube1, cube2])
    for cube in [cube1, cube2]:
        for attr in ["attr1", "attr2", "attr3"]:
            assert attr in cube.attributes.globals


@pytest.fixture
def cube_3d():
    """3D sample cube."""
    # DimCoords
    x = DimCoord([0, 1, 2], var_name="x")
    y = DimCoord([0, 1, 2], var_name="y")
    z = DimCoord([0, 1, 2, 3], var_name="z")

    # AuxCoords
    aux_x = AuxCoord(
        da.ones(3, chunks=1),
        bounds=da.ones((3, 3), chunks=(1, 1)),
        var_name="aux_x",
    )
    aux_z = AuxCoord(da.ones(4, chunks=1), var_name="aux_z")
    aux_xy = AuxCoord(da.ones((3, 3), chunks=(1, 1)), var_name="xy")
    aux_xz = AuxCoord(da.ones((3, 4), chunks=(1, 1)), var_name="xz")
    aux_yz = AuxCoord(da.ones((3, 4), chunks=(1, 1)), var_name="yz")
    aux_xyz = AuxCoord(
        da.ones((3, 3, 4), chunks=(1, 1, 1)),
        bounds=da.ones((3, 3, 4, 3), chunks=(1, 1, 1, 1)),
        var_name="xyz",
    )
    aux_coords_and_dims = [
        (aux_x, 0),
        (aux_z, 2),
        (aux_xy, (0, 1)),
        (aux_xz, (0, 2)),
        (aux_yz, (1, 2)),
        (aux_xyz, (0, 1, 2)),
    ]

    # CellMeasures and AncillaryVariables
    cell_measure = CellMeasure(
        da.ones((3, 4), chunks=(1, 1)),
        var_name="cell_measure",
    )
    anc_var = AncillaryVariable(
        da.ones((3, 4), chunks=(1, 1)),
        var_name="anc_var",
    )

    return Cube(
        da.ones((3, 3, 4), chunks=(1, 1, 1)),
        var_name="cube",
        dim_coords_and_dims=[(x, 0), (y, 1), (z, 2)],
        aux_coords_and_dims=aux_coords_and_dims,
        cell_measures_and_dims=[(cell_measure, (1, 2))],
        ancillary_variables_and_dims=[(anc_var, (0, 2))],
    )


def test_rechunk_cube_fully_lazy(cube_3d):
    """Test ``rechunk_cube``."""
    input_cube = cube_3d.copy()

    x_coord = input_cube.coord("x")
    result = rechunk_cube(input_cube, [x_coord, "y"], remaining_dims=2)

    assert input_cube == cube_3d
    assert result == cube_3d
    assert result.core_data().chunksize == (3, 3, 2)
    assert result.coord("aux_x").core_points().chunksize == (3,)
    assert result.coord("aux_z").core_points().chunksize == (1,)
    assert result.coord("xy").core_points().chunksize == (3, 3)
    assert result.coord("xz").core_points().chunksize == (3, 2)
    assert result.coord("yz").core_points().chunksize == (3, 2)
    assert result.coord("xyz").core_points().chunksize == (3, 3, 2)
    assert result.coord("aux_x").core_bounds().chunksize == (3, 2)
    assert result.coord("aux_z").core_bounds() is None
    assert result.coord("xy").core_bounds() is None
    assert result.coord("xz").core_bounds() is None
    assert result.coord("yz").core_bounds() is None
    assert result.coord("xyz").core_bounds().chunksize == (3, 3, 2, 2)
    assert result.cell_measure("cell_measure").core_data().chunksize == (3, 2)
    assert result.ancillary_variable("anc_var").core_data().chunksize == (3, 2)


@pytest.mark.parametrize("complete_dims", [["x", "y"], ["xy"]])
def test_rechunk_cube_partly_lazy(cube_3d, complete_dims):
    """Test ``rechunk_cube``."""
    input_cube = cube_3d.copy()

    # Realize some arrays
    input_cube.data  # noqa: B018
    input_cube.coord("xyz").points  # noqa: B018
    input_cube.coord("xyz").bounds  # noqa: B018
    input_cube.cell_measure("cell_measure").data  # noqa: B018

    result = rechunk_cube(input_cube, complete_dims, remaining_dims=2)

    assert input_cube == cube_3d
    assert result == cube_3d
    assert not result.has_lazy_data()
    assert result.coord("aux_x").core_points().chunksize == (3,)
    assert result.coord("aux_z").core_points().chunksize == (1,)
    assert result.coord("xy").core_points().chunksize == (3, 3)
    assert result.coord("xz").core_points().chunksize == (3, 2)
    assert result.coord("yz").core_points().chunksize == (3, 2)
    assert not result.coord("xyz").has_lazy_points()
    assert result.coord("aux_x").core_bounds().chunksize == (3, 2)
    assert result.coord("aux_z").core_bounds() is None
    assert result.coord("xy").core_bounds() is None
    assert result.coord("xz").core_bounds() is None
    assert result.coord("yz").core_bounds() is None
    assert not result.coord("xyz").has_lazy_bounds()
    assert not result.cell_measure("cell_measure").has_lazy_data()
    assert result.ancillary_variable("anc_var").core_data().chunksize == (3, 2)


@pytest.fixture
def lat_coord_1d():
    """1D latitude coordinate."""
    return DimCoord([0, 1], standard_name="latitude")


@pytest.fixture
def lon_coord_1d():
    """1D longitude coordinate."""
    return DimCoord([0, 1], standard_name="longitude")


@pytest.fixture
def lat_coord_2d():
    """2D latitude coordinate."""
    return AuxCoord([[0, 1]], standard_name="latitude")


@pytest.fixture
def lon_coord_2d():
    """2D longitude coordinate."""
    return AuxCoord([[0, 1]], standard_name="longitude")


def test_has_regular_grid_no_lat_lon():
    """Test `has_regular_grid`."""
    cube = Cube(0)
    assert has_regular_grid(cube) is False


def test_has_regular_grid_no_lat(lon_coord_1d):
    """Test `has_regular_grid`."""
    cube = Cube([0, 1], dim_coords_and_dims=[(lon_coord_1d, 0)])
    assert has_regular_grid(cube) is False


def test_has_regular_grid_no_lon(lat_coord_1d):
    """Test `has_regular_grid`."""
    cube = Cube([0, 1], dim_coords_and_dims=[(lat_coord_1d, 0)])
    assert has_regular_grid(cube) is False


def test_has_regular_grid_2d_lat(lat_coord_2d, lon_coord_1d):
    """Test `has_regular_grid`."""
    cube = Cube(
        [[0, 1]],
        dim_coords_and_dims=[(lon_coord_1d, 1)],
        aux_coords_and_dims=[(lat_coord_2d, (0, 1))],
    )
    assert has_regular_grid(cube) is False


def test_has_regular_grid_2d_lon(lat_coord_1d, lon_coord_2d):
    """Test `has_regular_grid`."""
    cube = Cube(
        [[0, 1]],
        dim_coords_and_dims=[(lat_coord_1d, 1)],
        aux_coords_and_dims=[(lon_coord_2d, (0, 1))],
    )
    assert has_regular_grid(cube) is False


def test_has_regular_grid_2d_lat_lon(lat_coord_2d, lon_coord_2d):
    """Test `has_regular_grid`."""
    cube = Cube(
        [[0, 1]],
        aux_coords_and_dims=[(lat_coord_2d, (0, 1)), (lon_coord_2d, (0, 1))],
    )
    assert has_regular_grid(cube) is False


def test_has_regular_grid_unstructured(lat_coord_1d, lon_coord_1d):
    """Test `has_regular_grid`."""
    cube = Cube(
        [[0, 1], [2, 3]],
        aux_coords_and_dims=[(lat_coord_1d, 0), (lon_coord_1d, 0)],
    )
    assert has_regular_grid(cube) is False


def test_has_regular_grid_true(lat_coord_1d, lon_coord_1d):
    """Test `has_regular_grid`."""
    cube = Cube(
        [[0, 1], [2, 3]],
        dim_coords_and_dims=[(lat_coord_1d, 0), (lon_coord_1d, 1)],
    )
    assert has_regular_grid(cube) is True


def test_has_irregular_grid_no_lat_lon():
    """Test `has_irregular_grid`."""
    cube = Cube(0)
    assert has_irregular_grid(cube) is False


def test_has_irregular_grid_no_lat(lon_coord_2d):
    """Test `has_irregular_grid`."""
    cube = Cube([[0, 1]], aux_coords_and_dims=[(lon_coord_2d, (0, 1))])
    assert has_irregular_grid(cube) is False


def test_has_irregular_grid_no_lon(lat_coord_2d):
    """Test `has_irregular_grid`."""
    cube = Cube([[0, 1]], aux_coords_and_dims=[(lat_coord_2d, (0, 1))])
    assert has_irregular_grid(cube) is False


def test_has_irregular_grid_1d_lon(lat_coord_2d, lon_coord_1d):
    """Test `has_irregular_grid`."""
    cube = Cube(
        [[0, 1]],
        dim_coords_and_dims=[(lon_coord_1d, 1)],
        aux_coords_and_dims=[(lat_coord_2d, (0, 1))],
    )
    assert has_irregular_grid(cube) is False


def test_has_irregular_grid_1d_lat(lat_coord_1d, lon_coord_2d):
    """Test `has_irregular_grid`."""
    cube = Cube(
        [[0, 1]],
        dim_coords_and_dims=[(lat_coord_1d, 1)],
        aux_coords_and_dims=[(lon_coord_2d, (0, 1))],
    )
    assert has_irregular_grid(cube) is False


def test_has_irregular_grid_1d_lat_lon(lat_coord_1d, lon_coord_1d):
    """Test `has_irregular_grid`."""
    cube = Cube(
        [0, 1],
        aux_coords_and_dims=[(lat_coord_1d, 0), (lon_coord_1d, 0)],
    )
    assert has_irregular_grid(cube) is False


def test_has_irregular_grid_regular_grid(lat_coord_1d, lon_coord_1d):
    """Test `has_irregular_grid`."""
    cube = Cube(
        [[0, 1], [2, 3]],
        dim_coords_and_dims=[(lat_coord_1d, 0), (lon_coord_1d, 1)],
    )
    assert has_irregular_grid(cube) is False


def test_has_irregular_grid_true(lat_coord_2d, lon_coord_2d):
    """Test `has_irregular_grid`."""
    cube = Cube(
        [[0, 1]],
        aux_coords_and_dims=[(lat_coord_2d, (0, 1)), (lon_coord_2d, (0, 1))],
    )
    assert has_irregular_grid(cube) is True


def test_has_unstructured_grid_no_lat_lon():
    """Test `has_unstructured_grid`."""
    cube = Cube(0)
    assert has_unstructured_grid(cube) is False


def test_has_unstructured_grid_no_lat(lon_coord_1d):
    """Test `has_unstructured_grid`."""
    cube = Cube([0, 1], dim_coords_and_dims=[(lon_coord_1d, 0)])
    assert has_unstructured_grid(cube) is False


def test_has_unstructured_grid_no_lon(lat_coord_1d):
    """Test `has_unstructured_grid`."""
    cube = Cube([0, 1], dim_coords_and_dims=[(lat_coord_1d, 0)])
    assert has_unstructured_grid(cube) is False


def test_has_unstructured_grid_2d_lat(lat_coord_2d, lon_coord_1d):
    """Test `has_unstructured_grid`."""
    cube = Cube(
        [[0, 1]],
        dim_coords_and_dims=[(lon_coord_1d, 1)],
        aux_coords_and_dims=[(lat_coord_2d, (0, 1))],
    )
    assert has_unstructured_grid(cube) is False


def test_has_unstructured_grid_2d_lon(lat_coord_1d, lon_coord_2d):
    """Test `has_unstructured_grid`."""
    cube = Cube(
        [[0, 1]],
        dim_coords_and_dims=[(lat_coord_1d, 1)],
        aux_coords_and_dims=[(lon_coord_2d, (0, 1))],
    )
    assert has_unstructured_grid(cube) is False


def test_has_unstructured_grid_2d_lat_lon(lat_coord_2d, lon_coord_2d):
    """Test `has_unstructured_grid`."""
    cube = Cube(
        [[0, 1]],
        aux_coords_and_dims=[(lat_coord_2d, (0, 1)), (lon_coord_2d, (0, 1))],
    )
    assert has_unstructured_grid(cube) is False


def test_has_unstructured_grid_regular_grid(lat_coord_1d, lon_coord_1d):
    """Test `has_unstructured_grid`."""
    cube = Cube(
        [[0, 1], [2, 3]],
        dim_coords_and_dims=[(lat_coord_1d, 0), (lon_coord_1d, 1)],
    )
    assert has_unstructured_grid(cube) is False


def test_has_unstructured_grid_true(lat_coord_1d, lon_coord_1d):
    """Test `has_unstructured_grid`."""
    cube = Cube(
        [[0, 1], [2, 3]],
        aux_coords_and_dims=[(lat_coord_1d, 0), (lon_coord_1d, 0)],
    )
    assert has_unstructured_grid(cube) is True


@pytest.mark.parametrize(
    (
        "old_units",
        "new_units",
        "old_standard_name",
        "new_standard_name",
        "err_msg",
    ),
    [
        ("m", "km", "altitude", "altitude", None),
        ("Pa", "hPa", "air_pressure", "air_pressure", None),
        (
            "m",
            "DU",
            "equivalent_thickness_at_stp_of_atmosphere_ozone_content",
            "equivalent_thickness_at_stp_of_atmosphere_ozone_content",
            None,
        ),
        (
            "m",
            "s",
            "altitude",
            ValueError,
            r"Unable to convert from 'Unit\('m'\)' to 'Unit\('s'\)'",
        ),
        (
            "unknown",
            "s",
            "air_temperature",
            UnitConversionError,
            r"Cannot convert from unknown units",
        ),
        (
            "kg m-2 s-1",
            "mm day-1",
            "precipitation_flux",
            ValueError,
            r"Cannot safely convert units from 'kg m-2 s-1' to 'mm day-1'; "
            r"standard_name changed from 'precipitation_flux' to "
            r"'lwe_precipitation_rate'",
        ),
    ],
)
def test_safe_convert_units(
    old_units,
    new_units,
    old_standard_name,
    new_standard_name,
    err_msg,
):
    """Test ``esmvalcore.preprocessor._units.safe_convert_units``."""
    cube = Cube(0, standard_name=old_standard_name, units=old_units)

    # Exceptions
    if isinstance(new_standard_name, type):
        with pytest.raises(new_standard_name, match=err_msg):
            safe_convert_units(cube, new_units)
        return

    # Regular test cases
    new_cube = safe_convert_units(cube, new_units)
    assert new_cube is cube
    assert new_cube.standard_name == new_standard_name
    assert new_cube.units == new_units


def test_ignore_iris_vague_metadata_warnings():
    """Test ``ignore_iris_vague_metadata_warnings``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x_coord = DimCoord([0, 3], bounds=[[-1, 1], [2, 4]], var_name="x")
        cube = Cube([1, 1], dim_coords_and_dims=[(x_coord, 0)])
        with ignore_iris_vague_metadata_warnings():
            cube.collapsed("x", iris.analysis.MEAN)


@pytest.fixture
def dummy_cubes(realistic_4d_cube):
    realistic_4d_cube.remove_ancillary_variable(
        realistic_4d_cube.ancillary_variables()[0],
    )
    realistic_4d_cube.data = realistic_4d_cube.lazy_data()
    return CubeList([realistic_4d_cube])


@pytest.fixture
def empty_cubes():
    return CubeList([Cube(0.0)])


@pytest.mark.parametrize(
    "conversion_func",
    [ncdata.iris.from_iris, ncdata.iris_xarray.cubes_to_xarray],
)
def test_dataset_to_iris(conversion_func, dummy_cubes):
    dataset = conversion_func(dummy_cubes)

    cubes = dataset_to_iris(dataset, filepath="path/to/file.nc")

    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.has_lazy_data()
    np.testing.assert_allclose(cube.data, dummy_cubes[0].data)
    assert cube.standard_name == dummy_cubes[0].standard_name
    assert cube.var_name == dummy_cubes[0].var_name
    assert cube.long_name == dummy_cubes[0].long_name
    assert cube.units == dummy_cubes[0].units
    assert str(cube.coord("latitude").units) == "degrees_north"
    assert str(cube.coord("longitude").units) == "degrees_east"
    assert cube.attributes["source_file"] == "path/to/file.nc"


@pytest.mark.parametrize(
    "conversion_func",
    [ncdata.iris.from_iris, ncdata.iris_xarray.cubes_to_xarray],
)
def test_dataset_to_iris_empty_cube(conversion_func, empty_cubes):
    dataset = conversion_func(empty_cubes)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cubes = dataset_to_iris(dataset)

    assert len(cubes) == 1
    cube = cubes[0]

    assert cube.has_lazy_data()
    np.testing.assert_allclose(cube.data, np.array(0.0))
    assert cube.standard_name is None
    assert cube.var_name == "unknown"
    assert cube.long_name is None
    assert cube.units == "unknown"
    assert not cube.coords()
    assert "source_file" not in cube.attributes


@pytest.mark.parametrize(
    "conversion_func",
    [ncdata.iris.from_iris, ncdata.iris_xarray.cubes_to_xarray],
)
def test_dataset_to_iris_ignore_warnings(conversion_func, dummy_cubes):
    dataset = conversion_func(dummy_cubes)
    if isinstance(dataset, xr.Dataset):
        dataset.ta.attrs["units"] = "invalid_units"
    else:
        dataset.variables["ta"].attributes["units"] = "invalid_units"

    ignore_warnings = [
        {
            "message": (
                r"Not all file objects were parsed correctly. See "
                r"iris.loading.LOAD_PROBLEMS for details."
            ),
            "category": IrisLoadWarning,
            "module": "iris",
        },
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dataset_to_iris(dataset, ignore_warnings=ignore_warnings)


@pytest.mark.parametrize(
    "conversion_func",
    [ncdata.iris.from_iris, ncdata.iris_xarray.cubes_to_xarray],
)
def test_dataset_to_iris_no_ignore_warnings(conversion_func, dummy_cubes):
    dataset = conversion_func(dummy_cubes)
    if isinstance(dataset, xr.Dataset):
        dataset.ta.attrs["cell_methods"] = "time: invalid_method"
    else:
        dataset.variables["ta"].set_attrval(
            "cell_methods",
            "time: invalid_method",
        )

    msg = r"NetCDF variable 'ta' contains unknown cell method 'invalid_method'"
    with pytest.warns(IrisUnknownCellMethodWarning, match=msg):
        dataset_to_iris(dataset)


def test_dataset_to_iris_invalid_type_fail():
    msg = (
        r"Expected type ncdata.NcData or xr.Dataset for dataset, got type "
        r"<class 'int'>"
    )
    with pytest.raises(TypeError, match=msg):
        dataset_to_iris(1)
