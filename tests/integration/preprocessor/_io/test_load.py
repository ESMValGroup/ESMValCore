"""Integration tests for :func:`esmvalcore.preprocessor._io.load`."""

import warnings
from importlib.resources import files as importlib_files
from pathlib import Path
from unittest import mock

import iris
import ncdata
import numpy as np
import pytest
import xarray as xr
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.preprocessor._io import _get_attr_from_field_coord, load


def _create_sample_cube():
    coord = DimCoord([1, 2], standard_name="latitude", units="degrees_north")
    return Cube([1, 2], var_name="sample", dim_coords_and_dims=((coord, 0),))


def test_load(tmp_path):
    """Test loading multiple files."""
    cube = _create_sample_cube()
    temp_file = tmp_path / "cube.nc"
    iris.save(cube, temp_file)

    cubes = load(temp_file)

    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.attributes.globals["source_file"] == str(temp_file)
    np.testing.assert_equal(cube.data, [1, 2])
    np.testing.assert_equal(cube.coord("latitude").points, [1, 2])


def test_load_grib():
    """Test loading a grib file."""
    grib_path = (
        Path(importlib_files("tests"))
        / "sample_data"
        / "iris-sample-data"
        / "polar_stereo.grib2"
    )
    cubes = load(grib_path)

    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.standard_name == "air_temperature"
    assert cube.units == "K"
    assert cube.shape == (200, 247)
    assert "source_file" in cube.attributes


def test_load_cube():
    """Test loading an Iris Cube."""
    cube = _create_sample_cube()
    cubes = load(cube)
    assert cubes == CubeList([cube])


def test_load_cubes():
    """Test loading an Iris CubeList."""
    cube = _create_sample_cube()
    cubes = load(CubeList([cube]))
    assert cubes == CubeList([cube])


def test_load_xarray_dataset(caplog):
    """Test loading an xarray.Dataset."""
    dataset = xr.Dataset(
        data_vars={"tas": ("time", [1, 2])},
        coords={"time": [0, 1]},
        attrs={"test": 1},
    )

    cubes = load(dataset)

    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "tas"
    assert cube.standard_name is None
    assert cube.long_name is None
    assert cube.units == "unknown"
    assert len(cube.coords()) == 1
    assert cube.coords()[0].var_name == "time"
    assert cube.attributes["test"] == 1

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        "does not contain attribute 'source_file'" in caplog.records[0].message
    )


def test_load_ncdata(caplog):
    """Test loading an ncdata.NcData."""
    dataset = ncdata.NcData(
        dimensions=(ncdata.NcDimension("time", 2),),
        variables=(ncdata.NcVariable("tas", ("time",), [0, 1]),),
    )

    cubes = load(dataset)

    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "tas"
    assert cube.standard_name is None
    assert cube.long_name is None
    assert cube.units == "unknown"
    assert not cube.coords()

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        "does not contain attribute 'source_file'" in caplog.records[0].message
    )


def test_load_invalid_type_fail():
    """Test loading an invalid type."""
    with pytest.raises(TypeError):
        load(1)


def test_callback_fix_lat_units(tmp_path):
    """Test callback for fixing units."""
    cube = _create_sample_cube()
    temp_file = tmp_path / "cube.nc"
    iris.save(cube, temp_file)

    cubes = load(temp_file)

    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.attributes.globals["source_file"] == str(temp_file)
    np.testing.assert_equal(cube.data, [1, 2])
    np.testing.assert_equal(cube.coord("latitude").points, [1, 2])
    assert str(cube.coord("latitude").units) == "degrees_north"


def test_get_attr_from_field_coord_none(mocker):
    """Test ``_get_attr_from_field_coord``."""
    attr = _get_attr_from_field_coord(mocker.sentinel.ncfield, None, "attr")
    assert attr is None


@mock.patch("iris.load_raw", autospec=True)
def test_fail_empty_cubes(mock_load_raw):
    """Test that ValueError is raised when cubes are empty."""
    mock_load_raw.return_value = CubeList([])
    msg = "myfilename does not contain any data"
    with pytest.raises(ValueError, match=msg):
        load("myfilename")


def load_with_warning(*_, **__):
    """Mock load with a warning."""
    warnings.warn(
        "This is a custom expected warning",
        category=UserWarning,
        stacklevel=2,
    )
    return CubeList([Cube(0)])


@mock.patch("iris.load_raw", autospec=True)
def test_do_not_ignore_warnings(mock_load_raw):
    """Test do not ignore specific warnings."""
    mock_load_raw.side_effect = load_with_warning
    ignore_warnings = [{"message": "non-relevant warning"}]

    # Check that warnings is raised
    with pytest.warns(UserWarning):
        cubes = load("myfilename", ignore_warnings=ignore_warnings)

    assert len(cubes) == 1
    assert cubes[0].attributes.globals["source_file"] == "myfilename"


@mock.patch("iris.load_raw", autospec=True)
def test_ignore_warnings(mock_load_raw):
    """Test ignore specific warnings."""
    mock_load_raw.side_effect = load_with_warning
    ignore_warnings = [{"message": "This is a custom expected warning"}]

    # Assert that no warning has been raised
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cubes = load("myfilename", ignore_warnings=ignore_warnings)

    assert len(cubes) == 1
    assert cubes[0].attributes.globals["source_file"] == "myfilename"
