"""Integration tests for :func:`esmvalcore.preprocessor._io.load`."""

import warnings
from importlib.resources import files as importlib_files
from pathlib import Path

import cf_units
import iris
import ncdata
import numpy as np
import pytest
import xarray as xr
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.exceptions import ESMValCoreLoadWarning
from esmvalcore.preprocessor._io import _get_attr_from_field_coord, load
from tests import assert_array_equal


@pytest.fixture
def sample_cube():
    coord = DimCoord([1, 2], standard_name="latitude", units="degrees_north")
    return Cube([1, 2], var_name="sample", dim_coords_and_dims=((coord, 0),))


def test_load(tmp_path, sample_cube):
    """Test loading multiple files."""
    temp_file = tmp_path / "cube.nc"
    iris.save(sample_cube, temp_file)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cubes = load(temp_file)

    assert len(cubes) == 1
    sample_cube = cubes[0]
    assert sample_cube.attributes.globals["source_file"] == str(temp_file)
    assert_array_equal(sample_cube.data, np.ma.array([1, 2]))
    assert_array_equal(sample_cube.coord("latitude").points, [1, 2])


def test_load_grib():
    """Test loading a grib file."""
    grib_path = (
        Path(importlib_files("tests"))
        / "sample_data"
        / "iris-sample-data"
        / "polar_stereo.grib2"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cubes = load(grib_path)

    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.standard_name == "air_temperature"
    assert cube.units == "K"
    assert cube.shape == (200, 247)
    assert "source_file" in cube.attributes


def test_load_cube(sample_cube):
    """Test loading an Iris Cube."""
    sample_cube.attributes.globals["source_file"] = "path/to/file.nc"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cubes = load(sample_cube)
    assert cubes == CubeList([sample_cube])


def test_load_cubes(sample_cube):
    """Test loading an Iris CubeList."""
    msg = "does not contain attribute 'source_file'"
    with pytest.warns(ESMValCoreLoadWarning, match=msg):
        cubes = load(CubeList([sample_cube]))
    assert cubes == CubeList([sample_cube])


def test_load_xarray_dataset():
    """Test loading an xarray.Dataset."""
    dataset = xr.Dataset(
        data_vars={"tas": ("time", [1, 2])},
        coords={"time": [0, 1]},
        attrs={"test": 1},
    )

    msg = "does not contain attribute 'source_file'"
    with pytest.warns(ESMValCoreLoadWarning, match=msg):
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


def test_load_ncdata():
    """Test loading an ncdata.NcData."""
    dataset = ncdata.NcData(
        dimensions=(ncdata.NcDimension("time", 2),),
        variables=(ncdata.NcVariable("tas", ("time",), [0, 1]),),
    )

    msg = "does not contain attribute 'source_file'"
    with pytest.warns(ESMValCoreLoadWarning, match=msg):
        cubes = load(dataset)

    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "tas"
    assert cube.standard_name is None
    assert cube.long_name is None
    assert cube.units == "unknown"
    assert not cube.coords()


def test_load_zarr_local():
    """Test loading a Zarr store as ncdata.NcData via Xarray."""
    zarr_path = (
        Path(importlib_files("tests"))
        / "sample_data"
        / "zarr-sample-data"
        / "example_field_0.zarr2"
    )

    cubes = load(zarr_path)

    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "q"
    assert cube.standard_name == "specific_humidity"
    assert cube.long_name is None
    assert cube.units == cf_units.Unit("1")
    coords = cube.coords()
    coord_names = [coord.standard_name for coord in coords]
    assert "longitude" in coord_names
    assert "latitude" in coord_names


def test_load_zarr_remote_simdata_bucket():
    """
    Test loading a Zarr store from a https Object Store.

    This is a good example of a production test dataset; on HEALPIX grid
    so not much we can do with it for now, but it's actual model data.
    """
    zarr_path = (
        "https://hackathon-o.s3-ext.jc.rl.ac.uk/sim-data/dev/v5/"
        "glm.n2560_RAL3p3/um.PT1H.hp_z2.zarr"
    )
    ignore_warnings = [
        {"message": "does not contain attribute"},
        {"message": "Missing CF-netCDF grid mapping variable"},
    ]

    cubes = load(zarr_path, ignore_warnings=ignore_warnings)
    for cube in cubes:
        if cube.standard_name == "air_temperature":
            coords = cube.coords()
            coord_names = [coord.standard_name for coord in coords]
            assert "time" in coord_names


def test_load_zarr_remote_simdata_permanent_test_bucket():
    """
    Test loading a Zarr store from a https Object Store.

    This is a good example of a production test dataset; on HEALPIX grid
    so not much we can do with it for now, but it's actual model data.
    Same file as test before ``test_load_zarr_remote_simdata_bucket``
    only this time the Zarr file has been moved to CEDA test bucket; that
    operation uses MinIO:

    mc cp -r pier/sim-data/dev/v5/glm.n2560_RAL3p3/um.PT1H.hp_z2.zarr .
    mc cp -r um.PT1H.hp_z2.zarr/ bryan/esmvaltool-zarr

    where ``pier`` is the remote bucket and ``bryan`` is the test bucket.
    """
    zarr_path = (
        "https://uor-aces-o.s3-ext.jc.rl.ac.uk/"
        "esmvaltool-zarr/um.PT1H.hp_z2.zarr"
    )
    ignore_warnings = [
        {"message": "does not contain attribute"},
        {"message": "Missing CF-netCDF grid mapping variable"},
    ]
    cubes = load(zarr_path, ignore_warnings=ignore_warnings)
    for cube in cubes:
        if cube.standard_name == "air_temperature":
            coords = cube.coords()
            coord_names = [coord.standard_name for coord in coords]
            assert "time" in coord_names


def test_load_zarr_remote_permanent_test_bucket():
    """
    Test loading a Zarr store from a https Object Store.

    We have a permanent bucket: esmvaltool-zarr at CEDA's object store
    "url": "https://uor-aces-o.s3-ext.jc.rl.ac.uk",
    where will host a number of test files.
    """
    zarr_path = (
        "https://uor-aces-o.s3-ext.jc.rl.ac.uk/"
        "esmvaltool-zarr/example_field_0.zarr2"
    )

    # with "dummy" storage options
    cubes = load(
        zarr_path,
        ignore_warnings=None,
        backend_kwargs={"storage_options": {}},
    )

    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "q"
    assert cube.standard_name == "specific_humidity"
    assert cube.long_name is None
    assert cube.units == cf_units.Unit("1")
    coords = cube.coords()
    coord_names = [coord.standard_name for coord in coords]
    assert "longitude" in coord_names
    assert "latitude" in coord_names

    # without storage_options
    cubes = load(zarr_path)

    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "q"
    assert cube.standard_name == "specific_humidity"
    assert cube.long_name is None
    assert cube.units == cf_units.Unit("1")
    coords = cube.coords()
    coord_names = [coord.standard_name for coord in coords]
    assert "longitude" in coord_names
    assert "latitude" in coord_names


def test_load_invalid_type_fail():
    """Test loading an invalid type."""
    with pytest.raises(TypeError):
        load(1)


def test_callback_fix_lat_units(tmp_path, sample_cube):
    """Test callback for fixing units."""
    temp_file = tmp_path / "cube.nc"
    iris.save(sample_cube, temp_file)

    cubes = load(temp_file)

    assert len(cubes) == 1
    sample_cube = cubes[0]
    assert sample_cube.attributes.globals["source_file"] == str(temp_file)
    assert_array_equal(sample_cube.data, np.ma.array([1, 2]))
    assert_array_equal(sample_cube.coord("latitude").points, [1, 2])
    assert str(sample_cube.coord("latitude").units) == "degrees_north"


def test_get_attr_from_field_coord_none(mocker):
    """Test ``_get_attr_from_field_coord``."""
    attr = _get_attr_from_field_coord(mocker.sentinel.ncfield, None, "attr")
    assert attr is None


def test_fail_empty_cubes(mocker):
    """Test that ValueError is raised when cubes are empty."""
    mocker.patch("iris.load_raw", autospec=True, return_value=CubeList([]))
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


def test_do_not_ignore_warnings(mocker):
    """Test do not ignore specific warnings."""
    mocker.patch("iris.load_raw", autospec=True, side_effect=load_with_warning)
    ignore_warnings = [{"message": "non-relevant warning"}]

    # Check that warnings is raised
    with pytest.warns(UserWarning, match="This is a custom expected warning"):
        cubes = load("myfilename", ignore_warnings=ignore_warnings)

    assert len(cubes) == 1
    assert cubes[0].attributes.globals["source_file"] == "myfilename"


def test_ignore_warnings(mocker):
    """Test ignore specific warnings."""
    mocker.patch("iris.load_raw", autospec=True, side_effect=load_with_warning)
    ignore_warnings = [{"message": "This is a custom expected warning"}]

    # Assert that no warning has been raised
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cubes = load("myfilename", ignore_warnings=ignore_warnings)

    assert len(cubes) == 1
    assert cubes[0].attributes.globals["source_file"] == "myfilename"
