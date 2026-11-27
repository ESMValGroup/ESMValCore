"""Integration tests for :func:`esmvalcore.preprocessor.save`."""

import logging
import re

import iris
import netCDF4
import numpy as np
import pytest
import zarr
from dask.delayed import Delayed
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.preprocessor import save


@pytest.fixture
def filename(tmp_path):
    return tmp_path / "test.nc"


@pytest.fixture
def cube():
    lat = DimCoord(
        np.asarray([1, 2], np.single),
        standard_name="latitude",
        units="degrees_north",
    )
    lon = DimCoord(
        np.asarray([1, 2], np.single),
        standard_name="longitude",
        units="degrees_east",
    )
    time = DimCoord(
        np.asarray([1, 2], np.single),
        standard_name="time",
        units="days since 2000-1-1",
    )

    rng = np.random.default_rng(42)
    return Cube(
        rng.random((2, 2, 2)),
        var_name="sample",
        units="1",
        dim_coords_and_dims=((lat, 0), (lon, 1), (time, 2)),
    )


def _compare_cubes(cube, loaded_cube):
    np.testing.assert_equal(cube.data, loaded_cube.data)
    for coord in cube.coords():
        np.testing.assert_equal(
            coord.points,
            loaded_cube.coord(coord.name()).points,
        )


def _check_chunks(path, expected_chunks):
    with netCDF4.Dataset(path, "r") as handler:
        chunking = handler.variables["sample"].chunking()
    assert expected_chunks == chunking


def test_save(cube, filename):
    """Test save."""
    delayed = save([cube], filename)
    assert delayed is None
    loaded_cube = iris.load_cube(filename)
    _compare_cubes(cube, loaded_cube)


def test_delayed_save(cube, filename):
    """Test save."""
    delayed = save([cube], filename, compute=False)
    assert isinstance(delayed, Delayed)
    delayed.compute()
    loaded_cube = iris.load_cube(filename)
    _compare_cubes(cube, loaded_cube)


def test_save_noop(cube, filename, caplog):
    """Test save."""
    cube.data = cube.lazy_data()
    save([cube], filename)
    with caplog.at_level(logging.DEBUG):
        save([cube], filename)
    assert re.findall("Not saving cubes .* to avoid data loss.", caplog.text)


def test_save_create_parent_dir(cube, tmp_path):
    filename = tmp_path / "preproc" / "something" / "test.nc"
    save([cube], filename)
    loaded_cube = iris.load_cube(filename)
    _compare_cubes(cube, loaded_cube)


def test_save_alias(cube, filename):
    """Test save."""
    save([cube], filename, alias="alias")
    loaded_cube = iris.load_cube(filename)
    _compare_cubes(cube, loaded_cube)
    assert loaded_cube.var_name == "alias"


def test_save_zlib(cube, filename):
    """Test save."""
    save([cube], filename, compress=True)
    loaded_cube = iris.load_cube(filename)
    _compare_cubes(cube, loaded_cube)
    with netCDF4.Dataset(filename, "r") as handler:
        sample_filters = handler.variables["sample"].filters()
    assert sample_filters["zlib"] is True
    assert sample_filters["shuffle"] is True
    assert sample_filters["complevel"] == 4


def test_fail_empty_cubes(filename):
    """Test save fails if empty cubes is provided."""
    empty_cubes = CubeList([])
    with pytest.raises(ValueError):
        save(empty_cubes, filename)


def test_fail_without_filename(cube):
    """Test save fails if filename is not provided."""
    with pytest.raises(TypeError):
        save([cube])


def test_save_optimized_map(cube, filename):
    """Test save."""
    save([cube], filename, optimize_access="map")
    loaded_cube = iris.load_cube(filename)
    _compare_cubes(cube, loaded_cube)
    _check_chunks(filename, [2, 2, 1])


def test_save_optimized_timeseries(cube, filename):
    """Test save."""
    save([cube], filename, optimize_access="timeseries")
    loaded_cube = iris.load_cube(filename)
    _compare_cubes(cube, loaded_cube)
    _check_chunks(filename, [1, 1, 2])


def test_save_optimized_lat(cube, filename):
    """Test save."""
    save([cube], filename, optimize_access="latitude")
    loaded_cube = iris.load_cube(filename)
    _compare_cubes(cube, loaded_cube)
    expected_chunks = [2, 1, 1]
    _check_chunks(filename, expected_chunks)


def test_save_optimized_lon_time(cube, filename):
    """Test save."""
    save([cube], filename, optimize_access="longitude time")
    loaded_cube = iris.load_cube(filename)
    _compare_cubes(cube, loaded_cube)
    _check_chunks(filename, [1, 2, 2])


@pytest.fixture
def zarr_filename(tmp_path):
    return tmp_path / "test.zarr"


def _open_zarr_array(path):
    return zarr.open_array(path / "sample", mode="r")


def test_save_zarr(cube, zarr_filename):
    delayed = save([cube], zarr_filename)
    assert delayed is None
    loaded_cube = iris.load_cube(zarr_filename)
    _compare_cubes(cube, loaded_cube)
    array = _open_zarr_array(zarr_filename)
    assert array.chunks == (2, 2, 2)
    assert array.shards is None
    assert array.compressors == ()


def test_delayed_save_zarr(cube, zarr_filename):
    cube.data = cube.lazy_data().rechunk((1, 2, 2))
    delayed = save([cube], zarr_filename, compute=False)
    assert isinstance(delayed, Delayed)
    delayed.compute()
    loaded_cube = iris.load_cube(zarr_filename)
    _compare_cubes(cube, loaded_cube)
    assert _open_zarr_array(zarr_filename).chunks == (1, 2, 2)


def test_save_zarr_overwrite(cube, zarr_filename):
    save([cube], zarr_filename)
    cube.data = cube.data + 1.0
    save([cube], zarr_filename)
    loaded_cube = iris.load_cube(zarr_filename)
    _compare_cubes(cube, loaded_cube)


def test_save_zarr_compress(cube, zarr_filename):
    save([cube], zarr_filename, compress=True)
    loaded_cube = iris.load_cube(zarr_filename)
    _compare_cubes(cube, loaded_cube)
    assert _open_zarr_array(zarr_filename).compressors


def test_save_zarr_optimized_no_shards(cube, zarr_filename):
    save([cube], zarr_filename, optimize_access="map", shards=None)
    loaded_cube = iris.load_cube(zarr_filename)
    _compare_cubes(cube, loaded_cube)
    array = _open_zarr_array(zarr_filename)
    assert array.chunks == (2, 2, 1)
    assert array.shards is None


@pytest.mark.parametrize(
    ("optimize_access", "chunks", "shards"),
    [
        ("map", (2, 2, 1), (2, 2, 2)),
        ("timeseries", (1, 1, 2), (2, 1, 2)),
        ("longitude time", (1, 2, 2), (2, 2, 2)),
    ],
)
def test_save_zarr_optimized(
    cube,
    zarr_filename,
    optimize_access,
    chunks,
    shards,
):
    # Shards are the Dask chunks, rounded up to a whole number of chunks.
    cube.data = cube.lazy_data().rechunk((2, 1, 2))
    save([cube], zarr_filename, optimize_access=optimize_access)
    loaded_cube = iris.load_cube(zarr_filename)
    _compare_cubes(cube, loaded_cube)
    array = _open_zarr_array(zarr_filename)
    assert array.chunks == chunks
    assert array.shards == shards
