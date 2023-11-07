"""Integration tests for :func:`esmvalcore.preprocessor.save`"""
import iris
import netCDF4
import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.preprocessor import save


@pytest.fixture
def filename(tmp_path):
    return tmp_path / 'test.nc'


@pytest.fixture
def cube():
    lat = DimCoord(
        np.asarray([1, 2], np.single),
        standard_name='latitude',
        units='degrees_north',
    )
    lon = DimCoord(
        np.asarray([1, 2], np.single),
        standard_name='longitude',
        units='degrees_east',
    )
    time = DimCoord(
        np.asarray([1, 2], np.single),
        standard_name='time',
        units='days since 2000-1-1',
    )

    cube = Cube(
        np.random.random_sample([2, 2, 2]),
        var_name='sample',
        units='1',
        dim_coords_and_dims=((lat, 0), (lon, 1), (time, 2)),
    )

    return cube


def _compare_cubes(cube, loaded_cube):
    np.testing.assert_equal(cube.data, loaded_cube.data)
    for coord in cube.coords():
        np.testing.assert_equal(coord.points,
                                loaded_cube.coord(coord.name()).points)


def _check_chunks(path, expected_chunks):
    with netCDF4.Dataset(path, 'r') as handler:
        chunking = handler.variables['sample'].chunking()
    assert expected_chunks == chunking


def test_save(cube, filename):
    """Test save."""
    path = save([cube], filename)
    loaded_cube = iris.load_cube(path)
    _compare_cubes(cube, loaded_cube)


def test_save_create_parent_dir(cube, tmp_path):
    filename = tmp_path / 'preproc' / 'something' / 'test.nc'
    path = save([cube], filename)
    loaded_cube = iris.load_cube(path)
    _compare_cubes(cube, loaded_cube)


def test_save_alias(cube, filename):
    """Test save."""
    path = save([cube], filename, alias='alias')
    loaded_cube = iris.load_cube(path)
    _compare_cubes(cube, loaded_cube)
    assert loaded_cube.var_name == 'alias'


def test_save_zlib(cube, filename):
    """Test save."""
    path = save([cube], filename, compress=True)
    loaded_cube = iris.load_cube(path)
    _compare_cubes(cube, loaded_cube)
    with netCDF4.Dataset(path, 'r') as handler:
        sample_filters = handler.variables['sample'].filters()
    assert sample_filters['zlib'] is True
    assert sample_filters['shuffle'] is True
    assert sample_filters['complevel'] == 4


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
    path = save([cube], filename, optimize_access='map')
    loaded_cube = iris.load_cube(path)
    _compare_cubes(cube, loaded_cube)
    _check_chunks(path, [2, 2, 1])


def test_save_optimized_timeseries(cube, filename):
    """Test save."""
    path = save([cube], filename, optimize_access='timeseries')
    loaded_cube = iris.load_cube(path)
    _compare_cubes(cube, loaded_cube)
    _check_chunks(path, [1, 1, 2])


def test_save_optimized_lat(cube, filename):
    """Test save."""
    path = save([cube], filename, optimize_access='latitude')
    loaded_cube = iris.load_cube(path)
    _compare_cubes(cube, loaded_cube)
    expected_chunks = [2, 1, 1]
    _check_chunks(path, expected_chunks)


def test_save_optimized_lon_time(cube, filename):
    """Test save."""
    path = save([cube], filename, optimize_access='longitude time')
    loaded_cube = iris.load_cube(path)
    _compare_cubes(cube, loaded_cube)
    _check_chunks(path, [1, 2, 2])
