"""Unit tests for the :func:`esmvalcore.preprocessor.regrid.regrid`
function."""
import dask
import dask.array as da
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._regrid
from esmvalcore.preprocessor import regrid
from esmvalcore.preprocessor._regrid import (
    _get_regridder,
    _horizontal_grid_is_close,
    _rechunk,
)


@pytest.fixture(autouse=True)
def clear_regridder_cache(monkeypatch):
    """Clear regridder cache before test runs."""
    monkeypatch.setattr(
        esmvalcore.preprocessor._regrid, '_CACHED_REGRIDDERS', {}
    )


def _make_coord(start: float, stop: float, step: int, *, name: str):
    """Helper function for creating a coord."""
    coord = iris.coords.DimCoord(
        np.linspace(start, stop, step),
        standard_name=name,
        units='degrees',
    )
    coord.guess_bounds()
    return coord


def _make_cube(*, lat: tuple, lon: tuple):
    """Helper function for creating a cube."""
    lat_coord = _make_coord(*lat, name='latitude')
    lon_coord = _make_coord(*lon, name='longitude')

    return iris.cube.Cube(
        np.zeros(
            [len(lat_coord.points), len(lon_coord.points)], dtype=np.float32
        ),
        dim_coords_and_dims=[(lat_coord, 0), (lon_coord, 1)],
    )


# 10x10
LAT_SPEC1 = (-85, 85, 18)
LON_SPEC1 = (5, 355, 36)

# almost 10x10, but different shape
LAT_SPEC2 = (-85, 85, 17)
LON_SPEC2 = (5, 355, 35)

# 10x10, but different coords
LAT_SPEC3 = (-90, 90, 18)
LON_SPEC3 = (0, 360, 36)

# 30x30
LAT_SPEC4 = (-75, 75, 30)
LON_SPEC4 = (15, 345, 30)


@pytest.fixture
def cube_10x10():
    """Test cube."""
    return _make_cube(lat=LAT_SPEC1, lon=LON_SPEC1)


@pytest.fixture
def cube_30x30():
    """Test cube."""
    return _make_cube(lat=LAT_SPEC4, lon=LON_SPEC4)


SCHEMES = ['area_weighted', 'linear', 'nearest']


@pytest.mark.parametrize('cache_weights', [True, False])
@pytest.mark.parametrize('scheme', SCHEMES)
def test_builtin_regridding(scheme, cache_weights, cube_10x10, cube_30x30):
    """Test `regrid.`"""
    _cached_regridders = esmvalcore.preprocessor._regrid._CACHED_REGRIDDERS
    assert _cached_regridders == {}

    res = regrid(cube_10x10, cube_30x30, scheme, cache_weights=cache_weights)

    assert res.coord('latitude') == cube_30x30.coord('latitude')
    assert res.coord('longitude') == cube_30x30.coord('longitude')
    assert res.dtype == np.float32
    assert np.allclose(res.data, 0.0)

    if cache_weights:
        assert len(_cached_regridders) == 1
        key = (scheme, (18,), (36,), (30,), (30,))
        assert key in _cached_regridders
    else:
        assert not _cached_regridders


@pytest.mark.parametrize('scheme', SCHEMES)
def test_invalid_target_grid(scheme, cube_10x10, mocker):
    """Test `regrid.`"""
    target_grid = mocker.sentinel.target_grid
    msg = "Expecting a cube"
    with pytest.raises(ValueError, match=msg):
        regrid(cube_10x10, target_grid, scheme)


def test_invalid_scheme(cube_10x10, cube_30x30):
    """Test `regrid.`"""
    msg = "Got invalid regridding scheme string 'wibble'"
    with pytest.raises(ValueError, match=msg):
        regrid(cube_10x10, cube_30x30, 'wibble')


def test_regrid_generic_missing_reference(cube_10x10, cube_30x30):
    """Test `regrid.`"""
    msg = "No reference specified for generic regridding."
    with pytest.raises(ValueError, match=msg):
        regrid(cube_10x10, cube_30x30, {})


def test_regrid_generic_invalid_reference(cube_10x10, cube_30x30):
    """Test `regrid.`"""
    msg = "Could not import specified generic regridding module."
    with pytest.raises(ValueError, match=msg):
        regrid(cube_10x10, cube_30x30, {'reference': 'this.does:not.exist'})


@pytest.mark.parametrize('cache_weights', [True, False])
def test_regrid_generic_regridding(cache_weights, cube_10x10, cube_30x30):
    """Test `regrid.`"""
    _cached_regridders = esmvalcore.preprocessor._regrid._CACHED_REGRIDDERS
    assert _cached_regridders == {}

    cube_gen = regrid(
        cube_10x10,
        cube_30x30,
        {
            'reference': 'iris.analysis:Linear',
            'extrapolation_mode': 'mask',
        },
        cache_weights=cache_weights,
    )
    cube_lin = regrid(
        cube_10x10, cube_30x30, 'linear', cache_weights=cache_weights
    )
    assert cube_gen.dtype == np.float32
    assert cube_lin.dtype == np.float32
    assert cube_gen == cube_lin

    if cache_weights:
        assert len(_cached_regridders) == 2
        key_1 = (
            "{'reference': 'iris.analysis:Linear', 'extrapolation_mode': "
            "'mask'}",
            (18,),
            (36,),
            (30,),
            (30,),
        )
        key_2 = ('linear', (18,), (36,), (30,), (30,))
        assert key_1 in _cached_regridders
        assert key_2 in _cached_regridders
    else:
        assert not _cached_regridders


@pytest.mark.parametrize(
    'cube2_spec, expected',
    (
        # equal lat/lon
        (
            {
                'lat': LAT_SPEC1,
                'lon': LON_SPEC1,
            },
            True,
        ),
        # different lon shape
        (
            {
                'lat': LAT_SPEC1,
                'lon': LON_SPEC2,
            },
            False,
        ),
        # different lat shape
        (
            {
                'lat': LAT_SPEC2,
                'lon': LON_SPEC1,
            },
            False,
        ),
        # different lon values
        (
            {
                'lat': LAT_SPEC1,
                'lon': LON_SPEC3,
            },
            False,
        ),
        # different lat values
        (
            {
                'lat': LAT_SPEC3,
                'lon': LON_SPEC1,
            },
            False,
        ),
    ),
)
def test_horizontal_grid_is_close(cube2_spec: dict, expected: bool):
    """Test for `_horizontal_grid_is_close`."""
    cube1 = _make_cube(lat=LAT_SPEC1, lon=LON_SPEC1)
    cube2 = _make_cube(**cube2_spec)

    assert _horizontal_grid_is_close(cube1, cube2) == expected


def test_regrid_is_skipped_if_grids_are_the_same():
    """Test that regridding is skipped if the grids are the same."""
    cube = _make_cube(lat=LAT_SPEC1, lon=LON_SPEC1)
    scheme = 'linear'

    # regridding to the same spec returns the same cube
    expected_same_cube = regrid(cube, target_grid='10x10', scheme=scheme)
    assert expected_same_cube is cube

    # regridding to a different spec returns a different cube
    expected_different_cube = regrid(cube, target_grid='5x5', scheme=scheme)
    assert expected_different_cube is not cube


def make_test_cube(shape):
    data = da.empty(shape, dtype=np.float32)
    cube = iris.cube.Cube(data)
    if len(shape) > 2:
        cube.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(shape[0]),
                standard_name='time',
            ),
            0,
        )
    cube.add_dim_coord(
        iris.coords.DimCoord(
            np.linspace(-90., 90., shape[-2], endpoint=True),
            standard_name='latitude',
        ),
        len(shape) - 2,
    )
    cube.add_dim_coord(
        iris.coords.DimCoord(
            np.linspace(0., 360., shape[-1]),
            standard_name='longitude',
        ),
        len(shape) - 1,
    )
    return cube


def test_rechunk_on_increased_grid():
    """Test that an increase in grid size rechunks."""
    with dask.config.set({'array.chunk-size': '128 M'}):

        time_dim = 246
        src_grid_dims = (91, 180)
        data = da.empty((time_dim, ) + src_grid_dims, dtype=np.float32)

        tgt_grid_dims = (2, 361, 720)
        tgt_grid = make_test_cube(tgt_grid_dims)
        result = _rechunk(iris.cube.Cube(data), tgt_grid)

        assert result.core_data().chunks == ((123, 123), (91, ), (180, ))


def test_no_rechunk_on_decreased_grid():
    """Test that a decrease in grid size does not rechunk."""
    with dask.config.set({'array.chunk-size': '128 M'}):

        time_dim = 200
        src_grid_dims = (361, 720)
        data = da.empty((time_dim, ) + src_grid_dims, dtype=np.float32)

        tgt_grid_dims = (91, 180)
        tgt_grid = make_test_cube(tgt_grid_dims)

        result = _rechunk(iris.cube.Cube(data), tgt_grid)

        assert result.core_data().chunks == data.chunks


def test_no_rechunk_2d():
    """Test that a 2D cube is not rechunked."""
    with dask.config.set({'array.chunk-size': '64 MiB'}):

        src_grid_dims = (361, 720)
        data = da.empty(src_grid_dims, dtype=np.float32)

        tgt_grid_dims = (3601, 7200)
        tgt_grid = da.empty(tgt_grid_dims, dtype=np.float32)

        result = _rechunk(iris.cube.Cube(data), iris.cube.Cube(tgt_grid))

        assert result.core_data().chunks == data.chunks


def test_no_rechunk_non_lazy():
    """Test that a cube with non-lazy data does not crash."""
    cube = iris.cube.Cube(np.arange(2 * 4).reshape([1, 2, 4]))
    tgt_cube = iris.cube.Cube(np.arange(4 * 8).reshape([4, 8]))
    result = _rechunk(cube, tgt_cube)
    assert result.data is cube.data


def test_no_rechunk_unsupported_grid():
    """Test that 2D target coordinates are ignored.

    Because they are not supported at the moment. This could be
    implemented at a later stage if needed.
    """
    cube = iris.cube.Cube(da.arange(2 * 4).reshape([1, 2, 4]))
    tgt_grid_dims = (5, 10)
    tgt_data = da.empty(tgt_grid_dims, dtype=np.float32)
    tgt_grid = iris.cube.Cube(tgt_data)
    lat_points = np.linspace(-90., 90., tgt_grid_dims[0], endpoint=True)
    lon_points = np.linspace(0., 360., tgt_grid_dims[1])

    tgt_grid.add_aux_coord(
        iris.coords.AuxCoord(
            np.broadcast_to(lat_points.reshape(-1, 1), tgt_grid_dims),
            standard_name='latitude',
        ),
        (0, 1),
    )
    tgt_grid.add_aux_coord(
        iris.coords.AuxCoord(
            np.broadcast_to(lon_points.reshape(1, -1), tgt_grid_dims),
            standard_name='longitude',
        ),
        (0, 1),
    )

    expected_chunks = cube.core_data().chunks
    result = _rechunk(cube, tgt_grid)
    assert result is cube
    assert result.core_data().chunks == expected_chunks


@pytest.mark.parametrize('scheme', SCHEMES)
def test_regridding_weights_use_cache(scheme, cube_10x10, cube_30x30, mocker):
    """Test `regrid.`"""
    _cached_regridders = esmvalcore.preprocessor._regrid._CACHED_REGRIDDERS
    assert _cached_regridders == {}

    src_lat = cube_10x10.coord('latitude')
    src_lon = cube_10x10.coord('longitude')
    tgt_lat = cube_30x30.coord('latitude')
    tgt_lon = cube_30x30.coord('longitude')
    key = (scheme, (18,), (36,), (30,), (30,))
    _cached_regridders[key] = {}
    _cached_regridders[key][(src_lat, src_lon, tgt_lat, tgt_lon)] = (
        mocker.sentinel.regridder
    )
    mock_load_scheme = mocker.patch.object(
        esmvalcore.preprocessor._regrid, '_load_scheme', autospec=True
    )

    reg = _get_regridder(cube_10x10, cube_30x30, scheme, cache_weights=True)

    assert reg == mocker.sentinel.regridder

    assert len(_cached_regridders) == 1
    assert key in _cached_regridders

    mock_load_scheme.assert_not_called()


def test_clear_regridding_weights_cache():
    """Test `regrid.cache_clear().`"""
    _cached_regridders = esmvalcore.preprocessor._regrid._CACHED_REGRIDDERS
    _cached_regridders['test'] = 'test'

    regrid.cache_clear()

    assert _cached_regridders == {}
