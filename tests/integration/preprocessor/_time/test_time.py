"""Unit tests for the :func:`esmvalcore.preprocessor._time` module."""

import dask.array as da
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import (
    AncillaryVariable,
    AuxCoord,
    CellMeasure,
    CellMethod,
    DimCoord,
)
from iris.cube import Cube
from iris.exceptions import CoordinateMultiDimError, CoordinateNotFoundError

from esmvalcore.preprocessor._time import climate_statistics, local_solar_time
from tests import assert_array_equal


@pytest.fixture
def easy_2d_cube():
    """Create easy 2D cube to test statistical operators."""
    time = DimCoord(
        [2.0, 3.0],
        bounds=[[-0.5, 2.5], [2.5, 3.5]],
        standard_name='time',
        units='days since 2000-01-01',
    )
    lat = DimCoord(
        [0.0, 1.0], standard_name='latitude', units='degrees'
    )
    cube = Cube(
        np.arange(4, dtype=np.float32).reshape(2, 2),
        standard_name='air_temperature',
        units='K',
        dim_coords_and_dims=[(time, 0), (lat, 1)],
    )
    return cube


@pytest.mark.parametrize(
    'operator,kwargs,expected_data,expected_units',
    [
        ('gmean', {}, [0.0, 1.7320509], 'K'),
        ('hmean', {}, [0.0, 1.5], 'K'),
        ('max', {}, [2.0, 3.0], 'K'),
        ('mean', {}, [0.5, 1.5], 'K'),
        ('mean', {'weights': False}, [1.0, 2.0], 'K'),
        ('median', {}, [1.0, 2.0], 'K'),
        ('min', {}, [0.0, 1.0], 'K'),
        ('peak', {}, [2.0, 3.0], 'K'),
        ('percentile', {'percent': 0.0}, [0.0, 1.0], 'K'),
        ('rms', {}, [1.0, 1.7320509], 'K'),
        ('rms', {'weights': False}, [1.414214, 2.236068], 'K'),
        ('std_dev', {}, [1.414214, 1.414214], 'K'),
        ('std_dev', {'ddof': 0}, [1.0, 1.0], 'K'),
        ('sum', {}, [2.0, 6.0], 'K day'),
        ('sum', {'weights': False}, [2.0, 4.0], 'K'),
        ('variance', {}, [2.0, 2.0], 'K2'),
        ('variance', {'ddof': 0}, [1.0, 1.0], 'K2'),
        ('wpercentile', {'percent': 50.0}, [0.5, 1.5], 'K'),
    ]
)
def test_statistical_operators(
    operator, kwargs, expected_data, expected_units, easy_2d_cube
):
    """Test ``climate_statistics`` with different operators."""
    res = climate_statistics(easy_2d_cube, operator, **kwargs)

    assert res.var_name == easy_2d_cube.var_name
    assert res.long_name == easy_2d_cube.long_name
    assert res.standard_name == easy_2d_cube.standard_name
    assert res.attributes == easy_2d_cube.attributes
    assert res.units == expected_units
    assert res.coord('latitude') == easy_2d_cube.coord('latitude')
    assert res.coord('time').shape == (1, )
    np.testing.assert_allclose(res.data, expected_data, atol=1e-6, rtol=1e-6)


@pytest.fixture
def realistic_4d_cube():
    """Create realistic 4D cube."""
    time = DimCoord(
        [11.0, 12.0],
        standard_name='time',
        units=Unit('hours since 1851-01-01', calendar='360_day'),
    )
    plev = DimCoord([50000], standard_name='air_pressure', units='Pa')
    lat = DimCoord([0.0, 1.0], standard_name='latitude', units='degrees')
    lon = DimCoord(
        [0.0, 20.0, 345.0], standard_name='longitude', units='degrees'
    )

    aux_2d_data = np.arange(2 * 3).reshape(2, 3)
    aux_2d_bounds = np.stack(
        (aux_2d_data - 1, aux_2d_data, aux_2d_data + 1), axis=-1
    )
    aux_2d = AuxCoord(aux_2d_data, var_name='aux_2d')
    aux_2d_with_bnds = AuxCoord(
        aux_2d_data, bounds=aux_2d_bounds, var_name='aux_2d_with_bnds'
    )
    aux_time = AuxCoord(['Jan', 'Jan'], var_name='aux_time')
    aux_lon = AuxCoord([0, 1, 2], var_name='aux_lon')

    cell_area = CellMeasure(
        np.arange(2 * 2 * 3).reshape(2, 2, 3) + 10,
        standard_name='cell_area',
        units='m2',
        measure='area',
    )
    type_var = AncillaryVariable(
        [['sea', 'land', 'lake'], ['lake', 'sea', 'land']],
        var_name='type',
        units='no_unit',
    )

    cube = Cube(
        np.ma.masked_inside(
            np.arange(2 * 1 * 2 * 3).reshape(2, 1, 2, 3), 1, 3
        ),
        var_name='ta',
        standard_name='air_temperature',
        long_name='Air Temperature',
        units='K',
        cell_methods=[CellMethod('mean', 'time')],
        dim_coords_and_dims=[(time, 0), (plev, 1), (lat, 2), (lon, 3)],
        aux_coords_and_dims=[
            (aux_2d, (0, 3)),
            (aux_2d_with_bnds, (0, 3)),
            (aux_time, 0),
            (aux_lon, 3),
        ],
        cell_measures_and_dims=[(cell_area, (0, 2, 3))],
        ancillary_variables_and_dims=[(type_var, (0, 3))],
        attributes={'test': 1},
    )
    return cube


def test_local_solar_time_regular(realistic_4d_cube):
    """Test ``local_solar_time``."""
    input_cube = realistic_4d_cube.copy()

    result = local_solar_time(input_cube)

    assert input_cube == realistic_4d_cube

    assert result.metadata == input_cube.metadata
    assert result.shape == input_cube.shape
    assert result.coord('time') != input_cube.coord('time')
    assert result.coord('air_pressure') == input_cube.coord('air_pressure')
    assert result.coord('latitude') == input_cube.coord('latitude')
    assert result.coord('longitude') == input_cube.coord('longitude')

    assert result.coord('time').standard_name == 'time'
    assert result.coord('time').var_name is None
    assert result.coord('time').long_name == 'Local Solar Time'
    assert result.coord('time').units == Unit(
        'hours since 1850-01-01', calendar='360_day'
    )
    assert result.coord('time').attributes == {}
    np.testing.assert_allclose(
        result.coord('time').points, [8651.0, 8652.0]
    )
    np.testing.assert_allclose(
        result.coord('time').bounds, [[8650.5, 8651.5], [8651.5, 8652.5]]
    )

    assert result.coord('aux_time') == input_cube.coord('aux_time')
    assert result.coord('aux_lon') == input_cube.coord('aux_lon')
    assert (
        result.coord('aux_2d').metadata == input_cube.coord('aux_2d').metadata
    )
    assert not result.coord('aux_2d').has_lazy_points()
    assert_array_equal(
        result.coord('aux_2d').points,
        np.ma.masked_equal([[0, 99, 5], [3, 1, 99]], 99),
    )
    assert not result.coord('aux_2d').has_bounds()
    assert (
        result.coord('aux_2d_with_bnds').metadata ==
        input_cube.coord('aux_2d_with_bnds').metadata
    )
    assert not result.coord('aux_2d_with_bnds').has_lazy_points()
    assert_array_equal(
        result.coord('aux_2d_with_bnds').points,
        np.ma.masked_equal([[0, 99, 5], [3, 1, 99]], 99),
    )
    assert not result.coord('aux_2d_with_bnds').has_lazy_bounds()
    assert_array_equal(
        result.coord('aux_2d_with_bnds').bounds,
        np.ma.masked_equal(
            [
                [[-1, 0, 1], [99, 99, 99], [4, 5, 6]],
                [[2, 3, 4], [0, 1, 2], [99, 99, 99]],
            ],
            99,
        ),
    )

    assert (
        result.cell_measure('cell_area').metadata ==
        input_cube.cell_measure('cell_area').metadata
    )
    assert not result.cell_measure('cell_area').has_lazy_data()
    assert_array_equal(
        result.cell_measure('cell_area').data,
        np.ma.masked_equal(
            [
                [[10, 99, 18], [13, 99, 21]],
                [[16, 11, 99], [19, 14, 99]],
            ],
            99,
        ),
    )
    assert (
        result.ancillary_variable('type').metadata ==
        input_cube.ancillary_variable('type').metadata
    )
    assert not result.ancillary_variable('type').has_lazy_data()
    assert_array_equal(
        result.ancillary_variable('type').data,
        np.ma.masked_equal(
            [['sea', 'miss', 'land'], ['lake', 'land', 'miss']], 'miss'
        ),
    )

    assert not result.has_lazy_data()
    assert_array_equal(
        result.data,
        np.ma.masked_equal(
            [
                [[[0, 99, 8], [99, 99, 11]]],
                [[[6, 99, 99], [9, 4, 99]]],
            ],
            99,
        ),
    )


def test_local_solar_time_1_time_step(realistic_4d_cube):
    """Test ``local_solar_time``."""
    input_cube = realistic_4d_cube[[0]]

    result = local_solar_time(input_cube)

    assert input_cube == realistic_4d_cube[[0]]

    assert result.metadata == input_cube.metadata
    assert result.shape == input_cube.shape
    assert result.coord('time') != input_cube.coord('time')
    assert result.coord('air_pressure') == input_cube.coord('air_pressure')
    assert result.coord('latitude') == input_cube.coord('latitude')
    assert result.coord('longitude') == input_cube.coord('longitude')

    assert result.coord('time').standard_name == 'time'
    assert result.coord('time').var_name is None
    assert result.coord('time').long_name == 'Local Solar Time'
    assert result.coord('time').units == Unit(
        'hours since 1850-01-01', calendar='360_day'
    )
    assert result.coord('time').attributes == {}
    np.testing.assert_allclose(result.coord('time').points, [8651.0])
    np.testing.assert_allclose(result.coord('time').bounds, [[8650.5, 8651.5]])

    assert result.coord('aux_time') == input_cube.coord('aux_time')
    assert result.coord('aux_lon') == input_cube.coord('aux_lon')
    assert (
        result.coord('aux_2d').metadata == input_cube.coord('aux_2d').metadata
    )
    assert not result.coord('aux_2d').has_lazy_points()
    assert_array_equal(
        result.coord('aux_2d').points, np.ma.masked_equal([[0, 99, 99]], 99)
    )
    assert not result.coord('aux_2d').has_bounds()
    assert (
        result.coord('aux_2d_with_bnds').metadata ==
        input_cube.coord('aux_2d_with_bnds').metadata
    )
    assert not result.coord('aux_2d_with_bnds').has_lazy_points()
    assert_array_equal(
        result.coord('aux_2d_with_bnds').points,
        np.ma.masked_equal([[0, 99, 99]], 99),
    )
    assert not result.coord('aux_2d_with_bnds').has_lazy_bounds()
    assert_array_equal(
        result.coord('aux_2d_with_bnds').bounds,
        np.ma.masked_equal([[[-1, 0, 1], [99, 99, 99], [99, 99, 99]]], 99),
    )

    assert (
        result.cell_measure('cell_area').metadata ==
        input_cube.cell_measure('cell_area').metadata
    )
    assert not result.cell_measure('cell_area').has_lazy_data()
    assert_array_equal(
        result.cell_measure('cell_area').data,
        np.ma.masked_equal([[[10, 99, 99], [13, 99, 99]]], 99),
    )
    assert (
        result.ancillary_variable('type').metadata ==
        input_cube.ancillary_variable('type').metadata
    )
    assert not result.ancillary_variable('type').has_lazy_data()
    assert_array_equal(
        result.ancillary_variable('type').data,
        np.ma.masked_equal([['sea', 'miss', 'miss']], 'miss'),
    )

    assert not result.has_lazy_data()
    assert_array_equal(
        result.data,
        np.ma.masked_equal([[[[0, 99, 99], [99, 99, 99]]]], 99),
    )


@pytest.fixture
def realistic_unstructured_cube():
    """Create realistic unstructured cube."""
    time = DimCoord(
        [0.0, 6.0, 12.0, 18.0, 24.0],
        bounds=[
            [-3.0, 3.0], [3.0, 9.0], [9.0, 15.0], [15.0, 21.0], [21.0, 27.0]
        ],
        var_name='time',
        standard_name='time',
        long_name='time',
        units=Unit('hours since 1851-01-01'),
    )

    lat = AuxCoord(
        [0.0, 0.0, 0.0, 0.0],
        var_name='lat',
        standard_name='latitude',
        long_name='latitude',
        units='degrees_north',
    )
    lon = AuxCoord(
        [0.0, 80 * np.pi / 180.0, -120 * np.pi / 180.0, 160 * np.pi / 180.0],
        var_name='lon',
        standard_name='longitude',
        long_name='longitude',
        units='rad',
    )
    aux_2d_data = da.ma.masked_inside(da.arange(4 * 5).reshape(4, 5), 3, 10)
    aux_2d_bounds = da.stack((aux_2d_data - 1, aux_2d_data + 1), axis=-1)
    aux_2d = AuxCoord(aux_2d_data, var_name='aux_2d')
    aux_2d_with_bnds = AuxCoord(
        aux_2d_data, bounds=aux_2d_bounds, var_name='aux_2d_with_bnds'
    )
    aux_0d = AuxCoord([0], var_name='aux_0d')

    cell_measure_2d = CellMeasure(
        da.ma.masked_inside(da.arange(4 * 5).reshape(4, 5), 3, 10),
        var_name='cell_measure',
    )
    anc_var_2d = AncillaryVariable(
        da.ma.masked_inside(da.arange(4 * 5).reshape(4, 5), 3, 10),
        var_name='anc_var',
    )

    cube = Cube(
        da.arange(4 * 5).reshape(4, 5),
        var_name='ta',
        standard_name='air_temperature',
        long_name='Air Temperature',
        units='K',
        dim_coords_and_dims=[(time, 1)],
        aux_coords_and_dims=[
            (lat, 0),
            (lon, 0),
            (aux_2d, (0, 1)),
            (aux_2d_with_bnds, (0, 1)),
            (aux_0d, ()),
        ],
        cell_measures_and_dims=[(cell_measure_2d, (0, 1))],
        ancillary_variables_and_dims=[(anc_var_2d, (0, 1))],
    )
    return cube


def test_local_solar_time_unstructured(realistic_unstructured_cube):
    """Test ``local_solar_time``."""
    input_cube = realistic_unstructured_cube.copy()

    result = local_solar_time(input_cube)

    assert input_cube == realistic_unstructured_cube

    assert result.metadata == input_cube.metadata
    assert result.shape == input_cube.shape
    assert result.coord('time') != input_cube.coord('time')
    assert result.coord('latitude') == input_cube.coord('latitude')
    assert result.coord('longitude') == input_cube.coord('longitude')

    assert result.coord('time').standard_name == 'time'
    assert result.coord('time').var_name == 'time'
    assert result.coord('time').long_name == 'Local Solar Time'
    assert result.coord('time').units == 'hours since 1850-01-01'
    assert result.coord('time').attributes == {}
    np.testing.assert_allclose(
        result.coord('time').points, [8760.0, 8766.0, 8772.0, 8778.0, 8784.0]
    )
    np.testing.assert_allclose(
        result.coord('time').bounds,
        [
            [8757.0, 8763.0],
            [8763.0, 8769.0],
            [8769.0, 8775.0],
            [8775.0, 8781.0],
            [8781.0, 8787.0],
        ],
    )

    assert result.coord('aux_0d') == input_cube.coord('aux_0d')
    assert (
        result.coord('aux_2d').metadata == input_cube.coord('aux_2d').metadata
    )
    assert result.coord('aux_2d').has_lazy_points()
    assert_array_equal(
        result.coord('aux_2d').points,
        np.ma.masked_equal(
            [
                [0, 1, 2, 99, 99],
                [99, 99, 99, 99, 99],
                [11, 12, 13, 14, 99],
                [99, 99, 15, 16, 17],
            ],
            99,
        ),
    )
    assert not result.coord('aux_2d').has_bounds()
    assert (
        result.coord('aux_2d_with_bnds').metadata ==
        input_cube.coord('aux_2d_with_bnds').metadata
    )
    assert result.coord('aux_2d_with_bnds').has_lazy_points()
    assert_array_equal(
        result.coord('aux_2d_with_bnds').points,
        np.ma.masked_equal(
            [
                [0, 1, 2, 99, 99],
                [99, 99, 99, 99, 99],
                [11, 12, 13, 14, 99],
                [99, 99, 15, 16, 17],
            ],
            99,
        ),
    )
    assert result.coord('aux_2d_with_bnds').has_lazy_bounds()
    assert_array_equal(
        result.coord('aux_2d_with_bnds').bounds,
        np.ma.masked_equal(
            [
                [[-1, 1], [0, 2], [1, 3], [99, 99], [99, 99]],
                [[99, 99], [99, 99], [99, 99], [99, 99], [99, 99]],
                [[10, 12], [11, 13], [12, 14], [13, 15], [99, 99]],
                [[99, 99], [99, 99], [14, 16], [15, 17], [16, 18]],
            ],
            99,
        ),
    )

    assert (
        result.cell_measure('cell_measure').metadata ==
        input_cube.cell_measure('cell_measure').metadata
    )
    assert result.cell_measure('cell_measure').has_lazy_data()
    assert_array_equal(
        result.cell_measure('cell_measure').data,
        np.ma.masked_equal(
            [
                [0, 1, 2, 99, 99],
                [99, 99, 99, 99, 99],
                [11, 12, 13, 14, 99],
                [99, 99, 15, 16, 17],
            ],
            99,
        ),
    )
    assert (
        result.ancillary_variable('anc_var').metadata ==
        input_cube.ancillary_variable('anc_var').metadata
    )
    assert result.ancillary_variable('anc_var').has_lazy_data()
    assert_array_equal(
        result.ancillary_variable('anc_var').data,
        np.ma.masked_equal(
            [
                [0, 1, 2, 99, 99],
                [99, 99, 99, 99, 99],
                [11, 12, 13, 14, 99],
                [99, 99, 15, 16, 17],
            ],
            99,
        ),
    )

    assert result.has_lazy_data()
    assert_array_equal(
        result.data,
        np.ma.masked_equal(
            [
                [0, 1, 2, 3, 4],
                [99, 5, 6, 7, 8],
                [11, 12, 13, 14, 99],
                [99, 99, 15, 16, 17],
            ],
            99,
        ),
    )


def test_local_solar_time_no_time_fail(realistic_4d_cube):
    """Test ``local_solar_time``."""
    realistic_4d_cube.remove_coord('time')
    msg = 'needs a dimensional coordinate `time`'
    with pytest.raises(CoordinateNotFoundError, match=msg):
        local_solar_time(realistic_4d_cube)


def test_local_solar_time_scalar_time_fail(realistic_4d_cube):
    """Test ``local_solar_time``."""
    input_cube = realistic_4d_cube[0]
    msg = 'needs a dimensional coordinate `time`'
    with pytest.raises(CoordinateNotFoundError, match=msg):
        local_solar_time(input_cube)


def test_local_solar_time_time_decreasing_fail(realistic_4d_cube):
    """Test ``local_solar_time``."""
    input_cube = realistic_4d_cube[::-1]
    msg = '`time` coordinate must be monotonically increasing'
    with pytest.raises(ValueError, match=msg):
        local_solar_time(input_cube)


def test_local_solar_time_no_lon_fail(realistic_4d_cube):
    """Test ``local_solar_time``."""
    realistic_4d_cube.remove_coord('longitude')
    msg = 'needs a coordinate `longitude`'
    with pytest.raises(CoordinateNotFoundError, match=msg):
        local_solar_time(realistic_4d_cube)


def test_local_solar_time_scalar_lon_fail(realistic_4d_cube):
    """Test ``local_solar_time``."""
    input_cube = realistic_4d_cube[..., 0]
    msg = 'needs a 1D coordinate `longitude`, got 0D'
    with pytest.raises(CoordinateMultiDimError, match=msg):
        local_solar_time(input_cube)


def test_local_solar_time_2d_lon_fail(easy_2d_cube):
    """Test ``local_solar_time``."""
    lon_coord = AuxCoord(easy_2d_cube.data, standard_name='longitude')
    easy_2d_cube.add_aux_coord(lon_coord, (0, 1))
    msg = 'needs a 1D coordinate `longitude`, got 2D'
    with pytest.raises(CoordinateMultiDimError, match=msg):
        local_solar_time(easy_2d_cube)
