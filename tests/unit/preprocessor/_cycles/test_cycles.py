"""Unit tests for :mod:`esmvalcore.preprocessor._cycles`."""
import iris
import iris.coord_categorisation
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.preprocessor._cycles import amplitude


@pytest.fixture
def annual_cycle_cube():
    """Cube including annual cycle."""
    time_units = Unit('days since 1850-01-01 00:00:00', calendar='noleap')
    n_times = 3 * 365
    n_lat = 4
    time_coord = iris.coords.DimCoord(
        np.arange(n_times, dtype=np.float64), var_name='time',
        standard_name='time', long_name='time', units=time_units)
    time_coord.guess_bounds()
    lat_coord = iris.coords.DimCoord(
        np.arange(n_lat, dtype=np.float64) * 10, var_name='lat',
        standard_name='latitude', long_name='latitude', units='degrees')
    lat_coord.guess_bounds()
    new_data = (np.sin(np.arange(n_times) * 2.0 * np.pi / 365.0) *
                (np.arange(n_times) + 1.0) * 0.005 + 0.005 *
                np.arange(n_times)).reshape(n_times, 1) * np.arange(n_lat)
    annual_cycle_cube = iris.cube.Cube(
        new_data, var_name='tas', standard_name='air_temperature',
        units='K', dim_coords_and_dims=[(time_coord, 0), (lat_coord, 1)])
    return annual_cycle_cube


def test_amplitude_fail_wrong_coord(annual_cycle_cube):
    """Test amplitude calculation when wrong coordinate is given."""
    with pytest.raises(iris.exceptions.CoordinateNotFoundError):
        amplitude(annual_cycle_cube, ['year', 'invalid_coord'])


ANNUAL_CYCLE_AMPLITUDE = [
    [0.0, 1.79357289, 3.58714578, 5.38071868],
    [0.0, 4.64430872, 9.28861744, 13.93292616],
    [0.0, 8.26307683, 16.52615367, 24.7892305],
]


def test_amplitude_annual_cycle_add_year(annual_cycle_cube):
    """Test amplitude of annual cycle when year is not given in cube."""
    assert not annual_cycle_cube.coords('year')
    amplitude_cube = amplitude(annual_cycle_cube, 'year')
    assert amplitude_cube.shape == (3, 4)
    assert amplitude_cube.coords('year')
    np.testing.assert_allclose(amplitude_cube.data, ANNUAL_CYCLE_AMPLITUDE)
    assert amplitude_cube.metadata == annual_cycle_cube.metadata


def test_amplitude_annual_cycle_do_not_add_year(annual_cycle_cube):
    """Test amplitude of annual cycle when year is given in cube."""
    assert not annual_cycle_cube.coords('year')
    iris.coord_categorisation.add_year(annual_cycle_cube, 'time')
    amplitude_cube = amplitude(annual_cycle_cube, 'year')
    assert amplitude_cube.shape == (3, 4)
    assert amplitude_cube.coords('year')
    np.testing.assert_allclose(amplitude_cube.data, ANNUAL_CYCLE_AMPLITUDE)
    assert amplitude_cube.metadata == annual_cycle_cube.metadata


@pytest.fixture
def diurnal_cycle_cube():
    """Cube including diurnal cycle."""
    time_units = Unit('hours since 1850-01-01 00:00:00', calendar='noleap')
    n_days = 2 * 365
    n_times = n_days * 4
    time_coord = iris.coords.DimCoord(
        np.arange(n_times, dtype=np.float64) * 6.0, var_name='time',
        standard_name='time', long_name='time', units=time_units)
    time_coord.guess_bounds()
    new_data = np.concatenate((
        [-2.0, -3.0, 0.0, 1.0] * int(n_days / 2),
        [-5.0, -1.0, 5.0, 0.0] * int(n_days / 2),
    ), axis=None)
    diurnal_cycle_cube = iris.cube.Cube(
        new_data, var_name='tas', standard_name='air_temperature',
        units='K', dim_coords_and_dims=[(time_coord, 0)])
    return diurnal_cycle_cube


DIURNAL_CYCLE_AMPLITUDE = [4.0] * 365 + [10.0] * 365


def test_amplitude_diurnal_cycle_add_coords(diurnal_cycle_cube):
    """Test amplitude of diurnal cycle when coords are not given in cube."""
    assert not diurnal_cycle_cube.coords('day_of_year')
    assert not diurnal_cycle_cube.coords('year')
    amplitude_cube = amplitude(diurnal_cycle_cube, ['day_of_year', 'year'])
    assert amplitude_cube.shape == (730,)
    assert amplitude_cube.coords('day_of_year')
    assert amplitude_cube.coords('year')
    np.testing.assert_allclose(amplitude_cube.data, DIURNAL_CYCLE_AMPLITUDE)
    assert amplitude_cube.metadata == diurnal_cycle_cube.metadata


def test_amplitude_diurnal_cycle_do_not_add_coords(diurnal_cycle_cube):
    """Test amplitude of diurnal cycle when coords are given in cube."""
    assert not diurnal_cycle_cube.coords('day_of_year')
    assert not diurnal_cycle_cube.coords('year')
    iris.coord_categorisation.add_day_of_year(diurnal_cycle_cube, 'time')
    iris.coord_categorisation.add_year(diurnal_cycle_cube, 'time')
    amplitude_cube = amplitude(diurnal_cycle_cube, ['day_of_year', 'year'])
    assert amplitude_cube.shape == (730,)
    assert amplitude_cube.coords('day_of_year')
    assert amplitude_cube.coords('year')
    np.testing.assert_allclose(amplitude_cube.data, DIURNAL_CYCLE_AMPLITUDE)
    assert amplitude_cube.metadata == diurnal_cycle_cube.metadata
