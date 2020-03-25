"""Test derivation of ``co2s``."""
import dask.array as da
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.co2s as co2s


def get_coord_spec():
    """Coordinate specs for cubes."""
    time_coord = iris.coords.DimCoord([0], var_name='time',
                                      standard_name='time',
                                      units='days since 0000-01-01 00:00:00')
    plev_coord = iris.coords.DimCoord([123456.0, 50000.0, 1000.0],
                                      var_name='plev',
                                      standard_name='air_pressure', units='Pa')
    lat_coord = iris.coords.DimCoord([0.0, 1.0], var_name='latitude',
                                     standard_name='latitude', units='degrees')
    lon_coord = iris.coords.DimCoord([0.0, 1.0], var_name='longitude',
                                     standard_name='longitude',
                                     units='degrees')
    coord_spec = [
        (time_coord, 0),
        (plev_coord, 1),
        (lat_coord, 2),
        (lon_coord, 3),
    ]
    return coord_spec


@pytest.fixture
def masked_cubes():
    """Masked CO2 cube."""
    coord_spec = get_coord_spec()
    co2_data = da.ma.masked_less([[[[170.0, -1.0],
                                    [-1.0, -1.0]],
                                   [[150.0, 100.0],
                                    [80.0, -1.0]],
                                   [[100.0, 50.0],
                                    [30.0, 10.0]]]], 0.0)
    cube = iris.cube.Cube(
        co2_data,
        var_name='co2',
        standard_name='mole_fraction_of_carbon_dioxide_in_air',
        units='1',
        dim_coords_and_dims=coord_spec,
    )
    return iris.cube.CubeList([cube])


@pytest.fixture
def unmasked_cubes():
    """Unmasked CO2 cube."""
    coord_spec = get_coord_spec()
    co2_data = da.array([[[[200.0, 100.0],
                           [80.0, 9.0]],
                          [[150.0, 80.0],
                           [70.0, 5.0]],
                          [[100.0, 50.0],
                           [30.0, 1.0]]]])
    cube = iris.cube.Cube(
        co2_data,
        var_name='co2',
        standard_name='mole_fraction_of_carbon_dioxide_in_air',
        units='1e-1',
        dim_coords_and_dims=coord_spec,
    )
    return iris.cube.CubeList([cube])


def test_co2_calculate_masked_cubes(masked_cubes):
    """Test function ``calculate`` with masked cube."""
    derived_var = co2s.DerivedVariable()
    out_cube = derived_var.calculate(masked_cubes)
    assert not np.ma.is_masked(out_cube.data)
    np.testing.assert_allclose(out_cube.data,
                               [[[170.0, 100.0],
                                 [80.0, 10.0]]])
    assert out_cube.units == 'mol mol-1'
    np.testing.assert_allclose(out_cube.coord('air_pressure').points,
                               123456.0)


def test_co2_calculate_unmasked_cubes(unmasked_cubes):
    """Test function ``calculate`` with unmasked cube."""
    derived_var = co2s.DerivedVariable()
    out_cube = derived_var.calculate(unmasked_cubes)
    assert not np.ma.is_masked(out_cube.data)
    np.testing.assert_allclose(out_cube.data,
                               [[[20.0, 10.0],
                                 [8.0, 0.9]]])
    assert out_cube.units == 'mol mol-1'
    np.testing.assert_allclose(out_cube.coord('air_pressure').points,
                               123456.0)
