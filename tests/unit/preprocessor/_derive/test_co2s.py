"""Test derivation of ``co2s``."""
import dask.array as da
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.co2s as co2s


def get_coord_spec(include_plev=True):
    """Coordinate specs for cubes."""
    time_coord = iris.coords.DimCoord([0], var_name='time',
                                      standard_name='time',
                                      units='days since 0000-01-01 00:00:00')
    lat_coord = iris.coords.DimCoord([0.0, 1.0], var_name='latitude',
                                     standard_name='latitude', units='degrees')
    lon_coord = iris.coords.DimCoord([0.0, 1.0], var_name='longitude',
                                     standard_name='longitude',
                                     units='degrees')
    if include_plev:
        plev_coord = iris.coords.DimCoord([100000.0, 90000.0, 50000.0],
                                          var_name='plev',
                                          standard_name='air_pressure',
                                          units='Pa')
        coord_spec = [
            (time_coord, 0),
            (plev_coord, 1),
            (lat_coord, 2),
            (lon_coord, 3),
        ]
    else:
        coord_spec = [
            (time_coord, 0),
            (lat_coord, 1),
            (lon_coord, 2),
        ]
    return coord_spec


def get_ps_cube():
    """Surface air pressure cube."""
    ps_data = [[[105000.0, 50000.0],
                [95000.0, 60000.0]]]
    coord_spec = get_coord_spec(include_plev=False)
    cube = iris.cube.Cube(
        ps_data,
        var_name='ps',
        standard_name='surface_air_pressure',
        units='Pa',
        dim_coords_and_dims=coord_spec,
    )
    return cube


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
    co2_cube = iris.cube.Cube(
        co2_data,
        var_name='co2',
        standard_name='mole_fraction_of_carbon_dioxide_in_air',
        units='1e-6',
        dim_coords_and_dims=coord_spec,
    )
    ps_cube = get_ps_cube()
    return iris.cube.CubeList([co2_cube, ps_cube])


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
    co2_cube = iris.cube.Cube(
        co2_data,
        var_name='co2',
        standard_name='mole_fraction_of_carbon_dioxide_in_air',
        units='1e-8',
        dim_coords_and_dims=coord_spec,
    )
    ps_cube = get_ps_cube()
    return iris.cube.CubeList([co2_cube, ps_cube])


def test_co2_calculate_masked_cubes(masked_cubes):
    """Test function ``calculate`` with masked cube."""
    derived_var = co2s.DerivedVariable()
    out_cube = derived_var.calculate(masked_cubes)
    assert not np.ma.is_masked(out_cube.data)
    np.testing.assert_allclose(out_cube.data,
                               [[[180.0, 50.0],
                                 [80.0, 10.0]]])
    assert out_cube.units == '1e-6'
    plev_coord = out_cube.coord('air_pressure')
    assert plev_coord.var_name == 'plev'
    assert plev_coord.standard_name == 'air_pressure'
    assert plev_coord.long_name == 'pressure'
    assert plev_coord.units == 'Pa'
    np.testing.assert_allclose(plev_coord.points,
                               [[[105000.0, 50000.0], [95000.0, 60000.0]]])


def test_co2_calculate_unmasked_cubes(unmasked_cubes):
    """Test function ``calculate`` with unmasked cube."""
    derived_var = co2s.DerivedVariable()
    out_cube = derived_var.calculate(unmasked_cubes)
    assert not np.ma.is_masked(out_cube.data)
    np.testing.assert_allclose(out_cube.data,
                               [[[2.25, 0.50],
                                 [0.75, 0.02]]])
    assert out_cube.units == '1e-6'
    plev_coord = out_cube.coord('air_pressure')
    assert plev_coord.var_name == 'plev'
    assert plev_coord.standard_name == 'air_pressure'
    assert plev_coord.long_name == 'pressure'
    assert plev_coord.units == 'Pa'
    np.testing.assert_allclose(plev_coord.points,
                               [[[105000.0, 50000.0], [95000.0, 60000.0]]])
