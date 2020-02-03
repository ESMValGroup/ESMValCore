"""Test derivation of `uajet`."""
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.uajet as uajet

TIME_COORD = iris.coords.DimCoord([1.0, 2.0, 3.0], standard_name='time')
LEV_COORD = iris.coords.DimCoord([80000.0, 83000.0, 87000.0],
                                 standard_name='air_pressure')
LON_COORD = iris.coords.DimCoord([0.0, 90.0, 180.0, 240.0, 360.0],
                                 standard_name='longitude')


def broadcast(lat_array):
    target_shape = (len(LEV_COORD.points), len(lat_array),
                    len(LON_COORD.points))
    lat_array = np.expand_dims(lat_array, -1)
    lat_array = np.broadcast_to(lat_array, target_shape)
    return lat_array


def gaussian(lat_array, shift):
    return np.exp(-(lat_array - shift)**2 / (2 * 10**2))


@pytest.fixture
def cubes():
    lat_array = np.array(
        [-90.0, -80.0, -70.0, -60.0, -50.0, -40.0, -30.0, -20.0, -10.0, 0.0])
    lat_coord = iris.coords.DimCoord(lat_array, standard_name='latitude')

    # Produce data using Gaussian
    y_40 = broadcast(gaussian(lat_array, -40.0))
    y_50 = broadcast(gaussian(lat_array, -50.0))
    y_60 = broadcast(gaussian(lat_array, -60.0))
    y_data = np.array([y_40, y_50, y_60])
    ua_cube = iris.cube.Cube(y_data,
                             standard_name='eastward_wind',
                             dim_coords_and_dims=[(TIME_COORD, 0),
                                                  (LEV_COORD, 1),
                                                  (lat_coord, 2),
                                                  (LON_COORD, 3)])

    # Dummy cube
    ta_cube = iris.cube.Cube([1.0], standard_name='air_temperature')
    return iris.cube.CubeList([ua_cube, ta_cube])


def test_uajet_calculation(cubes):
    derived_var = uajet.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    real_cube = iris.cube.Cube(
        [-40.0, -50.0, -60.0],
        units='degrees_north',
        dim_coords_and_dims=[(TIME_COORD, 0)],
        attributes={
            'plev': 85000,
            'lat_range_0': -80.0,
            'lat_range_1': -30.0,
        },
    )
    assert out_cube == real_cube
