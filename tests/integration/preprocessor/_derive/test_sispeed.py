"""Test derivation of `sispeed`."""

import math
from unittest import mock

import numpy as np

from iris.cube import Cube, CubeList
from iris.coords import AuxCoord

from esmvalcore.preprocessor._derive.sispeed import DerivedVariable


def get_cube(name, lat=((0.5, 1.5), (2.5, 3.5)), lon=((0.5, 1.5), (2.5, 3.5))):
    lat = np.array(lat)
    lon = np.array(lon)
    lat_bounds = np.array((lat - 0.5, lat + 0.5))
    lon_bounds = np.array((lon - 0.5, lon + 0.5))
    cube = Cube(np.ones((2, 2, 2)), name)
    cube.add_aux_coord(AuxCoord(lat, 'latitude', bounds=lat_bounds), (1, 2))
    cube.add_aux_coord(AuxCoord(lon, 'longitude', bounds=lon_bounds), (1, 2))
    return cube


@mock.patch(
    'esmvalcore.preprocessor._regrid.esmpy_regrid', autospec=True)
def test_sispeed_calculation(mock_regrid):
    """Test calculation of `sispeed."""
    siu = get_cube('sea_ice_x_velocity')
    siv = get_cube('sea_ice_y_velocity')
    derived_var = DerivedVariable()
    sispeed = derived_var.calculate(CubeList((siu, siv)))
    assert np.all(
        sispeed.data == np.full_like(sispeed.data, 1 * math.sqrt(2))
    )
    assert mock_regrid.call_count == 0


@mock.patch(
    'esmvalcore.preprocessor._regrid.esmpy_regrid', autospec=True)
def test_sispeed_calculation_coord_differ(mock_regrid):
    """Test calculation of `sispeed."""
    siu = get_cube('sea_ice_x_velocity')
    siv = get_cube(
        'sea_ice_y_velocity',
        lat=((0.25, 1.25), (2.25, 3.25)),
        lon=((0.25, 1.25), (2.25, 3.25))
    )
    mock_regrid.return_value = siu
    derived_var = DerivedVariable()
    sispeed = derived_var.calculate(CubeList((siu, siv)))
    assert np.all(
        sispeed.data == np.full_like(sispeed.data, 1 * math.sqrt(2))
    )
    assert mock_regrid.call_count == 1
