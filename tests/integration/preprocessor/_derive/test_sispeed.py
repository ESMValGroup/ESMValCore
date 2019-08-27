"""Test derivation of `sispeed`."""

import math

import numpy as np
import pytest

from iris.cube import Cube, CubeList
from iris.coords import AuxCoord

from esmvalcore.preprocessor._derive.sispeed import DerivedVariable


def get_cube(name, lat=((0, 1), (2, 3)), lon=((0, 1), (2, 3))):
    cube = Cube(np.ones((2, 2)), name)
    cube.add_aux_coord(AuxCoord(lat, 'latitude'), (0, 1))
    cube.add_aux_coord(AuxCoord(lon, 'longitude'), (0, 1))
    return cube


def test_sispeed_calculation():
    """Test calculation of `sispeed."""
    siu = get_cube('sea_ice_x_velocity')
    siv = get_cube('sea_ice_y_velocity')
    derived_var = DerivedVariable()
    sispeed = derived_var.calculate(CubeList((siu, siv)))
    assert np.all(
        sispeed.data == np.full_like(sispeed.data, 1 * math.sqrt(2))
    )


def test_sispeed_calculation_small_coord_difference():
    """Test calculation of `sispeed."""
    siu = get_cube('sea_ice_x_velocity')
    siv = get_cube(
        'sea_ice_y_velocity', lat=((2, 1), (2, 3)), lon=((2, 1), (2, 3))
    )
    derived_var = DerivedVariable()
    sispeed = derived_var.calculate(CubeList((siu, siv)))
    assert np.all(
        sispeed.data == np.full_like(sispeed.data, 1 * math.sqrt(2))
    )
    assert np.all(
        sispeed.coord('latitude').points == siu.coord('latitude').points
    )
    assert np.all(
        sispeed.coord('longitude').points == siu.coord('longitude').points
    )


def test_sispeed_calculation_lat_differ_too_much():
    """Test calculation of `sispeed."""
    siu = get_cube('sea_ice_x_velocity')
    siv = get_cube('sea_ice_y_velocity', lat=((6, 1), (2, 3)))
    derived_var = DerivedVariable()
    with pytest.raises(ValueError):
        derived_var.calculate(CubeList((siu, siv)))


def test_sispeed_calculation_lon_differ_too_much():
    """Test calculation of `sispeed."""
    siu = get_cube('sea_ice_x_velocity')
    siv = get_cube('sea_ice_y_velocity', lon=((6, 1), (2, 3)))
    derived_var = DerivedVariable()
    with pytest.raises(ValueError):
        derived_var.calculate(CubeList((siu, siv)))
