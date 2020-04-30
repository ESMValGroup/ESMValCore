"""Test bcc-csm1-1 fixes."""
import unittest

import iris
import numpy as np

from esmvalcore.cmor._fixes.cmip5.bcc_csm1_1 import Cl, Tos
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor.fix import Fix


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'bcc-csm1-1', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


class TestTos(unittest.TestCase):
    """Test tos fixes."""

    def test_get(self):
        """Test fix get."""
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'BCC-CSM1-1', 'Amon', 'tos'), [Tos(None)])


def test_tos_fix_data():
    """Test ``fix_data`` for ``tos``."""
    grid_lat = iris.coords.DimCoord(
        [20.0, 40.0],
        bounds=[[10.0, 30.0], [30.0, 50.0]],
        var_name='rlat',
        standard_name='grid_latitude',
    )
    grid_lon = iris.coords.DimCoord(
        [10.0, 20.0, 30.0],
        bounds=[[5.0, 15.0], [15.0, 25.0], [25.0, 35.0]],
        var_name='rlon',
        standard_name='grid_longitude',
    )
    latitude = iris.coords.AuxCoord(
        [[-40.0, -20.0, 0.0], [-20.0, 0.0, 20.0]],
        var_name='lat',
        standard_name='latitude',
        units='degrees_north',
    )
    longitude = iris.coords.AuxCoord(
        [[100.0, 140.0, 180.0], [80.0, 100.0, 120.0]],
        var_name='lon',
        standard_name='longitude',
        units='degrees_east',
    )

    # Create cube without bounds
    cube = iris.cube.Cube(
        np.full((2, 3), 300.0),
        var_name='tos',
        units='K',
        dim_coords_and_dims=[(grid_lat, 0), (grid_lon, 1)],
        aux_coords_and_dims=[(latitude, (0, 1)), (longitude, (0, 1))],
    )
    assert cube.coord('latitude').bounds is None
    assert cube.coord('longitude').bounds is None

    # Apply fix
    fix = Tos(None)
    fixed_cube = fix.fix_data(cube)
    assert fixed_cube is cube
    assert fixed_cube.coord('latitude').bounds is not None
    assert fixed_cube.coord('longitude').bounds is not None
    latitude_bounds = np.array([[[-40, -33.75, -23.75, -30.0],
                                 [-33.75, -6.25, 3.75, -23.75],
                                 [-6.25, -1.02418074021670e-14, 10.0, 3.75]],
                                [[-30.0, -23.75, -13.75, -20.0],
                                 [-23.75, 3.75, 13.75, -13.75],
                                 [3.75, 10.0, 20.0, 13.75]]])
    np.testing.assert_allclose(fixed_cube.coord('latitude').bounds,
                               latitude_bounds)
    longitude_bounds = np.array([[[140.625, 99.375, 99.375, 140.625],
                                  [99.375, 140.625, 140.625, 99.375],
                                  [140.625, 99.375, 99.375, 140.625]],
                                 [[140.625, 99.375, 99.375, 140.625],
                                  [99.375, 140.625, 140.625, 99.375],
                                  [140.625, 99.375, 99.375, 140.625]]])
    np.testing.assert_allclose(fixed_cube.coord('longitude').bounds,
                               longitude_bounds)
