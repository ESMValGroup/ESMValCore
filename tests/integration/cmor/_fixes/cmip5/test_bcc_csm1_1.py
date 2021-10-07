"""Test bcc-csm1-1 fixes."""
import unittest

import iris
import numpy as np

from esmvalcore.cmor._fixes.cmip5.bcc_csm1_1 import Cl, Tos
from esmvalcore.cmor._fixes.common import (
    ClFixHybridPressureCoord,
    OceanFixGrid,
)
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


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
            Fix.get_fixes('CMIP5', 'bcc-csm1-1', 'Amon', 'tos'), [Tos(None)])


def test_tos_fix():
    """Test fix for ``tos``."""
    assert Tos is OceanFixGrid


def test_tos_fix_metadata():
    """Test ``fix_metadata`` for ``tos``."""
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
    time_coord = iris.coords.DimCoord(
        1.0,
        bounds=[0.0, 2.0],
        var_name='time',
        standard_name='time',
        long_name='time',
        units='days since 1950-01-01',
    )

    # Create cube without bounds
    cube = iris.cube.Cube(
        np.full((1, 2, 3), 300.0),
        var_name='tos',
        standard_name='sea_surface_temperature',
        units='K',
        dim_coords_and_dims=[(time_coord, 0), (grid_lat, 1), (grid_lon, 2)],
        aux_coords_and_dims=[(latitude, (1, 2)), (longitude, (1, 2))],
    )
    assert cube.coord('latitude').bounds is None
    assert cube.coord('longitude').bounds is None

    # Apply fix
    vardef = get_var_info('CMIP6', 'Omon', 'tos')
    fix = Tos(vardef)
    cubes = iris.cube.CubeList([cube])
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes.extract_cube('sea_surface_temperature')
    assert fixed_cube is cube
    i_coord = fixed_cube.coord('cell index along first dimension')
    j_coord = fixed_cube.coord('cell index along second dimension')
    assert i_coord.var_name == 'i'
    assert i_coord.standard_name is None
    assert i_coord.long_name == 'cell index along first dimension'
    assert i_coord.units == '1'
    assert i_coord.circular is False
    assert j_coord.var_name == 'j'
    assert j_coord.standard_name is None
    assert j_coord.long_name == 'cell index along second dimension'
    assert j_coord.units == '1'
    np.testing.assert_allclose(i_coord.points, [0, 1, 2])
    np.testing.assert_allclose(i_coord.bounds,
                               [[-0.5, 0.5], [0.5, 1.5], [1.5, 2.5]])
    np.testing.assert_allclose(j_coord.points, [0, 1])
    np.testing.assert_allclose(j_coord.bounds, [[-0.5, 0.5], [0.5, 1.5]])
    assert fixed_cube.coord('latitude').bounds is not None
    assert fixed_cube.coord('longitude').bounds is not None
    latitude_bounds = np.array(
        [[[-43.48076211, -34.01923789, -22.00961894, -31.47114317],
          [-34.01923789, -10.0, 2.00961894, -22.00961894],
          [-10.0, -0.53847577, 11.47114317, 2.00961894]],
         [[-31.47114317, -22.00961894, -10.0, -19.46152423],
          [-22.00961894, 2.00961894, 14.01923789, -10.0],
          [2.00961894, 11.47114317, 23.48076211, 14.01923789]]]
    )
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
