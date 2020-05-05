"""Tests for the fixes of IPSL-CM6A-LR."""
import unittest

import iris
import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError

from esmvalcore.cmor._fixes.cmip6.ipsl_cm6a_lr import AllVars, Clcalipso
from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor.table import get_var_info


class TestAllVars(unittest.TestCase):
    """Tests for fixes of all variables."""

    def setUp(self):
        """Set up tests."""
        self.fix = AllVars(None)
        self.cube = Cube(np.random.rand(2, 2, 2), var_name='ch4')
        self.cube.add_aux_coord(
            AuxCoord(np.random.rand(2, 2),
                     var_name='nav_lat',
                     standard_name='latitude'), (1, 2))
        self.cube.add_aux_coord(
            AuxCoord(np.random.rand(2, 2),
                     var_name='nav_lon',
                     standard_name='longitude'), (1, 2))

    def test_fix_metadata_ocean_var(self):
        """Test ``fix_metadata`` for ocean variables."""
        cell_area = Cube(np.random.rand(2, 2), standard_name='cell_area')
        cubes = self.fix.fix_metadata(CubeList([self.cube, cell_area]))

        self.assertEqual(len(cubes), 1)
        cube = cubes[0]
        self.assertEqual(cube.coord('latitude').var_name, 'lat')
        self.assertEqual(cube.coord('longitude').var_name, 'lon')
        self.cube.coord('cell_area')

    def test_fix_data_other_var(self):
        """Test ``fix_metadata`` for other variables."""
        cubes = self.fix.fix_metadata(CubeList([self.cube]))

        self.assertEqual(len(cubes), 1)
        cube = cubes[0]
        self.assertEqual(cube.coord('latitude').var_name, 'nav_lat')
        self.assertEqual(cube.coord('longitude').var_name, 'nav_lon')
        with self.assertRaises(CoordinateNotFoundError):
            self.cube.coord('cell_area')


def test_get_clcalipso_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'IPSL-CM6A-LR', 'CFmon', 'clcalipso')
    assert fix == [Clcalipso(None), AllVars(None)]


@pytest.fixture
def clcalipso_cubes():
    """Cubes to test fix for ``clcalipso``."""
    alt_40_coord = iris.coords.DimCoord([0.0], var_name='height')
    cube = iris.cube.Cube([0.0], var_name='clcalipso',
                          dim_coords_and_dims=[(alt_40_coord.copy(), 0)])
    x_cube = iris.cube.Cube([0.0], var_name='x',
                            dim_coords_and_dims=[(alt_40_coord.copy(), 0)])
    return iris.cube.CubeList([cube, x_cube])


def test_clcalipso_fix_metadata(clcalipso_cubes):
    """Test ``fix_metadata`` for ``clcalipso``."""
    vardef = get_var_info('CMIP6', 'CFmon', 'clcalipso')
    fix = Clcalipso(vardef)
    cubes = fix.fix_metadata(clcalipso_cubes)
    assert len(cubes) == 1
    cube = cubes[0]
    coord = cube.coord('altitude')
    assert coord.long_name == 'altitude'
    assert coord.standard_name == 'altitude'
    assert coord.var_name == 'alt40'
