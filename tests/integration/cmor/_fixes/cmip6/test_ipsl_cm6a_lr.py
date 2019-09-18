import unittest

import numpy as np

from iris.cube import Cube, CubeList
from iris.coords import AuxCoord
from iris.exceptions import CoordinateNotFoundError

from esmvalcore.cmor._fixes.cmip6.ipsl_cm6a_lr import AllVars


class TestAllVars(unittest.TestCase):
    def setUp(self):
        self.fix = AllVars()
        self.cube = Cube(np.random.rand(2, 2, 2), var_name='ch4')
        self.cube.add_aux_coord(
            AuxCoord(
                np.random.rand(2, 2),
                var_name='nav_lat',
                standard_name='latitude'
            ),
            (1, 2)
        )
        self.cube.add_aux_coord(
            AuxCoord(
                np.random.rand(2, 2),
                var_name='nav_lon',
                standard_name='longitude'
            ),
            (1, 2)
        )

    def test_fix_metadata_ocean_var(self):
        cell_area = Cube(np.random.rand(2, 2), standard_name='cell_area')
        cubes = self.fix.fix_metadata(CubeList([self.cube, cell_area]))

        self.assertEqual(len(cubes), 1)
        cube = cubes[0]
        self.assertEquals(cube.coord('latitude').var_name, 'lat')
        self.assertEquals(cube.coord('longitude').var_name, 'lon')
        self.cube.coord('cell_area')

    def test_fix_data_other_var(self):
        cubes = self.fix.fix_metadata(CubeList([self.cube]))

        self.assertEqual(len(cubes), 1)
        cube = cubes[0]
        self.assertEqual(cube.coord('latitude').var_name, 'nav_lat')
        self.assertEqual(cube.coord('longitude').var_name, 'nav_lon')
        with self.assertRaises(CoordinateNotFoundError):
            self.cube.coord('cell_area')
