"""Tests for the fixes of IPSL-CM6A-LR."""
import unittest

import iris
import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError

from esmvalcore.cmor._fixes.cmip6.ipsl_cm6a_lr import AllVars, Clcalipso, Omon
from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor.table import get_var_info


class TestAllVars(unittest.TestCase):
    """Tests for fixes of all variables."""

    def setUp(self):
        """Set up tests."""
        vardef = get_var_info('CMIP6', 'Omon', 'tos')
        self.fix = AllVars(vardef)
        self.cube = Cube(np.random.rand(2, 2, 2), var_name='tos')
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
        self.assertEqual(cube.var_name, 'tos')
        self.assertEqual(cube.coord('latitude').var_name, 'lat')
        self.assertEqual(cube.coord('longitude').var_name, 'lon')

    def test_fix_data_no_lat(self):
        """Test ``fix_metadata`` when no latitude is present."""
        self.cube.remove_coord('latitude')
        cubes = self.fix.fix_metadata(CubeList([self.cube]))

        self.assertEqual(len(cubes), 1)
        cube = cubes[0]
        self.assertEqual(cube.coord('longitude').var_name, 'lon')
        with self.assertRaises(CoordinateNotFoundError):
            self.cube.coord('latitude')

    def test_fix_data_no_lon(self):
        """Test ``fix_metadata`` when no longitude is present."""
        self.cube.remove_coord('longitude')
        cubes = self.fix.fix_metadata(CubeList([self.cube]))

        self.assertEqual(len(cubes), 1)
        cube = cubes[0]
        self.assertEqual(cube.coord('latitude').var_name, 'lat')
        with self.assertRaises(CoordinateNotFoundError):
            self.cube.coord('longitude')

    def test_fix_data_no_lat_lon(self):
        """Test ``fix_metadata`` for cubes with no latitude and longitude."""
        self.cube.remove_coord('latitude')
        self.cube.remove_coord('longitude')
        cubes = self.fix.fix_metadata(CubeList([self.cube]))

        self.assertEqual(len(cubes), 1)
        with self.assertRaises(CoordinateNotFoundError):
            self.cube.coord('latitude')
        with self.assertRaises(CoordinateNotFoundError):
            self.cube.coord('longitude')


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


@pytest.fixture
def thetao_cubes():
    """Cubes to test fixes for ``thetao``."""
    time_coord = iris.coords.DimCoord(
        [0.0004, 1.09776], var_name='time', standard_name='time',
        units='days since 1850-01-01 00:00:00')
    lat_coord = iris.coords.DimCoord(
        [0.0, 1.0], var_name='lat', standard_name='latitude', units='degrees')
    lon_coord = iris.coords.DimCoord(
        [0.0, 1.0], var_name='lon', standard_name='longitude', units='degrees')
    lev_coord = iris.coords.DimCoord(
        [5.0, 10.0], bounds=[[2.5, 7.5], [7.5, 12.5]],
        var_name='olevel', standard_name=None, units='m',
        attributes={'positive': 'up'})
    coord_specs = [
        (time_coord, 0),
        (lev_coord, 1),
        (lat_coord, 2),
        (lon_coord, 3),
    ]
    thetao_cube = iris.cube.Cube(
        np.ones((2, 2, 2, 2)),
        var_name='thetao',
        dim_coords_and_dims=coord_specs,
    )
    return iris.cube.CubeList([thetao_cube])


def test_get_thetao_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'IPSL-CM6A-LR', 'Omon', 'thetao')
    assert fix == [Omon(None), AllVars(None)]


def test_thetao_fix_metadata(thetao_cubes):
    """Test ``fix_metadata`` for ``thetao``."""
    vardef = get_var_info('CMIP6', 'Omon', 'thetao')
    fix = Omon(vardef)
    out_cubes = fix.fix_metadata(thetao_cubes)
    assert out_cubes is thetao_cubes
    assert len(out_cubes) == 1
    out_cube = out_cubes[0]

    # Check metadata of depth coordinate
    depth_coord = out_cube.coord('depth')
    assert depth_coord.standard_name == 'depth'
    assert depth_coord.var_name == 'lev'
    assert depth_coord.long_name == 'ocean depth coordinate'
    assert depth_coord.units == 'm'
    assert depth_coord.attributes == {'positive': 'down'}
