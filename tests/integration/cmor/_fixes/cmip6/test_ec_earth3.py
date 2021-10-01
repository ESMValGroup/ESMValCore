"""Tests for EC-Earth3."""
import unittest

import cf_units
import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip6.ec_earth3 import AllVars, Siconca, Tas
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


class TestSiconca(unittest.TestCase):
    """Test sftof fixes."""

    def setUp(self):
        """Prepare tests."""
        self.cube = Cube([1.0], var_name='siconca', units='%')
        self.fix = Siconca(None)

    def test_get(self):
        """Test fix get."""
        assert Siconca(None) in Fix.get_fixes('CMIP6', 'EC-Earth3', 'SImon',
                                              'siconca')

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 100)
        self.assertEqual(cube.units, Unit('%'))


@pytest.fixture
def tas_cubes():
    """Cubes to test fixes for ``tas``."""
    time_coord = iris.coords.DimCoord([0.0, 1.0],
                                      var_name='time',
                                      standard_name='time',
                                      units='days since 1850-01-01 00:00:00')
    lat_coord = iris.coords.DimCoord([0.0, 1.0],
                                     bounds=[[-0.5, 0.5], [0.5, 1.5]],
                                     var_name='lat',
                                     standard_name='latitude',
                                     units='degrees')
    lat_coord_to_round = iris.coords.DimCoord(
        [0.0000000001, 0.9999999999],
        bounds=[[-0.5000000001, 0.5000000001], [0.5000000001, 1.5000000001]],
        var_name='lat',
        standard_name='latitude',
        units='degrees')
    lon_coord = iris.coords.DimCoord([0.0, 1.0],
                                     var_name='lon',
                                     standard_name='longitude',
                                     units='degrees')
    tas_cube = iris.cube.Cube(
        np.ones((2, 2, 2)),
        var_name='tas',
        dim_coords_and_dims=[(time_coord, 0), (lat_coord, 1), (lon_coord, 2)],
    )
    tas_cube_to_round = iris.cube.Cube(
        np.ones((2, 2, 2)),
        var_name='tas',
        dim_coords_and_dims=[(time_coord, 0), (lat_coord_to_round, 1),
                             (lon_coord, 2)],
    )

    return iris.cube.CubeList([tas_cube, tas_cube_to_round])


def test_get_tas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'EC-Earth3', 'Amon', 'tas')
    assert Tas(None) in fix


def test_tas_fix_metadata(tas_cubes):
    """Test ``fix_metadata`` for ``tas``."""
    vardef = get_var_info('CMIP6', 'Amon', 'tas')
    fix = Tas(vardef)
    fixed_cubes = fix.fix_metadata(tas_cubes)
    assert fixed_cubes[0].coord('latitude') == fixed_cubes[1].coord('latitude')


def test_get_allvars_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('CMIP6', 'EC-Earth3', 'Amon', 'tas')
    assert AllVars(None) in fixes


def test_allvars_r3i1p1f1_fix_calendar():
    vardef = get_var_info('CMIP6', 'Amon', 'tas')
    fix = AllVars(vardef)
    cube = iris.cube.Cube([1, 2])
    bad_unit = cf_units.Unit('days since 1850-01-01 00:00:00', 'gregorian')
    time_coord = iris.coords.DimCoord(
        [0.0, 1.0],
        var_name='time',
        standard_name='time',
        units=bad_unit,
    )
    cube.add_dim_coord(time_coord, 0)
    cube.attributes['experiment_id'] = 'historical'
    cube.attributes['variant_label'] = 'r3i1p1f1'

    fixed_cubes = fix.fix_metadata([cube])
    good_unit = cf_units.Unit('days since 1850-01-01 00:00:00',
                              'proleptic_gregorian')
    assert fixed_cubes[0].coord('time').units == good_unit


def test_allvars_r3i1p1f1_fix_latitude():
    lat_coord1 = iris.coords.DimCoord(
        [-71.22775],
        var_name='lat',
        standard_name='latitude',
        units='degrees',
    )
    lat_coord2 = iris.coords.DimCoord(
        [-71.22774993],
        var_name='lat',
        standard_name='latitude',
        units='degrees',
    )

    cube1 = iris.cube.Cube([0])
    cube1.attributes['variant_label'] = 'r3i1p1f1'
    cube1.add_dim_coord(lat_coord1, 0)

    cube2 = iris.cube.Cube([0])
    cube2.attributes['variant_label'] = 'r3i1p1f1'
    cube2.add_dim_coord(lat_coord2, 0)

    fix = AllVars(None)
    fixed_cubes = fix.fix_metadata([cube1, cube2])

    assert fixed_cubes[0].coord('latitude').points[0] == -71.228
    assert fixed_cubes[1].coord('latitude').points[0] == -71.228
