"""Tests for EC-Earth3-Veg."""
import unittest

import cf_units
import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip6.ec_earth3_veg import (
    CalendarFix,
    Siconc,
    Siconca,
    Tas,
)
from esmvalcore.cmor._fixes.fix import GenericFix
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
        self.assertListEqual(
            Fix.get_fixes('CMIP6', 'EC-Earth3-Veg', 'SImon', 'siconca'),
            [Siconca(None), GenericFix(None)])

    def test_fix_data(self):
        """Test data fix."""
        cube = self.fix.fix_data(self.cube)
        self.assertEqual(cube.data[0], 100)
        self.assertEqual(cube.units, Unit('%'))


def test_get_siconc_fix():
    """Test sinconc calendar is fixed."""
    fix = Fix.get_fixes('CMIP6', 'EC-Earth3-Veg', 'SImon', 'siconc')[0]
    assert isinstance(fix, CalendarFix)


def test_get_tos_fix():
    """Test tos calendar is fixed."""
    fix = Fix.get_fixes('CMIP6', 'EC-Earth3-Veg', 'Omon', 'tos')[0]
    assert isinstance(fix, CalendarFix)


def test_siconc_fix_calendar():
    vardef = get_var_info('CMIP6', 'SImon', 'siconc')
    fix = Siconc(vardef)
    cube = iris.cube.Cube([1, 2])
    bad_unit = cf_units.Unit('days since 1850-01-01 00:00:00', 'gregorian')
    time_coord = iris.coords.DimCoord(
        [0.0, 1.0],
        var_name='time',
        standard_name='time',
        units=bad_unit,
    )
    cube.add_dim_coord(time_coord, 0)

    fixed_cubes = fix.fix_metadata([cube])
    good_unit = cf_units.Unit('days since 1850-01-01 00:00:00',
                              'proleptic_gregorian')
    assert fixed_cubes[0].coord('time').units == good_unit


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
    fix = Fix.get_fixes('CMIP6', 'EC-Earth3-Veg', 'Amon', 'tas')
    assert fix == [Tas(None), GenericFix(None)]


def test_tas_fix_metadata(tas_cubes):
    """Test ``fix_metadata`` for ``tas``."""
    vardef = get_var_info('CMIP6', 'Amon', 'tas')
    fix = Tas(vardef)
    fixed_cubes = fix.fix_metadata(tas_cubes)
    assert fixed_cubes[0].coord('latitude') == fixed_cubes[1].coord('latitude')
