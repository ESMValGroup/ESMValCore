"""Tests for the fixes of MSWEP."""
from pathlib import Path

import iris
import numpy as np
import pytest

from esmvalcore.cmor._fixes.native6.mswep import (
    Pr,
    fix_longitude,
    fix_time_day,
    fix_time_month,
)
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import CMOR_TABLES


@pytest.mark.parametrize('mip_table', ('Amon', 'day'))
def test_get_pr_fix(mip_table):
    """Test whether the right fix gets found."""
    fix = Fix.get_fixes('native6', 'MSWEP', mip_table, 'pr')
    assert isinstance(fix[0], Pr)


@pytest.fixture
def cube_month():
    """Return extract from mswep monthly data (shape 3x5x5)."""
    # out = cube[0:3, 0:360:72, 0:720:144]
    # iris.save(out, 'mswep_month.nc')
    path = Path(__file__).with_name('mswep_month.nc')
    return iris.load_cube(str(path))


@pytest.fixture
def cube_day():
    """Return extract from mswep daily data (shape 3x5x5)."""
    # out = cube[0:3, 0:360:72, 0:720:144]
    # iris.save(out, 'mswep_day.nc')
    path = Path(__file__).with_name('mswep_day.nc')
    return iris.load_cube(str(path))


@pytest.fixture
def fix_month():
    """Return fix for monthly pr data."""
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('Amon', 'pr')
    return Pr(vardef)


@pytest.fixture
def fix_day():
    """Return fix for daily pr data."""
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('day', 'pr')
    return Pr(vardef)


def test_fix_names(fix_month, cube_month):
    """Test `Pr._fix_names`."""
    fix_month._fix_names(cube_month)

    vardef = fix_month.vardef

    assert cube_month.var_name == vardef.short_name
    assert cube_month.long_name == vardef.long_name
    assert cube_month.standard_name == vardef.standard_name

    coord_defs = tuple(coord_def for coord_def in vardef.coordinates.values())

    for coord_def in coord_defs:
        coord = cube_month.coord(axis=coord_def.axis)
        assert coord.long_name == coord_def.long_name


def test_fix_units_month(fix_month, cube_month):
    """Test `Pr._fix_units_month`."""
    fix_month._fix_units(cube_month)
    assert cube_month.units == fix_month.vardef.units


def test_fix_units_day(fix_day, cube_day):
    """Test `Pr._fix_units_day`."""
    fix_day._fix_units(cube_day)
    assert cube_day.units == fix_day.vardef.units


def test_fix_time_month(cube_month):
    """Test `fix_time_month`."""
    fix_time_month(cube_month)

    time = cube_month.coord('time')
    assert time.units == 'days since 1850-01-01'


def test_fix_time_day(cube_day):
    """Test `fix_time_day`."""
    fix_time_day(cube_day)

    time = cube_day.coord('time')
    assert time.units == 'days since 1850-01-01'


def test_fix_longitude(fix_month, cube_month):
    """Test `Pr._fix_longitude`."""
    unfixed_data = cube_month.data.copy()
    unfixed_lon = cube_month.coord(axis='X')
    shift = (unfixed_lon.points < 0).sum()

    fix_longitude(cube_month)

    lon = cube_month.coord(axis='X')

    assert lon.is_monotonic

    coord_def = fix_month.vardef.coordinates['longitude']
    valid_min = float(coord_def.valid_min)
    valid_max = float(coord_def.valid_max)

    assert lon.points.min() >= valid_min
    assert lon.points.max() <= valid_max

    # make sure that data are rolled correctly along lon axis
    assert np.all(unfixed_data[:, :, 0] == cube_month.data[:, :, -shift])


def test_fix_bounds(fix_month, cube_month):
    """Test `Pr._fix_bounds`."""
    fix_month._fix_bounds(cube_month)

    for axis in 'XYT':
        coord = cube_month.coord(axis=axis)
        assert coord.has_bounds()
