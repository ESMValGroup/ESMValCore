"""Tests for the fixes of CESM2-WACCM."""
import iris
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.cmip6.cesm2_waccm import Tas
from esmvalcore.cmor.fix import Fix


@pytest.fixture
def tas_cubes():
    ta_cube = iris.cube.Cube([1.0], var_name='ta')
    tas_cube = iris.cube.Cube([3.0], var_name='tas')
    return iris.cube.CubeList([ta_cube, tas_cube])


def test_get_tas_fix():
    fix = Fix.get_fixes('CMIP6', 'CESM2-WACCM', 'tas')
    assert fix == [Tas()]


def test_tas_fix_metadata(tas_cubes):
    for cube in tas_cubes:
        with pytest.raises(iris.exceptions.CoordinateNotFoundError):
            cube.coord('height')
    height_coord = iris.coords.AuxCoord(2.0,
                                        var_name='height',
                                        standard_name='height',
                                        long_name='height',
                                        units=Unit('m'),
                                        attributes={'positive': 'up'})
    fix = Tas()
    out_cubes = fix.fix_metadata(tas_cubes)
    assert out_cubes is tas_cubes
    for cube in out_cubes:
        if cube.var_name == 'tas':
            coord = cube.coord('height')
            assert coord == height_coord
        else:
            with pytest.raises(iris.exceptions.CoordinateNotFoundError):
                cube.coord('height')
