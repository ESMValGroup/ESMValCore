"""Tests for the fixes for driver ICHEC-EC-Earth."""
import iris
import pytest

from esmvalcore.cmor._fixes.cordex.ichec_ec_earth import dmi_hirham5
from esmvalcore.cmor.fix import Fix


@pytest.fixture
def cubes():
    correct_time_coord = iris.coords.DimCoord([0.0],
                                              var_name='time',
                                              standard_name='time',
                                              long_name='time')
    correct_height_coord = iris.coords.AuxCoord([2.0], var_name='height')
    correct_cube = iris.cube.Cube([10.0],
                                  var_name='tas',
                                  dim_coords_and_dims=[(correct_time_coord, 0)
                                                       ],
                                  aux_coords_and_dims=[(correct_height_coord,
                                                        ())])
    wrong_cube = iris.cube.Cube(
        [10.0],
        var_name='tas',
        dim_coords_and_dims=[(correct_time_coord, 0)],
    )
    return iris.cube.CubeList([correct_cube, wrong_cube])


def test_get_gerics_remo2015_fix():
    fix = Fix.get_fixes('CORDEX',
                        'GERICS-REMO2015',
                        'Amon',
                        'pr',
                        extra_facets={'driver': 'ICHEC-EC-Earth'})
    assert isinstance(fix[0], Fix)


def test_get_dmi_hirham5_fix():
    fix = Fix.get_fixes(
        'CORDEX',
        'DMI-HIRHAM5',
        'Amon',
        'tas',
        extra_facets={'driver': 'ICHEC-EC-Earth'},
    )
    assert isinstance(fix[0], Fix)


def test_get_knmi_racmo22e_fix():
    fix = Fix.get_fixes('CORDEX',
                        'KNMI-RACMO22E',
                        'Amon',
                        'pr',
                        extra_facets={'driver': 'ICHEC-EC-Earth'})
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_mohc_hadrem3ga705_fix(short_name):
    fix = Fix.get_fixes('CORDEX',
                        'MOHC-HadREM3-GA7-05',
                        'Amon',
                        short_name,
                        extra_facets={'driver': 'ICHEC-EC-Earth'})
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize('short_name', ['pr', 'tas'])
def test_get_smhi_rca4_fix(short_name):
    fix = Fix.get_fixes('CORDEX',
                        'SMHI-RCA4',
                        'Amon',
                        short_name,
                        extra_facets={'driver': 'ICHEC-EC-Earth'})
    assert isinstance(fix[0], Fix)


def test_dmi_hirham5_height_fix(cubes):
    fix = dmi_hirham5.Tas(None)
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        assert cube.coord('height').points == 2.0
