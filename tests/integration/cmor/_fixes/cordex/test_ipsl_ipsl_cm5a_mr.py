"""Tests for the fixes of driver IPSL-CM5A-MR."""
import iris
import pytest

from esmvalcore.cmor._fixes.cordex.ipsl_ipsl_cm5a_mr import wrf381p
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


@pytest.mark.parametrize(
    'short_name',
    ['tasmax', 'tasmin', 'tas', 'hurs', 'huss'])
def test_get_wrf381p_fix(short_name):
    fix = Fix.get_fixes(
        'CORDEX',
        'WRF381P',
        'Amon',
        short_name,
        extra_facets={'driver': 'IPSL-CM5A-MR'})
    assert isinstance(fix[0], Fix)


def test_wrf381p_height_fix():
    time_coord = iris.coords.DimCoord([0.0],
                                      var_name='time',
                                      standard_name='time',
                                      long_name='time')
    cube = iris.cube.Cube(
        [10.0],
        var_name='tas',
        dim_coords_and_dims=[(time_coord, 0)],
    )
    vardef = get_var_info('CMIP6', 'Amon', 'tas')
    fix = wrf381p.Tas(vardef)
    out_cubes = fix.fix_metadata([cube])
    assert out_cubes[0].coord('height').points == 2.0
