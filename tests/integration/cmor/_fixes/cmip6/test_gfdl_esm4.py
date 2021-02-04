"""Tests for the fixes of GFDL-ESM4."""

import iris
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.cmip6.gfdl_esm4 import Siconc
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def siconc_cubes():
    """Sample cube."""
    time_coord = iris.coords.DimCoord([0.0], standard_name='time',
                                      var_name='time',
                                      units='days since 6543-2-1')
    lat_coord = iris.coords.DimCoord([-30.0], standard_name='latitude',
                                     var_name='lat', units='degrees_north')
    lon_coord = iris.coords.DimCoord([30.0], standard_name='longitude',
                                     var_name='lon', units='degrees_east')
    coords_specs = [(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
    cube = iris.cube.Cube([[[22.0]]], standard_name='sea_ice_area_fraction',
                          var_name='siconc', units='%',
                          dim_coords_and_dims=coords_specs)
    return iris.cube.CubeList([cube])


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'GFDL-ESM4', 'SImon', 'siconc')
    assert fix == [Siconc(None)]


def test_siconc_fix_metadata(siconc_cubes):
    """Test ``fix_metadata`` for ``cl``."""
    assert len(siconc_cubes) == 1
    siconc_cube = siconc_cubes[0]
    assert siconc_cube.var_name == "siconc"

    # Extract siconc cube
    siconc_cube = siconc_cubes.extract_cube('sea_ice_area_fraction')
    assert not siconc_cube.coords('typesi')

    # Apply fix
    vardef = get_var_info('CMIP6', 'SImon', 'siconc')
    fix = Siconc(vardef)
    fixed_cubes = fix.fix_metadata(siconc_cubes)
    assert len(fixed_cubes) == 1
    fixed_siconc_cube = fixed_cubes.extract_cube(
        'sea_ice_area_fraction')
    fixed_typesi_coord = fixed_siconc_cube.coord('area_type')
    assert fixed_typesi_coord.points is not None
    assert fixed_typesi_coord.bounds is None
    np.testing.assert_equal(fixed_typesi_coord.points,
                            ['siconc'])
    np.testing.assert_equal(fixed_typesi_coord.units,
                            Unit('1'))
