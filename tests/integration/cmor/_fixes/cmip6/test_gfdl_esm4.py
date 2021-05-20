"""Tests for the fixes of GFDL-ESM4."""

import iris
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.cmip6.gfdl_esm4 import Fgco2, Omon, Siconc
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
        var_name='lev', standard_name=None, units='m',
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
    fix = Fix.get_fixes('CMIP6', 'GFDL-ESM4', 'Omon', 'thetao')
    assert fix == [Omon(None)]


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


def test_get_fgco2_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'GFDL-ESM4', 'Omon', 'fgco2')
    assert fix == [Fgco2(None), Omon(None)]


def test_fgco2_fix_metadata():
    """Test ``fix_metadata`` for ``fgco2``."""
    vardef = get_var_info('CMIP6', 'Omon', 'fgco2')
    cubes = iris.cube.CubeList([
        iris.cube.Cube(0.0, var_name='fgco2'),
    ])
    fix = Fgco2(vardef)
    out_cubes = fix.fix_metadata(cubes)
    assert out_cubes is cubes
    assert len(out_cubes) == 1
    out_cube = out_cubes[0]

    # Check depth coordinate
    depth_coord = out_cube.coord('depth')
    assert depth_coord.standard_name == 'depth'
    assert depth_coord.var_name == 'depth'
    assert depth_coord.long_name == 'depth'
    assert depth_coord.units == 'm'
    assert depth_coord.attributes == {'positive': 'down'}

    # Check values of depth coordinate
    np.testing.assert_allclose(depth_coord.points, 0.0)
    assert depth_coord.bounds is None
