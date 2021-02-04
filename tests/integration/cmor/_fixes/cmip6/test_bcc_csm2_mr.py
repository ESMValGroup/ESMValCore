"""Test fixes for BCC-CSM2-MR."""
import unittest.mock

import iris

from esmvalcore.cmor._fixes.cmip6.bcc_csm2_mr import (Cl, Cli,
                                                      Clw, Tos, Siconc)
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor.table import get_var_info


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'BCC-CSM2-MR', 'Amon', 'cl')
    assert fix == [Cl(None)]


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'BCC-CSM2-MR', 'Amon', 'cli')
    assert fix == [Cli(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is ClFixHybridPressureCoord


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'BCC-CSM2-MR', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is ClFixHybridPressureCoord


def test_get_tos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'BCC-CSM2-MR', 'Omon', 'tos')
    assert fix == [Tos(None)]


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'BCC-CSM2-MR', 'Omon', 'siconc')
    assert fix == [Siconc(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.bcc_csm2_mr.BaseTos.fix_data',
    autospec=True)
def test_tos_fix_data(mock_base_fix_data):
    """Test ``fix_data`` for ``tos``."""
    fix = Tos(None)
    fix.fix_data('cubes')
    mock_base_fix_data.assert_called_once_with(fix, 'cubes')


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.bcc_csm2_mr.BaseTos.fix_data',
    autospec=True)
def test_siconc_fix_data(mock_base_fix_data):
    """Test ``fix_data`` for ``siconc``."""
    fix = Siconc(None)
    fix.fix_data('cubes')
    mock_base_fix_data.assert_called_once_with(fix, 'cubes')


def test_tos_fix_metadata():
    """Test ``fix_metadata`` for ``tos``."""
    grid_lat = iris.coords.DimCoord([1.0],
                                    var_name='lat',
                                    standard_name='latitude',
                                    long_name='latitude',
                                    units='degrees_north',
                                    attributes={'1D': '1'})
    grid_lon = iris.coords.DimCoord([1.0],
                                    var_name='lon',
                                    standard_name='longitude',
                                    long_name='longitude',
                                    units='degrees_east',
                                    circular=True,
                                    attributes={'1D': '1'})
    latitude = iris.coords.AuxCoord([[0.0]],
                                    var_name='lat',
                                    standard_name='latitude',
                                    long_name='latitude',
                                    units='degrees_north')
    longitude = iris.coords.AuxCoord([[0]],
                                     var_name='lon',
                                     standard_name='longitude',
                                     long_name='longitude',
                                     units='degrees_east')
    cube = iris.cube.Cube(
        [[[0.0]]],
        var_name='tos',
        long_name='sea_surface_temperature',
        dim_coords_and_dims=[(grid_lat.copy(), 1), (grid_lon.copy(), 2)],
        aux_coords_and_dims=[(latitude.copy(), (1, 2)),
                             (longitude.copy(), (1, 2))],
    )
    cubes = iris.cube.CubeList([cube, iris.cube.Cube(0.0)])
    vardef = get_var_info('CMIP6', 'Omon', 'tos')
    fix = Tos(vardef)
    fixed_cubes = fix.fix_metadata(cubes)
    tos_cube = fixed_cubes.extract_strict('sea_surface_temperature')

    # No duplicates anymore
    assert len(tos_cube.coords('latitude')) == 1
    assert len(tos_cube.coords('longitude')) == 1

    # Latitude
    grid_lat = tos_cube.coord('grid_latitude')
    assert grid_lat.var_name == 'i'
    assert grid_lat.long_name == 'grid_latitude'
    assert grid_lat.standard_name is None
    assert grid_lat.units == '1'

    # Longitude
    grid_lon = tos_cube.coord('grid_longitude')
    assert grid_lon.var_name == 'j'
    assert grid_lon.long_name == 'grid_longitude'
    assert grid_lon.standard_name is None
    assert grid_lon.units == '1'
    assert not grid_lon.circular


def test_siconc_fix_metadata():
    """Test ``fix_metadata`` for ``tos``."""
    grid_lat = iris.coords.DimCoord([1.0],
                                    var_name='lat',
                                    standard_name='latitude',
                                    long_name='latitude',
                                    units='degrees_north',
                                    attributes={'1D': '1'})
    grid_lon = iris.coords.DimCoord([1.0],
                                    var_name='lon',
                                    standard_name='longitude',
                                    long_name='longitude',
                                    units='degrees_east',
                                    circular=True,
                                    attributes={'1D': '1'})
    latitude = iris.coords.AuxCoord([[0.0]],
                                    var_name='lat',
                                    standard_name='latitude',
                                    long_name='latitude',
                                    units='degrees_north')
    longitude = iris.coords.AuxCoord([[0]],
                                     var_name='lon',
                                     standard_name='longitude',
                                     long_name='longitude',
                                     units='degrees_east')

    cube = iris.cube.Cube(
        [[[0.0]]],
        var_name='siconc',
        long_name='sea_ice_area_fraction',
        dim_coords_and_dims=[(grid_lat.copy(), 1), (grid_lon.copy(), 2)],
        aux_coords_and_dims=[(latitude.copy(), (1, 2)),
                             (longitude.copy(), (1, 2))],
    )
    cubes = iris.cube.CubeList([cube, iris.cube.Cube(0.0)])
    vardef = get_var_info('CMIP6', 'SImon', 'siconc')
    fix = Siconc(vardef)
    fixed_cubes = fix.fix_metadata(cubes)
    siconc_cube = fixed_cubes.extract_strict('sea_ice_area_fraction')

    # No duplicates anymore
    assert len(siconc_cube.coords('latitude')) == 1
    assert len(siconc_cube.coords('longitude')) == 1

    # Latitude
    grid_lat = siconc_cube.coord('grid_latitude')
    assert grid_lat.var_name == 'i'
    assert grid_lat.long_name == 'grid_latitude'
    assert grid_lat.standard_name is None
    assert grid_lat.units == '1'

    # Longitude
    grid_lon = siconc_cube.coord('grid_longitude')
    assert grid_lon.var_name == 'j'
    assert grid_lon.long_name == 'grid_longitude'
    assert grid_lon.standard_name is None
    assert grid_lon.units == '1'
    assert not grid_lon.circular
