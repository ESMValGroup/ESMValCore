"""Tests for the fixes of FGOALS-g3."""
from unittest import mock

import iris
import numpy as np

from esmvalcore.cmor._fixes.cmip6.fgoals_g3 import Mrsos, Siconc, Tos
from esmvalcore.cmor._fixes.common import OceanFixGrid
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


def test_get_tos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'BCC-CSM2-MR', 'Omon', 'tos')
    assert fix == [Tos(None)]


def test_tos_fix():
    """Test fix for ``tos``."""
    assert issubclass(Tos, OceanFixGrid)


@mock.patch('esmvalcore.cmor._fixes.cmip6.fgoals_g3.OceanFixGrid.fix_metadata',
            autospec=True)
def test_tos_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``tos``."""
    mock_base_fix_metadata.side_effect = lambda x, y: y

    # Create test cube
    lat_coord = iris.coords.AuxCoord([3.14, 1200.0, 6.28],
                                     var_name='lat',
                                     standard_name='latitude')
    lon_coord = iris.coords.AuxCoord([1.0, 2.0, 1e30],
                                     var_name='lon',
                                     standard_name='longitude')
    cube = iris.cube.Cube([1.0, 2.0, 3.0],
                          var_name='tos',
                          standard_name='sea_surface_temperature',
                          aux_coords_and_dims=[(lat_coord, 0), (lon_coord, 0)])
    cubes = iris.cube.CubeList([cube])

    # Apply fix
    vardef = get_var_info('CMIP6', 'Omon', 'tos')
    fix = Tos(vardef)
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes[0]
    np.testing.assert_allclose(
        fixed_cube.coord('latitude').points, [3.14, 0.0, 6.28])
    np.testing.assert_allclose(
        fixed_cube.coord('longitude').points, [1.0, 2.0, 0.0])
    mock_base_fix_metadata.assert_called_once_with(fix, cubes)


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'BCC-CSM2-MR', 'SImon', 'siconc')
    assert fix == [Siconc(None)]


def test_siconc_fix():
    """Test fix for ``siconc``."""
    assert Siconc is Tos


def test_get_mrsos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'FGOALS-g3', 'Lmon', 'mrsos')
    assert fix == [Mrsos(None)]


def test_mrsos_fix():
    """Test fix for ``mrsos``."""
    assert issubclass(Mrsos, Fix)


@mock.patch('esmvalcore.cmor._fixes.cmip6.fgoals_g3.Fix.fix_metadata',
            autospec=True)
def test_mrsos_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``mrsos``."""
    mock_base_fix_metadata.side_effect = lambda x, y: y

    # Create test cube
    lat_coord = iris.coords.AuxCoord([1.0, 2.0, 3.0],
                                     var_name='lat',
                                     standard_name='latitude')
    lat_coord.bounds = [[0.5, 1.5], [-0.5, 0.5], [2.5, 3.5]]
    lon_coord = iris.coords.AuxCoord([1.0, 2.0, 3.0],
                                     var_name='lon',
                                     standard_name='longitude')
    lon_coord.bounds = [[0.5, 1.5], [-0.5, 0.5], [2.5, 3.5]]
    cube = iris.cube.Cube([1.0, 2.0, 3.0],
                          var_name='mrsos',
                          standard_name='mass_content_of_water_in_soil_layer',
                          aux_coords_and_dims=[(lat_coord, 0), (lon_coord, 0)])
    cubes = iris.cube.CubeList([cube])

    # Apply fix
    vardef = get_var_info('CMIP6', 'Lmon', 'mrsos')
    fix = Mrsos(vardef)
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes[0]
    np.testing.assert_allclose(
        fixed_cube.coord('latitude').bounds,
        [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]])
    np.testing.assert_allclose(
        fixed_cube.coord('longitude').bounds,
        [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]])
    mock_base_fix_metadata.assert_called_once_with(fix, cubes)


@mock.patch('esmvalcore.cmor._fixes.cmip6.fgoals_g3.Fix.fix_metadata',
            autospec=True)
def test_mrsos_fix_metadata_2(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``mrsos`` if no fix is necessary."""
    mock_base_fix_metadata.side_effect = lambda x, y: y

    # Create test cube
    lat_coord = iris.coords.AuxCoord([1.0, 2.0, 3.0],
                                     var_name='lat',
                                     standard_name='latitude')
    lat_coord.bounds = [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]]
    lon_coord = iris.coords.AuxCoord([1.0, 2.0, 3.0],
                                     var_name='lon',
                                     standard_name='longitude')
    lon_coord.bounds = [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]]
    cube = iris.cube.Cube([1.0, 2.0, 3.0],
                          var_name='mrsos',
                          standard_name='mass_content_of_water_in_soil_layer',
                          aux_coords_and_dims=[(lat_coord, 0), (lon_coord, 0)])
    cubes = iris.cube.CubeList([cube])

    # Apply fix
    vardef = get_var_info('CMIP6', 'Lmon', 'mrsos')
    fix = Mrsos(vardef)
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes[0]
    np.testing.assert_allclose(
        fixed_cube.coord('latitude').bounds,
        [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]])
    np.testing.assert_allclose(
        fixed_cube.coord('longitude').bounds,
        [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]])
    mock_base_fix_metadata.assert_called_once_with(fix, cubes)
