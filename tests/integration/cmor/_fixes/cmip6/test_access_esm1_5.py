"""Tests for the fixes of ACCESS-ESM1-5."""
import unittest.mock

import iris
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cmip6.access_esm1_5 import Cl, Cli, Clw, Hus, Zg
from esmvalcore.cmor._fixes.common import ClFixHybridHeightCoord
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info

B_POINTS = [
    0.99771648645401, 0.990881502628326, 0.979542553424835,
    0.9637770652771, 0.943695485591888, 0.919438362121582,
    0.891178011894226, 0.859118342399597, 0.823493480682373,
    0.784570515155792, 0.742646217346191, 0.698050200939178,
    0.651142716407776, 0.602314412593842, 0.55198872089386,
    0.500619947910309, 0.44869339466095, 0.39672577381134,
    0.34526526927948, 0.294891387224197, 0.24621507525444,
    0.199878215789795, 0.156554222106934, 0.116947874426842,
    0.0817952379584312, 0.0518637150526047, 0.0279368180781603,
    0.0107164792716503, 0.00130179093685001,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
]
B_BOUNDS = [
    [1, 0.994296252727509],
    [0.994296252727509, 0.985203862190247],
    [0.985203862190247, 0.971644043922424],
    [0.971644043922424, 0.953709840774536],
    [0.953709840774536, 0.931527435779572],
    [0.931527435779572, 0.905253052711487],
    [0.905253052711487, 0.875074565410614],
    [0.875074565410614, 0.84121161699295],
    [0.84121161699295, 0.80391401052475],
    [0.80391401052475, 0.763464510440826],
    [0.763464510440826, 0.720175802707672],
    [0.720175802707672, 0.674392521381378],
    [0.674392521381378, 0.626490533351898],
    [0.626490533351898, 0.576877355575562],
    [0.576877355575562, 0.525990784168243],
    [0.525990784168243, 0.474301367998123],
    [0.474301367998123, 0.422309905290604],
    [0.422309905290604, 0.370548874139786],
    [0.370548874139786, 0.3195820748806],
    [0.3195820748806, 0.270004868507385],
    [0.270004868507385, 0.222443267703056],
    [0.222443267703056, 0.177555426955223],
    [0.177555426955223, 0.136030226945877],
    [0.136030226945877, 0.0985881090164185],
    [0.0985881090164185, 0.0659807845950127],
    [0.0659807845950127, 0.0389823913574219],
    [0.0389823913574219, 0.0183146875351667],
    [0.0183146875351667, 0.00487210927531123],
    [0.00487210927531123, 0],
    [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
    [0, 0], [0, 0],
]


@pytest.fixture
def cl_cubes():
    """``cl`` cubes."""
    b_coord = iris.coords.AuxCoord(np.zeros_like(B_POINTS),
                                   bounds=np.zeros_like(B_BOUNDS),
                                   var_name='b')
    cube = iris.cube.Cube(
        np.ones_like(B_POINTS),
        var_name='cl',
        standard_name='cloud_area_fraction_in_atmosphere_layer',
        units='%',
        aux_coords_and_dims=[(b_coord, 0)],
    )
    return iris.cube.CubeList([cube])


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'ACCESS-ESM1-5', 'Amon', 'cl')
    assert fix == [Cl(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.access_esm1_5.ClFixHybridHeightCoord.'
    'fix_metadata', autospec=True)
def test_cl_fix_metadata(mock_base_fix_metadata, cl_cubes):
    """Test ``fix_metadata`` for ``cl``."""
    mock_base_fix_metadata.side_effect = lambda x, y: y
    fix = Cl(None)
    out_cube = fix.fix_metadata(cl_cubes)[0]
    b_coord = out_cube.coord(var_name='b')
    np.testing.assert_allclose(b_coord.points, B_POINTS)
    np.testing.assert_allclose(b_coord.bounds, B_BOUNDS)


def test_cl_fix():
    """Test fix for ``cl``."""
    assert issubclass(Cl, ClFixHybridHeightCoord)


def test_get_cli_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'ACCESS-ESM1-5', 'Amon', 'cli')
    assert fix == [Cli(None)]


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is Cl


def test_get_clw_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'ACCESS-ESM1-5', 'Amon', 'clw')
    assert fix == [Clw(None)]


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is Cl


@pytest.fixture
def cubes_with_wrong_air_pressure():
    """Cubes with wrong ``air_pressure`` coordinate."""
    air_pressure_coord = iris.coords.DimCoord(
        [1000.09, 600.6, 200.0],
        bounds=[[1200.00001, 800], [800, 400.8], [400.8, 1.9]],
        var_name='plev',
        standard_name='air_pressure',
        units='pa',
    )
    hus_cube = iris.cube.Cube(
        [0.0, 1.0, 2.0],
        var_name='hus',
        dim_coords_and_dims=[(air_pressure_coord, 0)],
    )
    zg_cube = hus_cube.copy()
    zg_cube.var_name = 'zg'
    return iris.cube.CubeList([hus_cube, zg_cube])


def test_get_hus_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'ACCESS-ESM1-5', 'Amon', 'hus')
    assert fix == [Hus(None)]


def test_hus_fix_metadata(cubes_with_wrong_air_pressure):
    """Test ``fix_metadata`` for ``hus``."""
    vardef = get_var_info('CMIP6', 'Amon', 'hus')
    fix = Hus(vardef)
    out_cubes = fix.fix_metadata(cubes_with_wrong_air_pressure)
    assert len(out_cubes) == 2
    hus_cube = out_cubes.extract_cube('hus')
    zg_cube = out_cubes.extract_cube('zg')
    assert hus_cube.var_name == 'hus'
    assert zg_cube.var_name == 'zg'
    np.testing.assert_allclose(hus_cube.coord('air_pressure').points,
                               [1000.0, 601.0, 200.0])
    np.testing.assert_allclose(hus_cube.coord('air_pressure').bounds,
                               [[1200.0, 800.0], [800.0, 401.0], [401.0, 2.0]])
    np.testing.assert_allclose(zg_cube.coord('air_pressure').points,
                               [1000.09, 600.6, 200.0])
    np.testing.assert_allclose(zg_cube.coord('air_pressure').bounds,
                               [[1200.00001, 800], [800, 400.8], [400.8, 1.9]])


def test_get_zg_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'ACCESS-ESM1-5', 'Amon', 'zg')
    assert fix == [Zg(None)]


def test_zg_fix_metadata(cubes_with_wrong_air_pressure):
    """Test ``fix_metadata`` for ``zg``."""
    vardef = get_var_info('CMIP6', 'Amon', 'zg')
    fix = Zg(vardef)
    out_cubes = fix.fix_metadata(cubes_with_wrong_air_pressure)
    assert len(out_cubes) == 2
    hus_cube = out_cubes.extract_cube('hus')
    zg_cube = out_cubes.extract_cube('zg')
    assert hus_cube.var_name == 'hus'
    assert zg_cube.var_name == 'zg'
    np.testing.assert_allclose(hus_cube.coord('air_pressure').points,
                               [1000.09, 600.6, 200.0])
    np.testing.assert_allclose(hus_cube.coord('air_pressure').bounds,
                               [[1200.00001, 800], [800, 400.8], [400.8, 1.9]])
    np.testing.assert_allclose(zg_cube.coord('air_pressure').points,
                               [1000.0, 601.0, 200.0])
    np.testing.assert_allclose(zg_cube.coord('air_pressure').bounds,
                               [[1200.0, 800.0], [800.0, 401.0], [401.0, 2.0]])
