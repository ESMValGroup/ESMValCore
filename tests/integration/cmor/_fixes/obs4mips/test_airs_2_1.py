"""Test AIRS-2-1 fixes."""
import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.obs4mips.airs_2_1 import AllVars
from esmvalcore.cmor.fix import Fix


def get_air_pressure_coord(points, units):
    """Get ``air_pressure`` coordinate."""
    return DimCoord(points, var_name='plev', standard_name='air_pressure',
                    long_name='pressure', units=units)


def test_get_allvars_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('obs4MIPs', 'AIRS-2-1', 'Amon', 'cl')
    assert fix == [AllVars(None)]


def test_allvars_fix_no_air_pressure():
    """Test fix for all variables."""
    cubes = CubeList([Cube(0.0, var_name='cl')])
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes.copy())

    assert len(out_cubes) == 1
    assert out_cubes[0] == cubes[0]


def test_allvars_fix_correct_air_pressure_pa():
    """Test fix for all variables."""
    air_pressure_coord = get_air_pressure_coord([100000.0, 80000.0], 'Pa')
    cube = Cube([0.0, 1.0], var_name='cl',
                dim_coords_and_dims=[(air_pressure_coord, 0)])
    cubes = CubeList([cube])
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes.copy())

    assert len(out_cubes) == 1
    assert out_cubes[0] == cubes[0]
    assert out_cubes[0].coord('air_pressure').units == 'Pa'
    np.testing.assert_allclose(out_cubes[0].coord('air_pressure').points,
                               [100000.0, 80000.0])


def test_allvars_fix_correct_air_pressure_hpa():
    """Test fix for all variables."""
    air_pressure_coord = get_air_pressure_coord([1000.0, 800.0], 'hPa')
    cube = Cube([0.0, 1.0], var_name='cl',
                dim_coords_and_dims=[(air_pressure_coord, 0)])
    cubes = CubeList([cube])
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes.copy())

    assert len(out_cubes) == 1
    assert out_cubes[0] == cubes[0]
    assert out_cubes[0].coord('air_pressure').units == 'hPa'
    np.testing.assert_allclose(out_cubes[0].coord('air_pressure').points,
                               [1000.0, 800.0])


def test_allvars_fix_incorrect_air_pressure():
    """Test fix for all variables."""
    air_pressure_coord = get_air_pressure_coord([100000.0, 80000.0], 'hPa')
    cube = Cube([0.0, 1.0], var_name='cl',
                dim_coords_and_dims=[(air_pressure_coord, 0)])
    cubes = CubeList([cube])
    fix = AllVars(None)
    out_cubes = fix.fix_metadata(cubes.copy())

    assert len(out_cubes) == 1
    assert out_cubes[0] != cubes[0]
    assert out_cubes[0].coord('air_pressure').units == 'Pa'
    np.testing.assert_allclose(out_cubes[0].coord('air_pressure').points,
                               [100000.0, 80000.0])
