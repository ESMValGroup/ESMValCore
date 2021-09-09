"""Test fixes for MPI-ESM1-2-HR."""
import iris

from esmvalcore.cmor._fixes.cmip6.mpi_esm1_2_hr import AllVars, Cl, Cli, Clw
from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor._fixes.fix import Fix


def test_get_allvars_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('CMIP6', 'MPI-ESM1-2-HR', 'Amon', 'tas')
    assert AllVars(None) in fixes
    assert len([fix for fix in fixes if isinstance(fix, AllVars)]) == 1


def test_allvars_r2i1p1f1():
    lat_coord1 = iris.coords.DimCoord(
        [-86.49036676628116],
        var_name='lat',
        standard_name='latitude',
        units='degrees',
    )
    lat_coord2 = iris.coords.DimCoord(
        [-86.49036676628118],
        var_name='lat',
        standard_name='latitude',
        units='degrees',
    )

    cube1 = iris.cube.Cube([0])
    cube1.attributes['variant_label'] = 'r2i1p1f1'
    cube1.add_dim_coord(lat_coord1, 0)

    cube2 = iris.cube.Cube([0])
    cube2.attributes['variant_label'] = 'r2i1p1f1'
    cube2.add_dim_coord(lat_coord2, 0)

    fix = AllVars(None)
    fixed_cubes = fix.fix_metadata([cube1, cube2])

    assert fixed_cubes[0].coord('latitude').points[0] == -86.49036676628
    assert fixed_cubes[1].coord('latitude').points[0] == -86.49036676628


def test_get_cl_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('CMIP6', 'MPI-ESM1-2-HR', 'Amon', 'cl')
    assert Cl(None) in fixes


def test_cl_fix():
    """Test fix for ``cl``."""
    assert Cl is ClFixHybridPressureCoord


def test_get_cli_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('CMIP6', 'MPI-ESM1-2-HR', 'Amon', 'cli')
    assert Cli(None) in fixes


def test_cli_fix():
    """Test fix for ``cli``."""
    assert Cli is ClFixHybridPressureCoord


def test_get_clw_fix():
    """Test getting of fix."""
    fixes = Fix.get_fixes('CMIP6', 'MPI-ESM1-2-HR', 'Amon', 'clw')
    assert Clw(None) in fixes


def test_clw_fix():
    """Test fix for ``clw``."""
    assert Clw is ClFixHybridPressureCoord
