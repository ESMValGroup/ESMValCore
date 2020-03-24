"""Test FGOALS-s2 fixes."""
import numpy as np
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.cmip5.fgoals_s2 import AllVars
from esmvalcore.cmor.fix import Fix


def test_get_allvars_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP5', 'FGOALS-s2', 'Amon', 'tas')
    assert fix == [AllVars(None)]


LAT_COORD = DimCoord(
    [-20.0, 0.0, 10.0],
    bounds=[[-30.0, -10.0], [-10.0, 5.0], [5.0, 20.0]],
    var_name='lat',
    standard_name='latitude',
)
LAT_COORD_MULT = AuxCoord(
    [[-20.0], [0.0], [10.0]],
    bounds=[[[-30.0, -10.0]], [[-10.0, 5.0]], [[5.0, 20.0]]],
    var_name='lat',
    standard_name='latitude',
)
LAT_COORD_SMALL = DimCoord([0.0],
                           bounds=[-45.0, 45.0],
                           var_name='lat',
                           standard_name='latitude')


def test_allvars_fix_metadata():
    """Test ``fix_metadata`` for all variables."""
    cubes = CubeList([
        Cube([1, 2, 3], dim_coords_and_dims=[(LAT_COORD.copy(), 0)]),
        Cube([[1], [2], [3]],
             aux_coords_and_dims=[(LAT_COORD_MULT.copy(), (0, 1))]),
        Cube([1], dim_coords_and_dims=[(LAT_COORD_SMALL.copy(), 0)]),
        Cube(0.0),
    ])
    fix = AllVars(None)
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 4
    assert fixed_cubes[0].coord('latitude') != LAT_COORD
    np.testing.assert_allclose(fixed_cubes[0].coord('latitude').points,
                               [-20.0, 0.0, 10.0])
    np.testing.assert_allclose(fixed_cubes[0].coord('latitude').bounds,
                               [[-25.0, -10.0], [-10.0, 5.0], [5.0, 20.0]])
    assert fixed_cubes[1].coord('latitude') == LAT_COORD_MULT
    assert fixed_cubes[2].coord('latitude') == LAT_COORD_SMALL
    assert fixed_cubes[3] == Cube(0.0)
