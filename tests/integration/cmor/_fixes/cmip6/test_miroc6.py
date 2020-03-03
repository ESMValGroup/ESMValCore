"""Test fixes for MIROC6."""
import unittest

import pytest
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.cmip6.miroc6 import Cl
from esmvalcore.cmor._fixes.fix import Fix


@pytest.fixture
def cl_cubes():
    """Cubes for ``cl.``."""
    ps_coord = AuxCoord(
        [[1.0]],
        var_name='ps',
        long_name='Surface Air Pressure',
        attributes={'a': 1, 'b': '2'},
    )
    cl_cube = Cube(
        [[0.0]],
        var_name='cl',
        standard_name='cloud_area_fraction_in_atmosphere_layer',
        aux_coords_and_dims=[(ps_coord.copy(), (0, 1))],
    )
    x_cube = Cube([[0.0]],
                  long_name='x',
                  aux_coords_and_dims=[(ps_coord.copy(), (0, 1))])
    cubes = CubeList([cl_cube, x_cube])
    return cubes


def test_get_cl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'MIROC6', 'Amon', 'cl')
    assert fix == [Cl(None)]


@unittest.mock.patch(
    'esmvalcore.cmor._fixes.cmip6.miroc6.BaseCl.fix_metadata',
    autospec=True)
def test_cl_fix_metadata(mock_base_fix_metadata, cl_cubes):
    """Test ``fix_metadata`` for ``cl``."""
    mock_base_fix_metadata.return_value = cl_cubes
    fix = Cl(None)
    fixed_cubes = fix.fix_metadata(cl_cubes)
    mock_base_fix_metadata.assert_called_once_with(fix, cl_cubes)
    assert len(fixed_cubes) == 2
    cl_cube = fixed_cubes.extract_strict(
        'cloud_area_fraction_in_atmosphere_layer')
    ps_coord_cl = cl_cube.coord('Surface Air Pressure')
    assert not ps_coord_cl.attributes
    x_cube = fixed_cubes.extract_strict('x')
    ps_coord_x = x_cube.coord('Surface Air Pressure')
    assert ps_coord_x.attributes == {'a': 1, 'b': '2'}
