"""Test derivation of ``toz``."""
import dask.array as da
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.toz as toz
from .test_co2s import get_coord_spec, get_ps_cube


@pytest.fixture
def masked_cubes():
    """Masked O3 cube."""
    coord_spec = get_coord_spec()
    o3_data = da.ma.masked_less([[[[0.0, -1.0],
                                   [-1.0, -1.0]],
                                  [[1.0, 2.0],
                                   [3.0, -1.0]],
                                  [[2.0, 2.0],
                                   [2.0, 2.0]]]], 0.0)
    o3_cube = iris.cube.Cube(
        o3_data,
        var_name='o3',
        standard_name='mole_fraction_of_ozone_in_air',
        units='1e-9',
        dim_coords_and_dims=coord_spec,
    )
    ps_cube = get_ps_cube()
    return iris.cube.CubeList([o3_cube, ps_cube])


@pytest.fixture
def unmasked_cubes():
    """Unmasked O3 cube."""
    coord_spec = get_coord_spec()
    o3_data = da.array([[[[2.0, 1.0],
                          [0.8, 1.0]],
                         [[1.5, 0.8],
                          [2.0, 3.0]],
                         [[4.0, 1.0],
                          [3.0, 2.0]]]])
    o3_cube = iris.cube.Cube(
        o3_data,
        var_name='o3',
        standard_name='mole_fraction_of_ozone_in_air',
        units='1e-9',
        dim_coords_and_dims=coord_spec,
    )
    ps_cube = get_ps_cube()
    return iris.cube.CubeList([o3_cube, ps_cube])


def test_toz_calculate_masked_cubes(masked_cubes):
    """Test function ``calculate`` with masked cube."""
    derived_var = toz.DerivedVariable()
    out_cube = derived_var.calculate(masked_cubes)
    assert not np.ma.is_masked(out_cube.data)
    np.testing.assert_allclose(out_cube.data,
                               [[[1.2988646378902597, 0.7871906896304607],
                                 [1.6924599827054907, 0.9446288275565529]]])
    assert out_cube.units == 'DU'


def test_toz_calculate_unmasked_cubes(unmasked_cubes):
    """Test function ``calculate`` with unmasked cube."""
    derived_var = toz.DerivedVariable()
    out_cube = derived_var.calculate(unmasked_cubes)
    assert not np.ma.is_masked(out_cube.data)
    np.testing.assert_allclose(out_cube.data,
                               [[[2.65676858, 0.39359534],
                                 [2.04669579, 0.94462883]]])
    assert out_cube.units == 'DU'


@pytest.mark.parametrize('project,out', [
    ('CMIP5', [{'short_name': 'tro3'}, {'short_name': 'ps'}]),
    ('TEST', [{'short_name': 'tro3'}, {'short_name': 'ps'}]),
    ('CMIP6', [{'short_name': 'o3'}, {'short_name': 'ps'}]),
])
def test_toz_required(project, out):
    """Test function ``required``."""
    derived_var = toz.DerivedVariable()
    output = derived_var.required(project)
    assert output == out
