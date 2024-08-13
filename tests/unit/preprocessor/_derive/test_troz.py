"""Test derivation of ``troz``."""
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.troz as troz

from .test_toz import get_masked_o3_cube, get_ps_cube


def get_o3_cube():
    """Get ``o3`` input cube."""
    o3_cube = get_masked_o3_cube()
    o3_cube.data = [[
        [[50.0, 70.0], [80.0, 90.0]],
        [[70.0, 90.0], [100.0, 110.0]],
        [[130, 140.0], [150.0, 160.0]],
    ]]
    o3_cube.units = '1e-9'
    return o3_cube


@pytest.fixture
def cubes():
    """Input cubes for derivation of ``troz``."""
    o3_cube = get_o3_cube()
    ps_cube = get_ps_cube()
    ps_cube.data = [[[101300.0, 101300.0], [101300.0, 101300.0]]]
    return iris.cube.CubeList([o3_cube, ps_cube])


@pytest.fixture
def cubes_no_lon():
    """Zonal mean input cubes for derivation of ``troz``."""
    o3_cube = get_o3_cube()
    o3_cube = o3_cube.collapsed('longitude', iris.analysis.MEAN)
    o3_cube.remove_coord('longitude')
    ps_cube = get_ps_cube()
    ps_cube.data = [[[101300.0, 101300.0], [101300.0, 101300.0]]]
    return iris.cube.CubeList([o3_cube, ps_cube])


def test_troz_calculate(cubes):
    """Test function ``calculate``."""
    derived_var = troz.DerivedVariable()

    out_cube = derived_var.calculate(cubes)

    assert out_cube.units == 'm'
    assert out_cube.shape == (1, 2, 2)
    assert not np.ma.is_masked(out_cube.data)
    expected_data = [[
            [16.255487740869038e-5, 21.1833014579557e-5],
            [23.647208316499057e-5, 26.111115175042404e-5],
    ]]
    np.testing.assert_allclose(out_cube.data, expected_data)


def test_troz_calculate_no_lon(cubes_no_lon):
    """Test function ``calculate`` for zonal mean cubes."""
    derived_var = troz.DerivedVariable()

    out_cube = derived_var.calculate(cubes_no_lon)

    assert out_cube.units == 'm'
    assert out_cube.shape == (1, 2, 1)
    assert not np.ma.is_masked(out_cube.data)
    np.testing.assert_allclose(
        out_cube.data, [[[18.71939459941238e-5], [24.87916174577073e-5]]]
    )


@pytest.mark.parametrize('project,out', [
    ('CMIP5', [{'short_name': 'tro3'}, {'short_name': 'ps'}]),
    ('TEST', [{'short_name': 'tro3'}, {'short_name': 'ps'}]),
    ('CMIP6', [{'short_name': 'o3'}, {'short_name': 'ps', 'mip': 'Amon'}]),
])
def test_toz_required(project, out):
    """Test function ``required``."""
    derived_var = troz.DerivedVariable()
    output = derived_var.required(project)
    assert output == out
