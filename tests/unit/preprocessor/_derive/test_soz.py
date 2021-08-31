"""Test derivation of ``soz``."""
import dask.array as da
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.soz as soz

from .test_co2s import get_coord_spec


def get_o3_cube():
    """Get ``o3`` input cube."""
    coord_spec = get_coord_spec()
    o3_data = da.ma.masked_greater([[[[50.0, 70.0],
                                      [80.0, 90.0]],
                                     [[125.0, 124.0],
                                      [126.0, 120.0]],
                                     [[100.0, 200.0],
                                      [300.0, 1200.0]]]], 1000.0)
    o3_cube = iris.cube.Cube(
        o3_data,
        var_name='o3',
        standard_name='mole_fraction_of_ozone_in_air',
        units='1e-9',
        dim_coords_and_dims=coord_spec,
    )
    return o3_cube


@pytest.fixture
def cubes():
    """Input cubes for derivation of ``soz``."""
    o3_cube = get_o3_cube()
    return iris.cube.CubeList([o3_cube])


@pytest.fixture
def cubes_no_lon():
    """Zonal mean input cubes for derivation of ``soz``."""
    o3_cube = get_o3_cube()
    o3_cube = o3_cube.collapsed('longitude', iris.analysis.MEAN)
    o3_cube.remove_coord('longitude')
    return iris.cube.CubeList([o3_cube])


def test_soz_calculate(cubes):
    """Test function ``calculate``."""
    derived_var = soz.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    assert out_cube.shape == (1, 2, 2)
    expected_data = np.ma.masked_invalid(
        [[[29.519650861142278, 110.2066965482645],
          [195.06585289042815, np.nan]]]
    )
    expected_mask = [[[False, False], [False, True]]]
    np.testing.assert_allclose(out_cube.data, expected_data)
    np.testing.assert_allclose(out_cube.data.mask, expected_mask)


def test_soz_calculate_no_lon(cubes_no_lon):
    """Test function ``calculate`` for zonal mean cubes."""
    derived_var = soz.DerivedVariable()
    out_cube = derived_var.calculate(cubes_no_lon)
    assert out_cube.shape == (1, 2, 1)
    assert not np.ma.is_masked(out_cube.data)
    np.testing.assert_allclose(out_cube.data,
                               [[[82.65502241119836], [165.31004482239672]]])


@pytest.mark.parametrize('project,out', [
    ('CMIP5', [{'short_name': 'tro3'}]),
    ('TEST', [{'short_name': 'tro3'}]),
    ('CMIP6', [{'short_name': 'o3'}]),
])
def test_soz_required(project, out):
    """Test function ``required``."""
    derived_var = soz.DerivedVariable()
    output = derived_var.required(project)
    assert output == out
