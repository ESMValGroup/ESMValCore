"""Test derivation of ``soz``."""

import dask.array as da
import iris
import numpy as np
import pytest

from esmvalcore.preprocessor._derive import soz

from .test_toz import get_masked_o3_cube, get_masked_o3_hybrid_plevs_cube


def get_o3_cube():
    """Get ``o3`` input cube."""
    o3_cube = get_masked_o3_cube()
    o3_cube.data = da.ma.masked_greater(
        [
            [
                [[500.0, 700.0], [800.0, 900.0]],
                [[1251.0, 1249.0], [1260.0, 1200.0]],
                [[1000.0, 2000.0], [3000.0, 12000.0]],
            ],
        ],
        10000.0,
    )
    o3_cube.units = "1e-10"
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
    o3_cube = o3_cube.collapsed("longitude", iris.analysis.MEAN)
    o3_cube.remove_coord("longitude")
    return iris.cube.CubeList([o3_cube])


@pytest.fixture
def cubes_hybrid_plevs():
    """Input cubes with hybrid pressure levels for derivation of ``soz``."""
    o3_cube = get_masked_o3_hybrid_plevs_cube()
    o3_cube.data = da.ma.masked_greater(
        [
            [
                [[500.0, 700.0], [800.0, 900.0]],
                [[1251.0, 1249.0], [1260.0, 1200.0]],
                [[1000.0, 2000.0], [3000.0, 12000.0]],
            ],
        ],
        10000.0,
    )
    o3_cube.units = "1e-10"
    return iris.cube.CubeList([o3_cube])


def test_soz_calculate(cubes):
    """Test function ``calculate``."""
    derived_var = soz.DerivedVariable()

    out_cube = derived_var.calculate(cubes)

    assert out_cube.units == "m"
    assert out_cube.shape == (1, 2, 2)
    expected_data = np.ma.masked_invalid(
        [
            [
                [29.543266581831194e-5, 110.2066965482645e-5],
                [195.06585289042815e-5, np.nan],
            ],
        ],
    )
    expected_mask = [[[False, False], [False, True]]]
    np.testing.assert_allclose(out_cube.data, expected_data)
    np.testing.assert_allclose(out_cube.data.mask, expected_mask)


def test_soz_calculate_no_lon(cubes_no_lon):
    """Test function ``calculate`` for zonal mean cubes."""
    derived_var = soz.DerivedVariable()

    out_cube = derived_var.calculate(cubes_no_lon)

    assert out_cube.units == "m"
    assert out_cube.shape == (1, 2, 1)
    assert not np.ma.is_masked(out_cube.data)
    np.testing.assert_allclose(
        out_cube.data,
        [[[82.65502241119836e-5], [165.31004482239672e-5]]],
    )


def test_soz_calculate_hybrid_plevs(cubes_hybrid_plevs):
    """Test function ``calculate`` for cubes with hybrid pressure levels."""
    derived_var = soz.DerivedVariable()

    out_cube = derived_var.calculate(cubes_hybrid_plevs)

    assert out_cube.units == "m"
    assert out_cube.shape == (1, 2, 2)
    expected_data = np.ma.masked_invalid(
        [[[np.nan, 32.40347475318536e-5], [44.53039332403313e-5, np.nan]]],
    )
    expected_mask = [[[True, False], [False, True]]]
    np.testing.assert_allclose(out_cube.data, expected_data)
    np.testing.assert_allclose(out_cube.data.mask, expected_mask)


@pytest.mark.parametrize(
    ("project", "out"),
    [
        ("CMIP5", [{"short_name": "tro3"}]),
        ("TEST", [{"short_name": "tro3"}]),
        ("CMIP6", [{"short_name": "o3"}]),
    ],
)
def test_soz_required(project, out):
    """Test function ``required``."""
    derived_var = soz.DerivedVariable()
    output = derived_var.required(project)
    assert output == out
