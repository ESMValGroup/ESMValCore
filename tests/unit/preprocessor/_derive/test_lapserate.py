"""Test derivation of ``lapserate``."""

import iris
import numpy as np
import pytest

from esmvalcore.preprocessor._derive import lapserate

from .test_shared import get_cube


@pytest.fixture
def cubes():
    """Input cubes for derivation of ``lapserate``."""
    ta_cube = get_cube(
        [[[[270.0]], [[260.0]]]],
        air_pressure_coord=True,
        standard_name="air_temperature",
        var_name="ta",
        units="K",
    )
    zg_cube = get_cube(
        [[[[1000.0]], [[2000.0]]]],
        air_pressure_coord=True,
        standard_name="geopotential_height",
        var_name="zg",
        units="m",
    )
    return iris.cube.CubeList(
        [ta_cube, zg_cube],
    )


def test_lapserate_calculate(cubes):
    """Test function ``calculate``."""
    derived_var = lapserate.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    assert out_cube.shape == (1, 2, 1, 1)
    assert out_cube.units == "K km-1"
    assert out_cube.coords("time")
    assert out_cube.coords("air_pressure")
    assert out_cube.coords("latitude")
    assert out_cube.coords("longitude")
    np.testing.assert_allclose(
        out_cube.data,
        [[[[10.0]], [[10.0]]]],
    )


def test_lapserate_required():
    """Test function ``required``."""
    derived_var = lapserate.DerivedVariable()
    output = derived_var.required(None)
    assert output == [
        {"short_name": "ta"},
        {"short_name": "zg"},
    ]
