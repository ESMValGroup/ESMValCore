"""Test derivation of `qep`."""

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from esmvalcore.preprocessor._derive import qep


@pytest.fixture
def cubes():
    evspsbl_cube = Cube(
        3,
        standard_name="water_evapotranspiration_flux",
        units="kg m-2 s-1",
    )
    pr_cube = Cube(1, standard_name="precipitation_flux", units="kg m-2 s-1")
    return CubeList([evspsbl_cube, pr_cube])


def test_qep_calculation(cubes):
    """Test calculation of `qep`."""
    derived_var = qep.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    np.testing.assert_equal(out_cube.data, 2)
    assert out_cube.units == "kg m-2 s-1"
    assert out_cube.attributes["positive"] == "up"


def test_qep_required():
    """Test function ``required``."""
    derived_var = qep.DerivedVariable()
    output = derived_var.required(None)
    assert output == [
        {"short_name": "evspsbl"},
        {"short_name": "pr"},
    ]
