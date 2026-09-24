"""Test derivation of `lwcre`."""

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from esmvalcore.preprocessor._derive import lwp


@pytest.mark.parametrize("special", [True, False])
def test_lwp_calculation(special: bool) -> None:
    """Test calculation of `lwp`."""
    derived_var = lwp.DerivedVariable()
    attrs = {
        "project_id": "CORDEX",
        "model_id": "CLMcom-CCLM4-8-17",
    }
    if special:
        attrs["driving_model_id"] = "MOHC-HadGEM2-ES"
        expected_data = 2
    else:
        expected_data = 1
    cubes = CubeList(
        [
            Cube(
                np.array([2]),
                var_name="clwvi",
                units="kg m-2",
                attributes=attrs,
            ),
            Cube(
                np.array([1]),
                var_name="clivi",
                units="kg m-2",
                attributes=attrs,
            ),
        ],
    )
    result = derived_var.calculate(cubes)
    np.testing.assert_equal(result.data, expected_data)
