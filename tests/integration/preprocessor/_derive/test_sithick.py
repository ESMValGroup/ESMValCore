"""Test derivation of `sispeed`."""

import numpy as np
from iris.cube import Cube, CubeList

from esmvalcore.preprocessor._derive.sithick import DerivedVariable


def test_sispeed_calculation():
    """Test calculation of `sithick`."""

    siconc = Cube(np.full((2, 2), 0.5), 'sea_ice_area_fraction')
    sivol = Cube(np.full((2, 2), 2.), 'sea_ice_thickness')

    derived_var = DerivedVariable()
    sispeed = derived_var.calculate(CubeList((siconc, sivol)))

    assert np.all(
        sispeed.data == np.ones_like(sispeed.data)
    )
