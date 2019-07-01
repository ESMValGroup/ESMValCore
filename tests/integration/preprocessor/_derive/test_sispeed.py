"""Test derivation of `sispeed`."""

import numpy as np
import math
from iris.cube import Cube, CubeList
from iris.coords import AuxCoord

from esmvalcore.preprocessor._derive.sispeed import DerivedVariable


def test_sispeed_calculation():
    """Test calculation of `fgco2_grid."""

    siu = Cube(np.ones((2, 2)), 'sea_ice_x_velocity')
    siu.add_aux_coord(AuxCoord(((0, 1), (2, 3)), 'latitude'), (0, 1))
    siu.add_aux_coord(AuxCoord(((0, 1), (2, 3)), 'longitude'), (0, 1))

    siv = Cube(np.ones((2, 2)), 'sea_ice_y_velocity')
    siv.add_aux_coord(AuxCoord(((1, 1), (2, 3)), 'latitude'), (0, 1))
    siv.add_aux_coord(AuxCoord(((1, 1), (2, 3)), 'longitude'), (0, 1))

    derived_var = DerivedVariable()
    sispeed = derived_var.calculate(CubeList((siu, siv)))
    assert np.all(
        sispeed.data == np.full_like(sispeed.data, 1 * math.sqrt(2))
    )
    assert np.all(
        sispeed.coord('latitude').points == siu.coord('latitude').points
    )
    assert np.all(
        sispeed.coord('longitude').points == siu.coord('longitude').points
    )
