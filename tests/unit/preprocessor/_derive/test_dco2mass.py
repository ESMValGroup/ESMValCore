"""Test derivation of ``dco2mass``."""
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.dco2mass as dco2mass


@pytest.fixture
def cubes():
    """Input cubes for derivation of ``dco2mass``."""
    time_coord = iris.coords.DimCoord(
        [10, 20, 50, 90, 100, 150],
        bounds=[[5, 15], [15, 25], [25, 75], [75, 95], [95, 125], [125, 175]],
        var_name='time',
        standard_name='time',
        long_name='time',
        units='min since 2000-01-01 00:00:00',
    )
    cube = iris.cube.Cube(
        [600.0, 1200.0, 3000.0, 5400.0, 6000.0, 9000.0],
        var_name='co2mass',
        standard_name='atmosphere_mass_of_carbon_dioxide',
        units='kg',
        dim_coords_and_dims=[(time_coord, 0)],
    )
    return iris.cube.CubeList([cube])


def test_dco2mass_calculate(cubes):
    """Test function ``calculate`` for ``dco2mass``."""
    derived_var = dco2mass.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    np.testing.assert_allclose(out_cube.data, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
