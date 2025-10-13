"""Test derivation of ``hurs``."""

import iris
import numpy as np
import pytest
from iris.cube import Cube, CubeList

from esmvalcore.preprocessor._derive import hurs


@pytest.fixture
def cubes():
    """Input cubes for derivation of ``hurs``."""
    time_coord = iris.coords.DimCoord(
        [0.0, 1.0, 2.0, 3.0],
        standard_name="time",
        var_name="time",
        units="days since 1950-01-01 00:00:00",
    )
    lat_coord = iris.coords.DimCoord(
        [45.0],
        standard_name="latitude",
        var_name="lat",
        units="degrees",
    )
    lon_coord = iris.coords.DimCoord(
        [10.0],
        standard_name="longitude",
        var_name="lon",
        units="degrees",
    )

    coord_specs = [(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]

    tdps_cube = Cube(
        [[[279.17]], [[282.73]], [[288.15]], [[288.25]]],
        dim_coords_and_dims=coord_specs,
        standard_name="dew_point_temperature",
        var_name="tdps",
        units="K",
    )
    tas_cube = Cube(
        [[[288.15]], [[288.15]], [[288.15]], [[288.15]]],
        dim_coords_and_dims=coord_specs,
        standard_name="air_temperature",
        var_name="tas",
        units="K",
    )
    return CubeList([tdps_cube, tas_cube])


def test_hurs_calculate(cubes):
    """Test function ``calculate``."""
    derived_var = hurs.DerivedVariable()
    required_vars = derived_var.required("CMIP6")
    expected_required_vars = [
        {"short_name": "tdps"},
        {"short_name": "tas"},
    ]
    assert required_vars == expected_required_vars
    out_cube = derived_var.calculate(cubes)
    assert out_cube.shape == (4, 1, 1)
    assert out_cube.units == "%"
    assert out_cube.coords("time")
    assert out_cube.coords("latitude")
    assert out_cube.coords("longitude")
    np.testing.assert_allclose(
        out_cube.data,
        [[[54.6093]], [[69.7301]], [[100.0]], [[100.0]]],
        rtol=0.00005,
    )
    np.testing.assert_allclose(
        out_cube.coord("time").points,
        [0.0, 1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(out_cube.coord("latitude").points, [45.0])
    np.testing.assert_allclose(out_cube.coord("longitude").points, [10.0])
