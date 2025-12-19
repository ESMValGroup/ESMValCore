"""Test derivation of ``pfr``."""

import dask.array as da
import iris
import numpy as np
import pytest

from esmvalcore.preprocessor._derive import pfr

from .test_shared import get_cube


@pytest.fixture
def cubes():
    time_coord = iris.coords.DimCoord([0.0, 1.0, 2.0], standard_name="time")
#    tsl_cube = iris.cube.Cube(
#        [[[20, 10, 0], [10, 10, 10]], [[10, 10, 0], [10, 10, 20]], [[10, 10, 20], [10, 10, 10]]],
#        units="K",
#        standard_name="soil_temperature",
#        var_name="tsl",
#        dim_coords_and_dims=[(time_coord, 0, 1)],
#    )
    tsl_cube = get_cube(
        [[[[270.0]], [[260.0]], [[250.0]]]],
        air_pressure_coord=False,
        depth_coord=True,
        standard_name="soil_temperature",
        var_name="tsl",
        units="K",
    )
    sftlf_cube = get_cube(
        [[[20, 10], [10, 10]], [[10, 10], [10, 10]], [[10, 10], [10, 10]]],
        air_pressure_coord=False,
        depth_coord=False,
        units="%",
        standard_name="land_area_fraction",
        var_name="sftlf",
        dim_coords_and_dims=[(time_coord, 0)],
    )
    mrsos_cube = iris.cube.Cube(
        [[[20, 10], [10, 10]], [[10, 10], [10, 10]], [[10, 10], [10, 10]]],
        air_pressure_coord=False,
        depth_coord=False,
        units="kg m-2",
        standard_name="mass_content_of_water_in_soil_layer",
        var_name="mrsos",
        dim_coords_and_dims=[(time_coord, 0)],
    )
    return iris.cube.CubeList([tsl_cube, sftlf_cube, mrsos_cube])


def test_pfr_calculation(cubes):
    """Test function ``calculate``."""
    derived_var = pfr.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    assert out_cube.units == cf_units.Unit("%")
    out_data = out_cube.data
    expected = np.ma.ones_like(cubes[0].data)
    expected[0][0][0] = 1.0
    np.testing.assert_array_equal(out_data.mask, expected.mask)
    np.testing.assert_array_equal(out_data[0][0][0], expected[0][0][0])


def test_pfr_required():
    """Test function ``required``."""
    derived_var = pfr.DerivedVariable()
    output = derived_var.required(None)
    assert output == [
        {"short_name": "tsl", "mip": "Lmon"},
        {"short_name": "sftlf", "mip": "fx"},
        {"short_name": "mrsos", "mip": "Lmon"},
    ]