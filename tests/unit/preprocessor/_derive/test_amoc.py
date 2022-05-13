"""Test derivation of `amoc`."""
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.amoc as amoc

from .test_shared import get_cube


@pytest.fixture
def cubes():
    # standard names
    msftmyz_name = 'ocean_meridional_overturning_mass_streamfunction'
    msftyz_name = 'ocean_y_overturning_mass_streamfunction'

    msftmyz_cube = get_cube([[[[100.]], [[100.]], [[100.]]]],
                            air_pressure_coord=False,
                            depth_coord=True,
                            standard_name=msftmyz_name)
    msftyz_cube = get_cube([[[[100.]], [[100.]], [[100.]]]],
                           air_pressure_coord=False,
                           depth_coord=True,
                           standard_name=msftyz_name)
    rando_cube = get_cube([[[[100.]], [[100.]], [[100.]]]],
                          air_pressure_coord=False,
                          depth_coord=True,
                          standard_name="air_temperature")
    msftmyz_cube.coord("latitude").points = np.array([26.0])
    msftyz_cube.coord("latitude").points = np.array([26.0])
    msftyz_cube.coord("latitude").standard_name = "grid_latitude"

    return \
        iris.cube.CubeList([msftmyz_cube]), \
        iris.cube.CubeList([msftyz_cube]), \
        iris.cube.CubeList([rando_cube])


def test_amoc_preamble(cubes):
    derived_var = amoc.DerivedVariable()

    cmip5_required = derived_var.required("CMIP5")
    assert "msftmyz" == cmip5_required[0]["short_name"]
    cmip6_required = derived_var.required("CMIP6")
    assert "msftyz" == cmip6_required[0]["short_name"]

    cmip5_cubes = cubes[0]
    cmip6_cubes = cubes[1]
    rando_cubes = cubes[2]

    with pytest.raises(ValueError) as verr:
        derived_var.calculate(cmip5_cubes)
        assert "doesn't contain Atlantic Region" in verr
    with pytest.raises(ValueError) as verr:
        derived_var.calculate(cmip6_cubes)
        assert "doesn't contain Atlantic Region" in verr
    with pytest.raises(iris.exceptions.ConstraintMismatchError) as verr:
        derived_var.calculate(rando_cubes)
        assert "standard names could not be found" in verr
