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

    msftmyz_cube = get_cube([[[100.]]],
                            depth_coord=True,
                            standard_name=msftmyz_name)
    msftyz_cube = get_cube([[[100.]]],
                           depth_coord=False,
                           standard_name=msftyz_name)

    return (iris.cube.CubeList([msftymz_cube]),
        iris.cube.CubeList([msftyz_cube])
    )


def test_amoc_calculation_cmip5(cubes):
    derived_var = amoc.DerivedVariable()
    cmip5_required = derived_var.required("CMIP5")
    # [{'short_name': 'msftmyz', 'mip': 'Omon'}]
    assert "mstftmyz" == cmip5_required[0]["short_name"]
    out_cube = derived_var.calculate(cubes)
    np.testing.assert_allclose(out_cube.data,
                               np.array([[[50.0]]]))
    assert out_cube.attributes['positive'] == 'up'
