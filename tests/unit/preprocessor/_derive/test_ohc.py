"""Test derivation of `ohc`."""
import cf_units
import iris
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.ohc as ohc


@pytest.fixture
def cubes():
    volcello_name = 'ocean_volume'
    thetao_name = 'sea_water_potential_temperature'
    volcello_cube = iris.cube.Cube([[1.0, 1.2], [0.8, 0.2]],
                                   units='m3',
                                   standard_name=volcello_name,
                                   var_name='volcello')
    time_coord = iris.coords.DimCoord([0., 1., 2.],
                                      standard_name='time')
    thetao_cube = iris.cube.Cube([[[1.0, 1.0], [1.0, 1.0]],
                                  [[1.0, 1.0], [1.0, 1.0]],
                                  [[1.0, 1.0], [1.0, 1.0]]],
                                 units='K',
                                 standard_name=thetao_name,
                                 var_name='thetao',
                                 dim_coords_and_dims=[(time_coord, 0)])
    return iris.cube.CubeList([volcello_cube, thetao_cube])


def test_ohc_calculation(cubes):
    derived_var = ohc.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    assert out_cube.units == cf_units.Unit('J')
    out_data = out_cube.data
    val = ohc.RHO_CP.points[0]
    volcello_data = np.broadcast_to(cubes[0].data*val,
                                    out_data.shape)
    np.testing.assert_allclose(out_data, volcello_data)
