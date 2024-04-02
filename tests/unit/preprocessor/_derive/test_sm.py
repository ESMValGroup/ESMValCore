import dask.array as da
import iris.coords
import iris.cube
import numpy as np

from esmvalcore.preprocessor._derive.sm import DerivedVariable
from tests import assert_array_equal


def test_sm():

    points = da.arange(0, 4, 2).astype(np.float32)
    bounds = da.asarray([[-1., 1.], [1., 3]])

    depth = iris.coords.AuxCoord(
        points,
        bounds=bounds,
        standard_name='depth',
    )
    cube = iris.cube.Cube(
        da.asarray([0, 998.2]),
        var_name='mrsos',
        aux_coords_and_dims=[
            (depth, 0),
        ],
    )

    result = DerivedVariable.calculate(iris.cube.CubeList([cube]))
    assert result.has_lazy_data()
    assert result.coord('depth').has_lazy_points()
    assert result.coord('depth').has_lazy_bounds()
    assert_array_equal(result.data, np.array([0, 0.5]))
