"""Test various issues we discovered with iris over time."""
import dask.array as da
import iris
import numpy as np
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube


def create_fully_lazy_cube():
    """Create cube with lazy aux coord and aggregate over this dimension."""
    dim_coord = DimCoord(da.arange(10), var_name='time')
    # fully lazy coord points
    aux_coord = AuxCoord(da.arange(10), var_name='year')
    cube = Cube(
        da.arange(10),
        var_name='x',
        dim_coords_and_dims=[(dim_coord, 0)],
        aux_coords_and_dims=[(aux_coord, 0)],
    )

    cube = cube.collapsed('time', iris.analysis.MEAN)

    return cube


def create_regular_cube():
    """Create cube with lazy aux coord and aggregate over this dimension."""
    dim_coord = DimCoord(np.arange(10), var_name='time')
    # fully lazy coord points
    aux_coord = AuxCoord(np.arange(10), var_name='year')
    cube = Cube(
        np.arange(10),
        var_name='x',
        dim_coords_and_dims=[(dim_coord, 0)],
        aux_coords_and_dims=[(aux_coord, 0)],
    )

    cube = cube.collapsed('time', iris.analysis.MEAN)

    return cube


def test_iris_save_with_lazy_coordinate(tmp_path):
    """
    Test saving a cube with fully lazy coords and data.

    Motivated by https://github.com/SciTools/iris/issues/4599
    """
    print("iris version:", iris.__version__)
    cube = create_fully_lazy_cube()
    save_path = tmp_path / 'test_iris_v32.nc'
    iris.save(cube, save_path)
    print("Attempted to load ", save_path)
    loaded_cube = iris.load_cube(save_path.as_posix())
    assert loaded_cube


def test_iris_save_with_regular_coordinate(tmp_path):
    """
    Test saving a cube with numpy array coords and data.

    Motivated by https://github.com/SciTools/iris/issues/4599
    """
    print("iris version:", iris.__version__)
    cube = create_regular_cube()
    save_path = tmp_path / 'test_iris_v32.nc'
    iris.save(cube, save_path)
    print("Attempted to load ", save_path)
    loaded_cube = iris.load_cube(save_path.as_posix())
    assert loaded_cube
