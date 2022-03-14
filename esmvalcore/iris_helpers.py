"""Auxiliary functions for :mod:`iris`."""
import dask.array as da
import iris
import iris.cube
import iris.util
import numpy as np


def add_leading_dim_to_cube(cube, dim_coord):
    """Add new leading dimension to cube.

    An input cube with shape ``(x, ..., z)`` will be transformed to a cube with
    shape ``(w, x, ..., z)`` where ``w`` is the length of ``dim_coord``. Note
    that the data is broadcasted to the new shape.

    Parameters
    ----------
    cube: iris.cube.Cube
        Input cube.
    dim_coord: iris.coords.DimCoord
        Dimensional coordinate that is used to describe the new leading
        dimension.

    Returns
    -------
    iris.cube.Cube
        Transformed input cube with new leading dimension.

    """
    new_shape = (dim_coord.shape[0], *cube.shape)

    # Transform cube from shape (x, ..., z) to (1, x, ..., z)
    cube = iris.util.new_axis(cube)

    # Create new cube with shape (w, x, ..., z) where w is length of dim_coord
    new_data = da.broadcast_to(cube.core_data(), new_shape)
    new_cube = iris.cube.Cube(new_data)

    # Add metadata
    new_cube.metadata = cube.metadata
    new_cube.add_dim_coord(dim_coord, 0)
    for coord in cube.coords(dim_coords=True):
        new_cube.add_dim_coord(coord, cube.coord_dims(coord))
    for coord in cube.coords(dim_coords=False):
        new_cube.add_aux_coord(coord, cube.coord_dims(coord))
    for cell_measure in cube.cell_measures():
        new_cube.add_cell_measure(
            cell_measure,
            cube.cell_measure_dims(cell_measure),
        )
    for ancillary_variable in cube.ancillary_variables():
        new_cube.add_ancillary_variable(
            ancillary_variable,
            cube.ancillary_variable_dims(ancillary_variable),
        )

    return new_cube


def date2num(date, unit, dtype=np.float64):
    """Convert datetime object into numeric value with requested dtype.

    This is a custom version of :func:`cf_units.Unit.date2num` that
    guarantees the correct dtype for the return value.

    Arguments
    ---------
    date : :class:`datetime.datetime` or :class:`cftime.datetime`
    unit : :class:`cf_units.Unit`
    dtype : a numpy dtype

    Returns
    -------
    :class:`numpy.ndarray` of type `dtype`
        the return value of `unit.date2num` with the requested dtype
    """
    num = unit.date2num(date)
    try:
        return num.astype(dtype)
    except AttributeError:
        return dtype(num)


def var_name_constraint(var_name):
    """:mod:`iris.Constraint` using `var_name` of a :mod:`iris.cube.Cube`."""
    return iris.Constraint(cube_func=lambda c: c.var_name == var_name)
