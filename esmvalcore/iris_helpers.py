"""Auxiliary functions for :mod:`iris`."""
import warnings
from typing import Any, Dict, List, Sequence

import dask.array as da
import iris
import iris.cube
import iris.util
import numpy as np
from iris import NameConstraint
from iris.cube import Cube
from iris.exceptions import CoordinateMultiDimError

from esmvalcore.exceptions import ESMValCoreDeprecationWarning


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
        dimension. Needs to be 1D.

    Returns
    -------
    iris.cube.Cube
        Transformed input cube with new leading dimension.

    Raises
    ------
    CoordinateMultiDimError
        ``dim_coord`` is not 1D.

    """
    # Only 1D dim_coords are supported
    if dim_coord.ndim > 1:
        raise CoordinateMultiDimError(dim_coord)
    new_shape = (dim_coord.shape[0], *cube.shape)

    # Cache ancillary variables and cell measures (iris.util.new_axis drops
    # those) and determine corresponding dimensions in new cube
    ancillary_variables = []
    for ancillary_variable in cube.ancillary_variables():
        new_dims = tuple(
            d + 1 for d in cube.ancillary_variable_dims(ancillary_variable)
        )
        ancillary_variables.append((ancillary_variable, new_dims))
    cell_measures = []
    for cell_measure in cube.cell_measures():
        new_dims = tuple(d + 1 for d in cube.cell_measure_dims(cell_measure))
        cell_measures.append((cell_measure, new_dims))

    # Transform cube from shape (x, ..., z) to (1, x, ..., z)
    cube = iris.util.new_axis(cube)

    # Create new cube with shape (w, x, ..., z) where w is length of dim_coord
    # and already add ancillary variables and cell measures
    new_data = da.broadcast_to(cube.core_data(), new_shape)
    new_cube = Cube(
        new_data,
        ancillary_variables_and_dims=ancillary_variables,
        cell_measures_and_dims=cell_measures,
    )

    # Add metadata
    # Note: using cube.coord_dims() for determining the positions for the
    # coordinates of the new cube is correct here since cube has the shape (1,
    # x, ..., z) at this stage
    new_cube.metadata = cube.metadata
    new_cube.add_dim_coord(dim_coord, 0)
    for coord in cube.coords(dim_coords=True):
        new_cube.add_dim_coord(coord, cube.coord_dims(coord))
    for coord in cube.coords(dim_coords=False):
        new_cube.add_aux_coord(coord, cube.coord_dims(coord))

    return new_cube


def date2num(date, unit, dtype=np.float64):
    """Convert datetime object into numeric value with requested dtype.

    This is a custom version of :meth:`cf_units.Unit.date2num` that
    guarantees the correct dtype for the return value.

    Arguments
    ---------
    date : :class:`datetime.datetime` or :class:`cftime.datetime`
    unit : :class:`cf_units.Unit`
    dtype : a numpy dtype

    Returns
    -------
    :class:`numpy.ndarray` of type `dtype`
        The return value of ``unit.date2num`` with the requested dtype.
    """
    num = unit.date2num(date)
    try:
        return num.astype(dtype)
    except AttributeError:
        return dtype(num)


def equalize_cube_attributes(cubes: Sequence[Cube], delimiter='|') -> None:
    """Equalize attributes of all given cubes in-place.

    Note
    ----
    This function differs from :func:`iris.util.equalise_attributes` in this
    respect that it does not delete attributes that are not identical but
    rather concatenates them (sorted) using the given ``delimiter``. E.g., the
    attributes ``exp: historical`` and ``exp: ssp585`` end up as ``exp:
    historical|ssp585`` using the default ``delimiter``.

    Parameters
    ----------
    cubes:
        Input cubes whose attributes will be modified in-place.
    delimiter:
        Delimiter that is used to concatenate non-identical attributes.

    """
    if len(cubes) <= 1:
        return

    # Step 1: collect all attribute values in a list
    attributes: Dict[Any, List[Any]] = {}
    for cube in cubes:
        for (attr, val) in cube.attributes.items():
            attributes.setdefault(attr, [])
            attributes[attr].append(val)

    # Step 2: if values are not equal, first convert them to strings (so that
    # set() can be used); then extract unique elements from this list, sort it,
    # and use the delimiter to join all elements to a single string
    final_attributes: Dict[Any, List[Any]] = {}
    for (attr, vals) in attributes.items():
        if _contains_identical_values(vals):
            final_attributes[attr] = vals[0]
        else:
            vals = sorted(list({str(v) for v in vals}))
            final_attributes[attr] = delimiter.join(vals)

    # Step 3: modify the cubes in-place
    for cube in cubes:
        cube.attributes = final_attributes


def _contains_identical_values(sequence: Sequence) -> bool:
    """Check if an iterable contains identical values.

    Note
    ----
    We use :func:`np.array_equal` here since it is very versatile and works
    with all kinds of input types.

    """
    for (idx, val) in enumerate(sequence[:-1]):
        if not np.array_equal(val, sequence[idx + 1]):
            return False
    return True


def var_name_constraint(var_name):
    """:class:`iris.Constraint` using ``var_name``.

    Warning
    -------
    .. deprecated:: 2.6.0
        This function has been deprecated in ESMValCore version 2.6.0 and is
        scheduled for removal in version 2.8.0. Please use the function
        :class:`iris.NameConstraint` with the argument ``var_name`` instead:
        this is an exact replacement.

    Parameters
    ----------
    var_name: str
        ``var_name`` used for the constraint.

    Returns
    -------
    iris.Constraint
        Constraint.

    """
    deprecation_msg = (
        "The function ``var_name_constraint`` has been deprecated in "
        "ESMValCore version 2.6.0 and is scheduled for removal in version "
        "2.8.0. Please use the function ``iris.NameConstraint`` with the "
        "argument ``var_name`` instead: this is an exact replacement."
    )
    warnings.warn(deprecation_msg, ESMValCoreDeprecationWarning)
    return NameConstraint(var_name=var_name)
