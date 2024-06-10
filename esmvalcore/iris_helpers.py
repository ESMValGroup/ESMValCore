"""Auxiliary functions for :mod:`iris`."""
from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Sequence

import dask.array as da
import iris
import iris.cube
import iris.util
import numpy as np
from iris.coords import Coord
from iris.cube import Cube
from iris.exceptions import CoordinateMultiDimError, CoordinateNotFoundError

from esmvalcore.typing import NetCDFAttr


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


def merge_cube_attributes(
    cubes: Sequence[Cube],
    delimiter: str = ' ',
) -> None:
    """Merge attributes of all given cubes in-place.

    After this operation, the attributes of all given cubes are equal. This is
    useful for operations that combine cubes, such as
    :meth:`iris.cube.CubeList.merge_cube` or
    :meth:`iris.cube.CubeList.concatenate_cube`.

    Note
    ----
    This function differs from :func:`iris.util.equalise_attributes` in this
    respect that it does not delete attributes that are not identical but
    rather concatenates them (sorted) using the given ``delimiter``. E.g., the
    attributes ``exp: historical`` and ``exp: ssp585`` end up as ``exp:
    historical ssp585`` using the default ``delimiter = ' '``.

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
    attributes: Dict[str, List[NetCDFAttr]] = {}
    for cube in cubes:
        for (attr, val) in cube.attributes.items():
            attributes.setdefault(attr, [])
            attributes[attr].append(val)

    # Step 2: use the first cube in which an attribute occurs to decide if an
    # attribute is global or local.
    final_attributes = iris.cube.CubeAttrsDict()
    for cube in cubes:
        for attr, value in cube.attributes.locals.items():
            if attr not in final_attributes:
                final_attributes.locals[attr] = value
        for attr, value in cube.attributes.globals.items():
            if attr not in final_attributes:
                final_attributes.globals[attr] = value

    # Step 3: if values are not equal, first convert them to strings (so that
    # set() can be used); then extract unique elements from this list, sort it,
    # and use the delimiter to join all elements to a single string.
    for (attr, vals) in attributes.items():
        set_of_str = sorted({str(v) for v in vals})
        if len(set_of_str) == 1:
            final_attributes[attr] = vals[0]
        else:
            final_attributes[attr] = delimiter.join(set_of_str)

    # Step 4: modify the cubes in-place
    for cube in cubes:
        cube.attributes = final_attributes


def _rechunk(
    array: da.core.Array,
    complete_dims: list[int],
    remaining_dims: int | Literal['auto'],
) -> da.core.Array:
    """Rechunk a given array so that it is not chunked along given dims."""
    new_chunks: list[str | int] = [remaining_dims] * array.ndim
    for dim in complete_dims:
        new_chunks[dim] = -1
    return array.rechunk(new_chunks)


def _rechunk_dim_metadata(
    cube: Cube,
    complete_dims: Iterable[int],
    remaining_dims: int | Literal['auto'] = 'auto',
) -> None:
    """Rechunk dimensional metadata of a cube (in-place)."""
    # Non-dimensional coords that span complete_dims
    # Note: dimensional coords are always realized (i.e., numpy arrays), so no
    # chunking is necessary
    for coord in cube.coords(dim_coords=False):
        dims = cube.coord_dims(coord)
        complete_dims_ = [dims.index(d) for d in complete_dims if d in dims]
        if complete_dims_:
            if coord.has_lazy_points():
                coord.points = _rechunk(
                    coord.lazy_points(), complete_dims_, remaining_dims
                )
            if coord.has_bounds() and coord.has_lazy_bounds():
                coord.bounds = _rechunk(
                    coord.lazy_bounds(), complete_dims_, remaining_dims
                )

    # Rechunk cell measures that span complete_dims
    for measure in cube.cell_measures():
        dims = cube.cell_measure_dims(measure)
        complete_dims_ = [dims.index(d) for d in complete_dims if d in dims]
        if complete_dims_ and measure.has_lazy_data():
            measure.data = _rechunk(
                measure.lazy_data(), complete_dims_, remaining_dims
            )

    # Rechunk ancillary variables that span complete_dims
    for anc_var in cube.ancillary_variables():
        dims = cube.ancillary_variable_dims(anc_var)
        complete_dims_ = [dims.index(d) for d in complete_dims if d in dims]
        if complete_dims_ and anc_var.has_lazy_data():
            anc_var.data = _rechunk(
                anc_var.lazy_data(), complete_dims_, remaining_dims
            )


def rechunk_cube(
    cube: Cube,
    complete_coords: Iterable[Coord | str],
    remaining_dims: int | Literal['auto'] = 'auto',
) -> Cube:
    """Rechunk cube so that it is not chunked along given dimensions.

    This will rechunk the cube's data, but also all non-dimensional
    coordinates, cell measures, and ancillary variables that span at least one
    of the given dimensions.

    Note
    ----
    This will only rechunk `dask` arrays. `numpy` arrays are not changed.

    Parameters
    ----------
    cube:
        Input cube.
    complete_coords:
        (Names of) coordinates along which the output cubes should not be
        chunked.
    remaining_dims:
        Chunksize of the remaining dimensions.

    Returns
    -------
    Cube
        Rechunked cube. This will always be a copy of the input cube.

    """
    cube = cube.copy()  # do not modify input cube

    complete_dims = []
    for coord in complete_coords:
        coord = cube.coord(coord)
        complete_dims.extend(cube.coord_dims(coord))
    complete_dims = list(set(complete_dims))

    # Rechunk data
    if cube.has_lazy_data():
        cube.data = _rechunk(cube.lazy_data(), complete_dims, remaining_dims)

    # Rechunk dimensional metadata
    _rechunk_dim_metadata(cube, complete_dims, remaining_dims=remaining_dims)

    return cube


def has_regular_grid(cube: Cube) -> bool:
    """Check if a cube has a regular grid.

    "Regular" refers to a rectilinear grid with 1D latitude and 1D longitude
    coordinates orthogonal to each other.

    Parameters
    ----------
    cube:
        Cube to be checked.

    Returns
    -------
    bool
        ``True`` if input cube has a regular grid, else ``False``.

    """
    try:
        lat = cube.coord('latitude')
        lon = cube.coord('longitude')
    except CoordinateNotFoundError:
        return False
    if lat.ndim != 1 or lon.ndim != 1:
        return False
    if cube.coord_dims(lat) == cube.coord_dims(lon):
        return False
    return True


def has_irregular_grid(cube: Cube) -> bool:
    """Check if a cube has an irregular grid.

    "Irregular" refers to a general curvilinear grid with 2D latitude and 2D
    longitude coordinates with common dimensions.

    Parameters
    ----------
    cube:
        Cube to be checked.

    Returns
    -------
    bool
        ``True`` if input cube has an irregular grid, else ``False``.

    """
    try:
        lat = cube.coord('latitude')
        lon = cube.coord('longitude')
    except CoordinateNotFoundError:
        return False
    if lat.ndim == 2 and lon.ndim == 2:
        return True
    return False


def has_unstructured_grid(cube: Cube) -> bool:
    """Check if a cube has an unstructured grid.

    "Unstructured" refers to a grid with 1D latitude and 1D longitude
    coordinates with common dimensions (i.e., a simple list of points).

    Parameters
    ----------
    cube:
        Cube to be checked.

    Returns
    -------
    bool
        ``True`` if input cube has an unstructured grid, else ``False``.

    """
    try:
        lat = cube.coord('latitude')
        lon = cube.coord('longitude')
    except CoordinateNotFoundError:
        return False
    if lat.ndim != 1 or lon.ndim != 1:
        return False
    if cube.coord_dims(lat) != cube.coord_dims(lon):
        return False
    return True
