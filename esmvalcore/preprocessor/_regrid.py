"""Horizontal and vertical regridding module."""

import importlib
import logging
import os
import re
from copy import deepcopy
from decimal import Decimal
from typing import Dict

import iris
import numpy as np
import stratify
from dask import array as da
from geopy.geocoders import Nominatim
from iris.analysis import AreaWeighted, Linear, Nearest, UnstructuredNearest
from iris.util import broadcast_to_shape

from ..cmor._fixes.shared import add_altitude_from_plev, add_plev_from_altitude
from ..cmor.fix import fix_file, fix_metadata
from ..cmor.table import CMOR_TABLES
from ._ancillary_vars import add_ancillary_variable, add_cell_measure
from ._io import GLOBAL_FILL_VALUE, concatenate_callback, load
from ._regrid_esmpy import ESMF_REGRID_METHODS
from ._regrid_esmpy import regrid as esmpy_regrid

logger = logging.getLogger(__name__)

# Regular expression to parse a "MxN" cell-specification.
_CELL_SPEC = re.compile(
    r'''\A
        \s*(?P<dlon>\d+(\.\d+)?)\s*
        x
        \s*(?P<dlat>\d+(\.\d+)?)\s*
        \Z
     ''', re.IGNORECASE | re.VERBOSE)

# Default fill-value.
_MDI = 1e+20

# Stock cube - global grid extents (degrees).
_LAT_MIN = -90.0
_LAT_MAX = 90.0
_LAT_RANGE = _LAT_MAX - _LAT_MIN
_LON_MIN = 0.0
_LON_MAX = 360.0
_LON_RANGE = _LON_MAX - _LON_MIN

# A cached stock of standard horizontal target grids.
_CACHE: Dict[str, iris.cube.Cube] = dict()

# Supported point interpolation schemes.
POINT_INTERPOLATION_SCHEMES = {
    'linear': Linear(extrapolation_mode='mask'),
    'nearest': Nearest(extrapolation_mode='mask'),
}

# Supported horizontal regridding schemes.
HORIZONTAL_SCHEMES = {
    'linear': Linear(extrapolation_mode='mask'),
    'linear_extrapolate': Linear(extrapolation_mode='extrapolate'),
    'nearest': Nearest(extrapolation_mode='mask'),
    'area_weighted': AreaWeighted(),
    'unstructured_nearest': UnstructuredNearest(),
}

# Supported vertical interpolation schemes.
VERTICAL_SCHEMES = (
    'linear',
    'nearest',
    'linear_extrapolate',
    'nearest_extrapolate',
)


def parse_cell_spec(spec):
    """Parse an MxN cell specification string.

    Parameters
    ----------
    spec: str
        ``MxN`` degree cell-specification for the global grid.

    Returns
    -------
    tuple
        tuple of (float, float) of parsed (lon, lat)

    Raises
    ------
    ValueError
        if the MxN cell specification is malformed.
    ValueError
        invalid longitude and latitude delta in cell specification.
    """
    cell_match = _CELL_SPEC.match(spec)
    if cell_match is None:
        emsg = 'Invalid MxN cell specification for grid, got {!r}.'
        raise ValueError(emsg.format(spec))

    cell_group = cell_match.groupdict()
    dlon = float(cell_group['dlon'])
    dlat = float(cell_group['dlat'])

    if (np.trunc(_LON_RANGE / dlon) * dlon) != _LON_RANGE:
        emsg = ('Invalid longitude delta in MxN cell specification '
                'for grid, got {!r}.')
        raise ValueError(emsg.format(dlon))

    if (np.trunc(_LAT_RANGE / dlat) * dlat) != _LAT_RANGE:
        emsg = ('Invalid latitude delta in MxN cell specification '
                'for grid, got {!r}.')
        raise ValueError(emsg.format(dlat))

    return dlon, dlat


def _generate_cube_from_dimcoords(latdata, londata, circular: bool = False):
    """Generate cube from lat/lon points.

    Parameters
    ----------
    latdata : np.ndarray
        List of latitudes.
    londata : np.ndarray
        List of longitudes.
    circular : bool
        Wrap longitudes around the full great circle. Bounds will not be
        generated for circular coordinates.

    Returns
    -------
    :class:`~iris.cube.Cube`
    """
    lats = iris.coords.DimCoord(latdata,
                                standard_name='latitude',
                                units='degrees_north',
                                var_name='lat',
                                circular=circular)

    lons = iris.coords.DimCoord(londata,
                                standard_name='longitude',
                                units='degrees_east',
                                var_name='lon',
                                circular=circular)

    if not circular:
        # cannot guess bounds for wrapped coordinates
        lats.guess_bounds()
        lons.guess_bounds()

    # Construct the resultant stock cube, with dummy data.
    shape = (latdata.size, londata.size)
    dummy = np.empty(shape, dtype=np.dtype('int8'))
    coords_spec = [(lats, 0), (lons, 1)]
    cube = iris.cube.Cube(dummy, dim_coords_and_dims=coords_spec)

    return cube


def _global_stock_cube(spec, lat_offset=True, lon_offset=True):
    """Create a stock cube.

    Create a global cube with M degree-east by N degree-north regular grid
    cells.

    The longitude range is from 0 to 360 degrees. The latitude range is from
    -90 to 90 degrees. Each cell grid point is calculated as the mid-point of
    the associated MxN cell.

    Parameters
    ----------
    spec : str
        Specifies the 'MxN' degree cell-specification for the global grid.
    lat_offset : bool
        Offset the grid centers of the latitude coordinate w.r.t. the
        pole by half a grid step. This argument is ignored if `target_grid`
        is a cube or file.
    lon_offset : bool
        Offset the grid centers of the longitude coordinate w.r.t. Greenwich
        meridian by half a grid step.
        This argument is ignored if `target_grid` is a cube or file.

    Returns
    -------
    :class:`~iris.cube.Cube`
    """
    dlon, dlat = parse_cell_spec(spec)
    mid_dlon, mid_dlat = dlon / 2, dlat / 2

    # Construct the latitude coordinate, with bounds.
    if lat_offset:
        latdata = np.linspace(_LAT_MIN + mid_dlat, _LAT_MAX - mid_dlat,
                              int(_LAT_RANGE / dlat))
    else:
        latdata = np.linspace(_LAT_MIN, _LAT_MAX, int(_LAT_RANGE / dlat) + 1)

    # Construct the longitude coordinat, with bounds.
    if lon_offset:
        londata = np.linspace(_LON_MIN + mid_dlon, _LON_MAX - mid_dlon,
                              int(_LON_RANGE / dlon))
    else:
        londata = np.linspace(_LON_MIN, _LON_MAX - dlon,
                              int(_LON_RANGE / dlon))

    cube = _generate_cube_from_dimcoords(latdata=latdata, londata=londata)

    return cube


def _spec_to_latlonvals(*, start_latitude: float, end_latitude: float,
                        step_latitude: float, start_longitude: float,
                        end_longitude: float, step_longitude: float) -> tuple:
    """Define lat/lon values from spec.

    Create a regional cube starting defined by the target specification.

    The latitude must be between -90 and +90. The longitude is not bounded, but
    wraps around the full great circle.

    Parameters
    ----------
    start_latitude : float
        Latitude value of the first grid cell center (start point). The grid
        includes this value.
    end_latitude : float
        Latitude value of the last grid cell center (end point). The grid
        includes this value only if it falls on a grid point. Otherwise, it
        cuts off at the previous value.
    step_latitude : float
        Latitude distance between the centers of two neighbouring cells.
    start_longitude : float
        Latitude value of the first grid cell center (start point). The grid
        includes this value.
    end_longitude : float
        Longitude value of the last grid cell center (end point). The grid
        includes this value only if it falls on a grid point. Otherwise, it
        cuts off at the previous value.
    step_longitude : float
        Longitude distance between the centers of two neighbouring cells.

    Returns
    -------
    xvals : np.array
        List of longitudes
    yvals : np.array
        List of latitudes
    """
    if step_latitude == 0:
        raise ValueError('Latitude step cannot be 0, '
                         f'got step_latitude={step_latitude}.')

    if step_longitude == 0:
        raise ValueError('Longitude step cannot be 0, '
                         f'got step_longitude={step_longitude}.')

    if (start_latitude < _LAT_MIN) or (end_latitude > _LAT_MAX):
        raise ValueError(
            f'Latitude values must lie between {_LAT_MIN}:{_LAT_MAX}, '
            f'got start_latitude={start_latitude}:end_latitude={end_latitude}.'
        )

    def get_points(start, stop, step):
        """Calculate grid points."""
        # use Decimal to avoid floating point errors
        num = int(Decimal(stop - start) // Decimal(str(step)))
        stop = start + num * step
        return np.linspace(start, stop, num + 1)

    latitudes = get_points(start_latitude, end_latitude, step_latitude)
    longitudes = get_points(start_longitude, end_longitude, step_longitude)

    return latitudes, longitudes


def _regional_stock_cube(spec: dict):
    """Create a regional stock cube.

    Returns
    -------
    :class:`~iris.cube.Cube`.
    """
    latdata, londata = _spec_to_latlonvals(**spec)

    cube = _generate_cube_from_dimcoords(latdata=latdata,
                                         londata=londata,
                                         circular=True)

    def add_bounds_from_step(coord, step):
        """Calculate bounds from the given step."""
        bound = step / 2
        points = coord.points
        coord.bounds = np.vstack((points - bound, points + bound)).T

    add_bounds_from_step(cube.coord('latitude'), spec['step_latitude'])
    add_bounds_from_step(cube.coord('longitude'), spec['step_longitude'])

    return cube


def _attempt_irregular_regridding(cube, scheme):
    """Check if irregular regridding with ESMF should be used."""
    if isinstance(scheme, str) and scheme in ESMF_REGRID_METHODS:
        try:
            lat_dim = cube.coord('latitude').ndim
            lon_dim = cube.coord('longitude').ndim
            if lat_dim == lon_dim == 2:
                return True
        except iris.exceptions.CoordinateNotFoundError:
            pass
    return False


def extract_location(cube, location, scheme):
    """Extract a point using a location name, with interpolation.

    Extracts a single location point from a cube, according
    to the interpolation scheme ``scheme``.

    The function just retrieves the coordinates of the location and then calls
    the ``extract_point`` preprocessor.

    It can be used to locate cities and villages, but also mountains or other
    geographical locations.

    Note
    ----
    The geolocator needs a working internet connection.

    Parameters
    ----------
    cube : cube
        The source cube to extract a point from.

    location : str
        The reference location. Examples: 'mount everest',
        'romania','new york, usa'

    scheme : str
        The interpolation scheme. 'linear' or 'nearest'. No default.

    Returns
    -------
    Returns a cube with the extracted point, and with adjusted
    latitude and longitude coordinates.

    Raises
    ------
    ValueError:
        If location is not supplied as a preprocessor parameter.
    ValueError:
        If scheme is not supplied as a preprocessor parameter.
    ValueError:
        If given location cannot be found by the geolocator.
    """
    if location is None:
        raise ValueError("Location needs to be specified."
                         " Examples: 'mount everest', 'romania',"
                         " 'new york, usa'")
    if scheme is None:
        raise ValueError("Interpolation scheme needs to be specified."
                         " Use either 'linear' or 'nearest'.")
    geolocator = Nominatim(user_agent='esmvalcore')
    geolocation = geolocator.geocode(location)
    if geolocation is None:
        raise ValueError(f'Requested location {location} can not be found.')
    logger.info("Extracting data for %s (%s °N, %s °E)", geolocation,
                geolocation.latitude, geolocation.longitude)

    return extract_point(cube, geolocation.latitude,
                         geolocation.longitude, scheme)


def extract_point(cube, latitude, longitude, scheme):
    """Extract a point, with interpolation.

    Extracts a single latitude/longitude point from a cube, according
    to the interpolation scheme `scheme`.

    Multiple points can also be extracted, by supplying an array of
    latitude and/or longitude coordinates. The resulting point cube
    will match the respective latitude and longitude coordinate to
    those of the input coordinates. If the input coordinate is a
    scalar, the dimension will be missing in the output cube (that is,
    it will be a scalar).

    If the point to be extracted has at least one of the coordinate point
    values outside the interval of the cube's same coordinate values, then
    no extrapolation will be performed, and the resulting extracted cube
    will have fully masked data.

    Parameters
    ----------
    cube : cube
        The source cube to extract a point from.

    latitude, longitude : float, or array of float
        The latitude and longitude of the point.

    scheme : str
        The interpolation scheme. 'linear' or 'nearest'. No default.

    Returns
    -------
    :py:class:`~iris.cube.Cube`
        Returns a cube with the extracted point(s), and with adjusted
        latitude and longitude coordinates (see above). If desired point
        outside values for at least one coordinate, this cube will have fully
        masked data.

    Raises
    ------
    ValueError:
        If the interpolation scheme is None or unrecognized.

    Examples
    --------
    With a cube that has the coordinates

    - latitude: [1, 2, 3, 4]
    - longitude: [1, 2, 3, 4]
    - data values: [[[1, 2, 3, 4], [5, 6, ...], [...], [...],
                      ... ]]]

    >>> point = extract_point(cube, 2.5, 2.5, 'linear')  # doctest: +SKIP
    >>> point.data  # doctest: +SKIP
    array([ 8.5, 24.5, 40.5, 56.5])

    Extraction of multiple points at once, with a nearest matching scheme.
    The values for 0.1 will result in masked values, since this lies outside
    the cube grid.

    >>> point = extract_point(cube, [1.4, 2.1], [0.1, 1.1],
    ...                       'nearest')  # doctest: +SKIP
    >>> point.data.shape  # doctest: +SKIP
    (4, 2, 2)
    >>> # x, y, z indices of masked values
    >>> np.where(~point.data.mask)     # doctest: +SKIP
    (array([0, 0, 1, 1, 2, 2, 3, 3]), array([0, 1, 0, 1, 0, 1, 0, 1]),
    array([1, 1, 1, 1, 1, 1, 1, 1]))
    >>> point.data[~point.data.mask].data  # doctest: +SKIP
    array([ 1,  5, 17, 21, 33, 37, 49, 53])
    """
    msg = f"Unknown interpolation scheme, got {scheme!r}."
    scheme = POINT_INTERPOLATION_SCHEMES.get(scheme.lower())
    if not scheme:
        raise ValueError(msg)

    point = [('latitude', latitude), ('longitude', longitude)]
    cube = cube.interpolate(point, scheme=scheme)
    return cube


def regrid(cube, target_grid, scheme, lat_offset=True, lon_offset=True):
    """Perform horizontal regridding.

    Note that the target grid can be a cube (:py:class:`~iris.cube.Cube`),
    path to a cube (``str``), a grid spec (``str``) in the form
    of `MxN`, or a ``dict`` specifying the target grid.

    For the latter, the ``target_grid`` should be a ``dict`` with the
    following keys:

    - ``start_longitude``: longitude at the center of the first grid cell.
    - ``end_longitude``: longitude at the center of the last grid cell.
    - ``step_longitude``: constant longitude distance between grid cell \
        centers.
    - ``start_latitude``: latitude at the center of the first grid cell.
    - ``end_latitude``: longitude at the center of the last grid cell.
    - ``step_latitude``: constant latitude distance between grid cell centers.

    Parameters
    ----------
    cube : :py:class:`~iris.cube.Cube`
        The source cube to be regridded.
    target_grid : Cube or str or dict
        The (location of a) cube that specifies the target or reference grid
        for the regridding operation.

        Alternatively, a string cell specification may be provided,
        of the form ``MxN``, which specifies the extent of the cell, longitude
        by latitude (degrees) for a global, regular target grid.

        Alternatively, a dictionary with a regional target grid may
        be specified (see above).

    scheme : str or dict
        The regridding scheme to perform. If both source and target grid are
        structured (regular or irregular), can be one of the built-in schemes
        ``linear``, ``linear_extrapolate``, ``nearest``, ``area_weighted``,
        ``unstructured_nearest``.
        Alternatively, a `dict` that specifies generic regridding (see below).
    lat_offset : bool
        Offset the grid centers of the latitude coordinate w.r.t. the
        pole by half a grid step. This argument is ignored if ``target_grid``
        is a cube or file.
    lon_offset : bool
        Offset the grid centers of the longitude coordinate w.r.t. Greenwich
        meridian by half a grid step.
        This argument is ignored if ``target_grid`` is a cube or file.

    Returns
    -------
    :py:class:`~iris.cube.Cube`
        Regridded cube.

    See Also
    --------
    extract_levels : Perform vertical regridding.

    Notes
    -----
    This preprocessor allows for the use of arbitrary :doc:`Iris <iris:index>`
    regridding schemes, that is anything that can be passed as a scheme to
    :meth:`iris.cube.Cube.regrid` is possible. This enables the use of further
    parameters for existing schemes, as well as the use of more advanced
    schemes for example for unstructured meshes.
    To use this functionality, a dictionary must be passed for the scheme with
    a mandatory entry of ``reference`` in the form specified for the object
    reference of the `entry point data model <https://packaging.python.org/en/
    latest/specifications/entry-points/#data-model>`_,
    i.e. ``importable.module:object.attr``. This is used as a factory for the
    scheme. Any further entries in the dictionary are passed as keyword
    arguments to the factory.

    For example, to use the familiar :class:`iris.analysis.Linear` regridding
    scheme with a custom extrapolation mode, use

    .. code-block:: yaml

        my_preprocessor:
          regrid:
            target: 1x1
            scheme:
              reference: iris.analysis:Linear
              extrapolation_mode: nanmask

    To use the area weighted regridder available in
    :class:`esmf_regrid.schemes.ESMFAreaWeighted`, make sure that
    :doc:`iris-esmf-regrid:index` is installed and use

    .. code-block:: yaml

        my_preprocessor:
          regrid:
            target: 1x1
            scheme:
              reference: esmf_regrid.schemes:ESMFAreaWeighted

    .. note::

        Note that :doc:`iris-esmf-regrid:index` is still experimental.
    """
    if isinstance(scheme, dict):
        try:
            object_ref = scheme.pop("reference")
        except KeyError as key_err:
            raise ValueError(
                "No reference specified for generic regridding.") from key_err
        module_name, separator, scheme_name = object_ref.partition(":")
        try:
            obj = importlib.import_module(module_name)
        except ImportError as import_err:
            raise ValueError(
                "Could not import specified generic regridding module. "
                "Please double check spelling and that the required module is "
                "installed.") from import_err
        if separator:
            for attr in scheme_name.split('.'):
                obj = getattr(obj, attr)
        loaded_scheme = obj(**scheme)
    else:
        loaded_scheme = HORIZONTAL_SCHEMES.get(scheme.lower())
    if loaded_scheme is None:
        emsg = 'Unknown regridding scheme, got {!r}.'
        raise ValueError(emsg.format(scheme))

    if isinstance(target_grid, str):
        if os.path.isfile(target_grid):
            target_grid = iris.load_cube(target_grid)
        else:
            # Generate a target grid from the provided cell-specification,
            # and cache the resulting stock cube for later use.
            target_grid = _CACHE.setdefault(
                target_grid,
                _global_stock_cube(target_grid, lat_offset, lon_offset),
            )
            # Align the target grid coordinate system to the source
            # coordinate system.
            src_cs = cube.coord_system()
            xcoord = target_grid.coord(axis='x', dim_coords=True)
            ycoord = target_grid.coord(axis='y', dim_coords=True)
            xcoord.coord_system = src_cs
            ycoord.coord_system = src_cs

    elif isinstance(target_grid, dict):
        # Generate a target grid from the provided specification,
        target_grid = _regional_stock_cube(target_grid)

    if not isinstance(target_grid, iris.cube.Cube):
        raise ValueError('Expecting a cube, got {}.'.format(target_grid))

    # Unstructured regridding requires x2 2d spatial coordinates,
    # so ensure to purge any 1d native spatial dimension coordinates
    # for the regridder.
    if scheme == 'unstructured_nearest':
        for axis in ['x', 'y']:
            coords = cube.coords(axis=axis, dim_coords=True)
            if coords:
                [coord] = coords
                cube.remove_coord(coord)

    # Return non-regridded cube if horizontal grid is the same.
    if not _horizontal_grid_is_close(cube, target_grid):
        original_dtype = cube.core_data().dtype

        # For 'unstructured_nearest', make sure that consistent fill value is
        # used since the data is not masked after regridding (see
        # https://github.com/SciTools/iris/issues/4463)
        # Note: da.ma.set_fill_value() works with any kind of input data
        # (masked and unmasked, numpy and dask)
        if scheme == 'unstructured_nearest':
            if np.issubdtype(cube.dtype, np.integer):
                fill_value = np.iinfo(cube.dtype).max
            else:
                fill_value = GLOBAL_FILL_VALUE
            da.ma.set_fill_value(cube.core_data(), fill_value)

        # Perform the horizontal regridding
        if _attempt_irregular_regridding(cube, scheme):
            cube = esmpy_regrid(cube, target_grid, scheme)
        else:
            cube = cube.regrid(target_grid, loaded_scheme)

        # Preserve dtype and use masked arrays for 'unstructured_nearest'
        # scheme (see https://github.com/SciTools/iris/issues/4463)
        if scheme == 'unstructured_nearest':
            try:
                cube.data = cube.core_data().astype(original_dtype,
                                                    casting='same_kind')
            except TypeError as exc:
                logger.warning(
                    "dtype of data changed during regridding from '%s' to "
                    "'%s': %s", original_dtype, cube.core_data().dtype,
                    str(exc))
            cube.data = da.ma.masked_equal(cube.core_data(), fill_value)
    else:
        # force target coordinates
        for coord in ['latitude', 'longitude']:
            cube.coord(coord).points = target_grid.coord(coord).points
            cube.coord(coord).bounds = target_grid.coord(coord).bounds

    return cube


def _horizontal_grid_is_close(cube1, cube2):
    """Check if two cubes have the same horizontal grid definition.

    The result of the function is a boolean answer, if both cubes have the
    same horizontal grid definition. The function checks both longitude and
    latitude, based on extent and resolution.

    Parameters
    ----------
    cube1 : cube
        The first of the cubes to be checked.
    cube2 : cube
        The second of the cubes to be checked.

    Returns
    -------
    bool

    .. note::

        The current implementation checks if the bounds and the
        grid shapes are the same.
        Exits on first difference.
    """
    # Go through the 2 expected horizontal coordinates longitude and latitude.
    for coord in ['latitude', 'longitude']:
        coord1 = cube1.coord(coord)
        coord2 = cube2.coord(coord)

        if not coord1.shape == coord2.shape:
            return False

        if not np.allclose(coord1.bounds, coord2.bounds):
            return False

    return True


def _create_cube(src_cube, data, src_levels, levels):
    """Generate a new cube with the interpolated data.

    The resultant cube is seeded with `src_cube` metadata and coordinates,
    excluding any source coordinates that span the associated vertical
    dimension. The `levels` of interpolation are used along with the
    associated source cube vertical coordinate metadata to add a new
    vertical coordinate to the resultant cube.

    Parameters
    ----------
    src_cube : cube
        The source cube that was vertically interpolated.
    data : array
        The payload resulting from interpolating the source cube
        over the specified levels.
    src_levels : array
        Vertical levels of the source data
    levels : array
        The vertical levels of interpolation.

    Returns
    -------
    cube

    .. note::

        If there is only one level of interpolation, the resultant cube
        will be collapsed over the associated vertical dimension, and a
        scalar vertical coordinate will be added.
    """
    # Get the source cube vertical coordinate and associated dimension.
    z_coord = src_cube.coord(axis='z', dim_coords=True)
    z_dim, = src_cube.coord_dims(z_coord)

    if data.shape[z_dim] != levels.size:
        emsg = ('Mismatch between data and levels for data dimension {!r}, '
                'got data shape {!r} with levels shape {!r}.')
        raise ValueError(emsg.format(z_dim, data.shape, levels.shape))

    # Construct the resultant cube with the interpolated data
    # and the source cube metadata.
    kwargs = deepcopy(src_cube.metadata)._asdict()
    result = iris.cube.Cube(data, **kwargs)

    # Add the appropriate coordinates to the cube, excluding
    # any coordinates that span the z-dimension of interpolation.
    for coord in src_cube.dim_coords:
        [dim] = src_cube.coord_dims(coord)
        if dim != z_dim:
            result.add_dim_coord(coord.copy(), dim)

    for coord in src_cube.aux_coords:
        dims = src_cube.coord_dims(coord)
        if z_dim not in dims:
            result.add_aux_coord(coord.copy(), dims)

    for coord in src_cube.derived_coords:
        dims = src_cube.coord_dims(coord)
        if z_dim not in dims:
            result.add_aux_coord(coord.copy(), dims)

    # Construct the new vertical coordinate for the interpolated
    # z-dimension, using the associated source coordinate metadata.
    metadata = src_levels.metadata

    kwargs = {
        'standard_name': metadata.standard_name,
        'long_name': metadata.long_name,
        'var_name': metadata.var_name,
        'units': metadata.units,
        'attributes': metadata.attributes,
        'coord_system': metadata.coord_system,
        'climatological': metadata.climatological,
    }

    try:
        coord = iris.coords.DimCoord(levels, **kwargs)
        result.add_dim_coord(coord, z_dim)
    except ValueError:
        coord = iris.coords.AuxCoord(levels, **kwargs)
        result.add_aux_coord(coord, z_dim)

    # Collapse the z-dimension for the scalar case.
    if levels.size == 1:
        slicer = [slice(None)] * result.ndim
        slicer[z_dim] = 0
        result = result[tuple(slicer)]

    return result


def _vertical_interpolate(cube, src_levels, levels, interpolation,
                          extrapolation):
    """Perform vertical interpolation."""
    # Determine the source levels and axis for vertical interpolation.
    z_axis, = cube.coord_dims(cube.coord(axis='z', dim_coords=True))

    # Broadcast the 1d source cube vertical coordinate to fully
    # describe the spatial extent that will be interpolated.
    src_levels_broadcast = broadcast_to_shape(src_levels.points, cube.shape,
                                              cube.coord_dims(src_levels))

    # force mask onto data as nan's
    cube.data = da.ma.filled(cube.core_data(), np.nan)

    # Now perform the actual vertical interpolation.
    new_data = stratify.interpolate(levels,
                                    src_levels_broadcast,
                                    cube.core_data(),
                                    axis=z_axis,
                                    interpolation=interpolation,
                                    extrapolation=extrapolation)

    # Calculate the mask based on the any NaN values in the interpolated data.
    mask = np.isnan(new_data)

    if np.any(mask):
        # Ensure that the data is masked appropriately.
        new_data = np.ma.array(new_data, mask=mask, fill_value=_MDI)

    # Construct the resulting cube with the interpolated data.
    return _create_cube(cube, new_data, src_levels, levels.astype(float))


def _preserve_fx_vars(cube, result):
    vertical_dim = set(cube.coord_dims(cube.coord(axis='z', dim_coords=True)))
    if cube.cell_measures():
        for measure in cube.cell_measures():
            measure_dims = set(cube.cell_measure_dims(measure))
            if vertical_dim.intersection(measure_dims):
                logger.warning(
                    'Discarding use of z-axis dependent cell measure %s '
                    'in variable %s, as z-axis has been interpolated',
                    measure.var_name, result.var_name)
            else:
                add_cell_measure(result, measure, measure.measure)
    if cube.ancillary_variables():
        for ancillary_var in cube.ancillary_variables():
            ancillary_dims = set(cube.ancillary_variable_dims(ancillary_var))
            if vertical_dim.intersection(ancillary_dims):
                logger.warning(
                    'Discarding use of z-axis dependent ancillary variable %s '
                    'in variable %s, as z-axis has been interpolated',
                    ancillary_var.var_name, result.var_name)
            else:
                add_ancillary_variable(result, ancillary_var)


def parse_vertical_scheme(scheme):
    """Parse the scheme provided for level extraction.

    Parameters
    ----------
    scheme : str
        The vertical interpolation scheme to use. Choose from
        'linear',
        'nearest',
        'linear_extrapolate',
        'nearest_extrapolate'.

    Returns
    -------
    (str, str)
        A tuple containing the interpolation and extrapolation scheme.
    """
    # Check if valid scheme is given
    if scheme not in VERTICAL_SCHEMES:
        raise ValueError(
            f"Unknown vertical interpolation scheme, got '{scheme}', possible "
            f"schemes are {VERTICAL_SCHEMES}")

    # This allows us to put level 0. to load the ocean surface.
    extrap_scheme = 'nan'

    if scheme == 'linear_extrapolate':
        scheme = 'linear'
        extrap_scheme = 'nearest'

    if scheme == 'nearest_extrapolate':
        scheme = 'nearest'
        extrap_scheme = 'nearest'

    return scheme, extrap_scheme


def extract_levels(cube,
                   levels,
                   scheme,
                   coordinate=None,
                   rtol=1e-7,
                   atol=None):
    """Perform vertical interpolation.

    Parameters
    ----------
    cube : iris.cube.Cube
        The source cube to be vertically interpolated.
    levels : ArrayLike
        One or more target levels for the vertical interpolation. Assumed
        to be in the same S.I. units of the source cube vertical dimension
        coordinate. If the requested levels are sufficiently close to the
        levels of the cube, cube slicing will take place instead of
        interpolation.
    scheme : str
        The vertical interpolation scheme to use. Choose from
        'linear',
        'nearest',
        'linear_extrapolate',
        'nearest_extrapolate'.
    coordinate :  optional str
        The coordinate to interpolate. If specified, pressure levels
        (if present) can be converted to height levels and vice versa using
        the US standard atmosphere. E.g. 'coordinate = altitude' will convert
        existing pressure levels (air_pressure) to height levels (altitude);
        'coordinate = air_pressure' will convert existing height levels
        (altitude) to pressure levels (air_pressure).
    rtol : float
        Relative tolerance for comparing the levels in `cube` to the requested
        levels. If the levels are sufficiently close, the requested levels
        will be assigned to the cube and no interpolation will take place.
    atol : float
        Absolute tolerance for comparing the levels in `cube` to the requested
        levels. If the levels are sufficiently close, the requested levels
        will be assigned to the cube and no interpolation will take place.
        By default, `atol` will be set to 10^-7 times the mean value of
        the levels on the cube.

    Returns
    -------
    iris.cube.Cube
        A cube with the requested vertical levels.


    See Also
    --------
    regrid : Perform horizontal regridding.
    """
    interpolation, extrapolation = parse_vertical_scheme(scheme)

    # Ensure we have a non-scalar array of levels.
    levels = np.array(levels, ndmin=1)

    # Get the source cube vertical coordinate, if available.
    if coordinate:
        coord_names = [coord.name() for coord in cube.coords()]
        if coordinate not in coord_names:
            # Try to calculate air_pressure from altitude coordinate or
            # vice versa using US standard atmosphere for conversion.
            if coordinate == 'air_pressure' and 'altitude' in coord_names:
                # Calculate pressure level coordinate from altitude.
                add_plev_from_altitude(cube)
            if coordinate == 'altitude' and 'air_pressure' in coord_names:
                # Calculate altitude coordinate from pressure levels.
                add_altitude_from_plev(cube)
        src_levels = cube.coord(coordinate)
    else:
        src_levels = cube.coord(axis='z', dim_coords=True)

    if (src_levels.shape == levels.shape and np.allclose(
            src_levels.points,
            levels,
            rtol=rtol,
            atol=1e-7 * np.mean(src_levels.points) if atol is None else atol,
    )):
        # Only perform vertical extraction/interpolation if the source
        # and target levels are not "similar" enough.
        result = cube
        # Set the levels to the requested values
        src_levels.points = levels
    elif len(src_levels.shape) == 1 and \
            set(levels).issubset(set(src_levels.points)):
        # If all target levels exist in the source cube, simply extract them.
        name = src_levels.name()
        coord_values = {name: lambda cell: cell.point in set(levels)}
        constraint = iris.Constraint(coord_values=coord_values)
        result = cube.extract(constraint)
        # Ensure the constraint did not fail.
        if not result:
            emsg = 'Failed to extract levels {!r} from cube {!r}.'
            raise ValueError(emsg.format(list(levels), name))
    else:
        # As a last resort, perform vertical interpolation.
        result = _vertical_interpolate(
            cube,
            src_levels,
            levels,
            interpolation,
            extrapolation,
        )
        _preserve_fx_vars(cube, result)

    return result


def get_cmor_levels(cmor_table, coordinate):
    """Get level definition from a CMOR coordinate.

    Parameters
    ----------
    cmor_table: str
        CMOR table name
    coordinate: str
        CMOR coordinate name

    Returns
    -------
    list[int]

    Raises
    ------
    ValueError:
        If the CMOR table is not defined, the coordinate does not specify any
        levels or the string is badly formatted.
    """
    if cmor_table not in CMOR_TABLES:
        raise ValueError(
            "Level definition cmor_table '{}' not available".format(
                cmor_table))

    if coordinate not in CMOR_TABLES[cmor_table].coords:
        raise ValueError('Coordinate {} not available for {}'.format(
            coordinate, cmor_table))

    cmor = CMOR_TABLES[cmor_table].coords[coordinate]

    if cmor.requested:
        return [float(level) for level in cmor.requested]
    if cmor.value:
        return [float(cmor.value)]

    raise ValueError(
        'Coordinate {} in {} does not have requested values'.format(
            coordinate, cmor_table))


def get_reference_levels(filename, project, dataset, short_name, mip,
                         frequency, fix_dir):
    """Get level definition from a reference dataset.

    Parameters
    ----------
    filename: str
        Path to the reference file
    project : str
        Name of the project
    dataset : str
        Name of the dataset
    short_name : str
        Name of the variable
    mip : str
        Name of the mip table
    frequency : str
        Time frequency
    fix_dir : str
        Output directory for fixed data

    Returns
    -------
    list[float]

    Raises
    ------
    ValueError:
        If the dataset is not defined, the coordinate does not specify any
        levels or the string is badly formatted.
    """
    filename = fix_file(
        file=filename,
        short_name=short_name,
        project=project,
        dataset=dataset,
        mip=mip,
        output_dir=fix_dir,
    )
    cubes = load(filename, callback=concatenate_callback)
    cubes = fix_metadata(
        cubes=cubes,
        short_name=short_name,
        project=project,
        dataset=dataset,
        mip=mip,
        frequency=frequency,
    )
    cube = cubes[0]
    try:
        coord = cube.coord(axis='Z')
    except iris.exceptions.CoordinateNotFoundError:
        raise ValueError('z-coord not available in {}'.format(filename))
    return coord.points.tolist()


def extract_coordinate_points(cube, definition, scheme):
    """Extract points from any coordinate with interpolation.

    Multiple points can also be extracted, by supplying an array of
    coordinates. The resulting point cube will match the respective
    coordinates to those of the input coordinates.
    If the input coordinate is a scalar, the dimension will be a
    scalar in the output cube.

    Parameters
    ----------
    cube : cube
        The source cube to extract a point from.
    defintion : dict(str, float or array of float)
        The coordinate - values pairs to extract
    scheme : str
        The interpolation scheme. 'linear' or 'nearest'. No default.

    Returns
    -------
    :py:class:`~iris.cube.Cube`
        Returns a cube with the extracted point(s), and with adjusted
        latitude and longitude coordinates (see above). If desired point
        outside values for at least one coordinate, this cube will have fully
        masked data.

    Raises
    ------
    ValueError:
        If the interpolation scheme is not provided or is not recognised.
    """

    msg = f"Unknown interpolation scheme, got {scheme!r}."
    scheme = POINT_INTERPOLATION_SCHEMES.get(scheme.lower())
    if not scheme:
        raise ValueError(msg)
    cube = cube.interpolate(definition.items(), scheme=scheme)
    return cube
