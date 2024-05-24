"""Horizontal and vertical regridding module."""
from __future__ import annotations

import functools
import importlib
import inspect
import logging
import os
import re
import ssl
import warnings
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import dask.array as da
import iris
import numpy as np
import stratify
from geopy.geocoders import Nominatim
from iris.analysis import AreaWeighted, Linear, Nearest
from iris.cube import Cube

from esmvalcore.cmor._fixes.shared import (
    add_altitude_from_plev,
    add_plev_from_altitude,
)
from esmvalcore.cmor.table import CMOR_TABLES
from esmvalcore.exceptions import ESMValCoreDeprecationWarning
from esmvalcore.iris_helpers import has_irregular_grid, has_unstructured_grid
from esmvalcore.preprocessor._shared import (
    broadcast_to_shape,
    get_array_module,
    preserve_float_dtype,
)
from esmvalcore.preprocessor._supplementary_vars import (
    add_ancillary_variable,
    add_cell_measure,
)
from esmvalcore.preprocessor.regrid_schemes import (
    ESMPyAreaWeighted,
    ESMPyLinear,
    ESMPyNearest,
    GenericFuncScheme,
    UnstructuredLinear,
    UnstructuredNearest,
)

if TYPE_CHECKING:
    from esmvalcore.dataset import Dataset

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

# Supported point interpolation schemes.
POINT_INTERPOLATION_SCHEMES = {
    'linear': Linear(extrapolation_mode='mask'),
    'nearest': Nearest(extrapolation_mode='mask'),
}

# Supported horizontal regridding schemes for regular grids (= rectilinear
# grids; i.e., grids that can be described with 1D latitude and 1D longitude
# coordinates orthogonal to each other)
HORIZONTAL_SCHEMES_REGULAR = {
    'area_weighted': AreaWeighted(),
    'linear': Linear(extrapolation_mode='mask'),
    'nearest': Nearest(extrapolation_mode='mask'),
}

# Supported horizontal regridding schemes for irregular grids (= general
# curvilinear grids; i.e., grids that can be described with 2D latitude and 2D
# longitude coordinates with common dimensions)
HORIZONTAL_SCHEMES_IRREGULAR = {
    'area_weighted': ESMPyAreaWeighted(),
    'linear': ESMPyLinear(),
    'nearest': ESMPyNearest(),
}

# Supported horizontal regridding schemes for unstructured grids (i.e., grids,
# that can be described with 1D latitude and 1D longitude coordinate with
# common dimensions)
HORIZONTAL_SCHEMES_UNSTRUCTURED = {
    'linear': UnstructuredLinear(),
    'nearest': UnstructuredNearest(),
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
    iris.cube.Cube
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
    cube = Cube(dummy, dim_coords_and_dims=coords_spec)

    return cube


@functools.lru_cache
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
    iris.cube.Cube
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
    iris.cube.Cube
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
    try:
        # Try to use the default SSL context, see
        # https://github.com/ESMValGroup/ESMValCore/issues/2012 for more
        # information.
        ssl_context = ssl.create_default_context()
        geolocator = Nominatim(user_agent='esmvalcore',
                               ssl_context=ssl_context)
    except ssl.SSLError:
        logger.warning(
            "ssl.create_default_context() encountered a problem, not using it."
        )
        geolocator = Nominatim(user_agent='esmvalcore')
    geolocation = geolocator.geocode(location)
    if geolocation is None:
        raise ValueError(f'Requested location {location} can not be found.')
    logger.info("Extracting data for %s (%s °N, %s °E)", geolocation,
                geolocation.latitude, geolocation.longitude)

    return extract_point(cube, geolocation.latitude, geolocation.longitude,
                         scheme)


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
    iris.cube.Cube
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


def is_dataset(dataset):
    """Test if something is an `esmvalcore.dataset.Dataset`."""
    # Use this function to avoid circular imports
    return hasattr(dataset, 'facets')


def _get_target_grid_cube(
    cube: Cube,
    target_grid: Cube | Dataset | Path | str | dict,
    lat_offset: bool = True,
    lon_offset: bool = True,
) -> Cube:
    """Get target grid cube."""
    if is_dataset(target_grid):
        target_grid = target_grid.copy()  # type: ignore
        target_grid.supplementaries.clear()  # type: ignore
        target_grid.files = [target_grid.files[0]]  # type: ignore
        target_grid_cube = target_grid.load()  # type: ignore
    elif isinstance(target_grid, (str, Path)) and os.path.isfile(target_grid):
        target_grid_cube = iris.load_cube(target_grid)
    elif isinstance(target_grid, str):
        # Generate a target grid from the provided cell-specification
        target_grid_cube = _global_stock_cube(target_grid, lat_offset,
                                              lon_offset)
        # Align the target grid coordinate system to the source
        # coordinate system.
        src_cs = cube.coord_system()
        xcoord = target_grid_cube.coord(axis='x', dim_coords=True)
        ycoord = target_grid_cube.coord(axis='y', dim_coords=True)
        xcoord.coord_system = src_cs
        ycoord.coord_system = src_cs
    elif isinstance(target_grid, dict):
        # Generate a target grid from the provided specification,
        target_grid_cube = _regional_stock_cube(target_grid)
    else:
        target_grid_cube = target_grid

    if not isinstance(target_grid_cube, Cube):
        raise ValueError(f'Expecting a cube, got {target_grid}.')

    return target_grid_cube


def _attempt_irregular_regridding(cube: Cube, scheme: str) -> bool:
    """Check if irregular regridding with ESMF should be used."""
    if not has_irregular_grid(cube):
        return False
    if scheme not in HORIZONTAL_SCHEMES_IRREGULAR:
        raise ValueError(
            f"Regridding scheme '{scheme}' does not support irregular data, "
            f"expected one of {list(HORIZONTAL_SCHEMES_IRREGULAR)}")
    return True


def _attempt_unstructured_regridding(cube: Cube, scheme: str) -> bool:
    """Check if unstructured regridding should be used."""
    if not has_unstructured_grid(cube):
        return False
    if scheme not in HORIZONTAL_SCHEMES_UNSTRUCTURED:
        raise ValueError(
            f"Regridding scheme '{scheme}' does not support unstructured "
            f"data, expected one of {list(HORIZONTAL_SCHEMES_UNSTRUCTURED)}")
    return True


def _load_scheme(src_cube: Cube, scheme: str | dict):
    """Return scheme that can be used in :meth:`iris.cube.Cube.regrid`."""
    loaded_scheme: Any = None

    # Deprecations
    if scheme == 'unstructured_nearest':
        msg = (
            "The regridding scheme `unstructured_nearest` has been deprecated "
            "in ESMValCore version 2.11.0 and is scheduled for removal in "
            "version 2.13.0. Please use the scheme `nearest` instead. This is "
            "an exact replacement for data on unstructured grids. Since "
            "version 2.11.0, ESMValCore is able to determine the most "
            "suitable regridding scheme based on the input data.")
        warnings.warn(msg, ESMValCoreDeprecationWarning)
        scheme = 'nearest'

    if scheme == 'linear_extrapolate':
        msg = (
            "The regridding scheme `linear_extrapolate` has been deprecated "
            "in ESMValCore version 2.11.0 and is scheduled for removal in "
            "version 2.13.0. Please use a generic scheme with `reference: "
            "iris.analysis:Linear` and `extrapolation_mode: extrapolate` "
            "instead (see https://docs.esmvaltool.org/projects/ESMValCore/en/"
            "latest/recipe/preprocessor.html#generic-regridding-schemes)."
            "This is an exact replacement.")
        warnings.warn(msg, ESMValCoreDeprecationWarning)
        scheme = 'linear'
        loaded_scheme = Linear(extrapolation_mode='extrapolate')
        logger.debug("Loaded regridding scheme %s", loaded_scheme)
        return loaded_scheme

    # Scheme is a dict -> assume this describes a generic regridding scheme
    if isinstance(scheme, dict):
        loaded_scheme = _load_generic_scheme(scheme)

    # Scheme is a str -> load appropriate regridding scheme depending on the
    # type of input data
    elif _attempt_irregular_regridding(src_cube, scheme):
        loaded_scheme = HORIZONTAL_SCHEMES_IRREGULAR[scheme]
    elif _attempt_unstructured_regridding(src_cube, scheme):
        loaded_scheme = HORIZONTAL_SCHEMES_UNSTRUCTURED[scheme]
    else:
        loaded_scheme = HORIZONTAL_SCHEMES_REGULAR.get(scheme)

    if loaded_scheme is None:
        raise ValueError(
            f"Got invalid regridding scheme string '{scheme}', expected one "
            f"of {list(HORIZONTAL_SCHEMES_REGULAR)}")

    logger.debug("Loaded regridding scheme %s", loaded_scheme)

    return loaded_scheme


def _load_generic_scheme(scheme: dict):
    """Load generic regridding scheme."""
    scheme = dict(scheme)  # do not overwrite original scheme

    try:
        object_ref = scheme.pop("reference")
    except KeyError as key_err:
        raise ValueError(
            "No reference specified for generic regridding.") from key_err
    module_name, separator, scheme_name = object_ref.partition(":")
    try:
        obj: Any = importlib.import_module(module_name)
    except ImportError as import_err:
        raise ValueError(
            f"Could not import specified generic regridding module "
            f"'{module_name}'. Please double check spelling and that the "
            f"required module is installed.") from import_err
    if separator:
        for attr in scheme_name.split('.'):
            obj = getattr(obj, attr)

    # If `obj` is a function that requires `src_cube` and `grid_cube`, use
    # GenericFuncScheme
    scheme_args = inspect.getfullargspec(obj).args
    if 'src_cube' in scheme_args and 'grid_cube' in scheme_args:
        loaded_scheme = GenericFuncScheme(obj, **scheme)
    else:
        loaded_scheme = obj(**scheme)

    return loaded_scheme


_CACHED_REGRIDDERS: dict[tuple, dict] = {}


def _get_regridder(
    src_cube: Cube,
    tgt_cube: Cube,
    scheme: str | dict,
    cache_weights: bool,
):
    """Get regridder to actually perform regridding.

    Note
    ----
    If possible, this uses an existing regridder to reduce runtime (see also
    https://scitools-iris.readthedocs.io/en/latest/userguide/
    interpolation_and_regridding.html#caching-a-regridder.)
    """
    # (1) Weights caching enabled
    if cache_weights:
        # To search for a matching regridder in the cache, first check the
        # regridding scheme name and shapes of source and target coordinates.
        # Only if these match, check coordinates themselves (this is much more
        # expensive).
        coord_key = _get_coord_key(src_cube, tgt_cube)
        name_shape_key = _get_name_and_shape_key(src_cube, tgt_cube, scheme)
        if name_shape_key in _CACHED_REGRIDDERS:
            # We cannot simply do a test for `coord_key in
            # _CACHED_REGRIDDERS[shape_key]` below since the hash() of a
            # coordinate is simply its id() (thus, coordinates loaded from two
            # different files would never be considered equal)
            for (key, regridder) in _CACHED_REGRIDDERS[name_shape_key].items():
                if key == coord_key:
                    return regridder

        # Regridder is not in cached -> return a new one and cache it
        loaded_scheme = _load_scheme(src_cube, scheme)
        regridder = loaded_scheme.regridder(src_cube, tgt_cube)
        _CACHED_REGRIDDERS.setdefault(name_shape_key, {})
        _CACHED_REGRIDDERS[name_shape_key][coord_key] = regridder

    # (2) Weights caching disabled
    else:
        loaded_scheme = _load_scheme(src_cube, scheme)
        regridder = loaded_scheme.regridder(src_cube, tgt_cube)

    return regridder


def _get_coord_key(src_cube: Cube, tgt_cube: Cube) -> tuple:
    """Get dict key from coordinates."""
    src_lat = src_cube.coord('latitude')
    src_lon = src_cube.coord('longitude')
    tgt_lat = tgt_cube.coord('latitude')
    tgt_lon = tgt_cube.coord('longitude')
    return (src_lat, src_lon, tgt_lat, tgt_lon)


def _get_name_and_shape_key(
    src_cube: Cube,
    tgt_cube: Cube,
    scheme: str | dict,
) -> tuple:
    """Get dict key from scheme name and coordinate shapes."""
    name = str(scheme)
    shapes = [c.shape for c in _get_coord_key(src_cube, tgt_cube)]
    return (name, *shapes)


@preserve_float_dtype
def regrid(
    cube: Cube,
    target_grid: Cube | Dataset | Path | str | dict,
    scheme: str | dict,
    lat_offset: bool = True,
    lon_offset: bool = True,
    cache_weights: bool = False,
) -> Cube:
    """Perform horizontal regridding.

    Note that the target grid can be a :class:`~iris.cube.Cube`, a
    :class:`~esmvalcore.dataset.Dataset`, a path to a cube
    (:class:`~pathlib.Path` or :obj:`str`), a grid spec (:obj:`str`) in the
    form of `MxN`, or a :obj:`dict` specifying the target grid.

    For the latter, the `target_grid` should be a :obj:`dict` with the
    following keys:

    - ``start_longitude``: longitude at the center of the first grid cell.
    - ``end_longitude``: longitude at the center of the last grid cell.
    - ``step_longitude``: constant longitude distance between grid cell
        centers.
    - ``start_latitude``: latitude at the center of the first grid cell.
    - ``end_latitude``: longitude at the center of the last grid cell.
    - ``step_latitude``: constant latitude distance between grid cell centers.

    Parameters
    ----------
    cube:
        The source cube to be regridded.
    target_grid:
        The (location of a) cube that specifies the target or reference grid
        for the regridding operation.
        Alternatively, a :class:`~esmvalcore.dataset.Dataset` can be provided.
        Alternatively, a string cell specification may be provided,
        of the form ``MxN``, which specifies the extent of the cell, longitude
        by latitude (degrees) for a global, regular target grid.
        Alternatively, a dictionary with a regional target grid may
        be specified (see above).
    scheme:
        The regridding scheme to perform. If the source grid is structured
        (i.e., rectilinear or curvilinear), can be one of the built-in schemes
        ``linear``, ``nearest``, ``area_weighted``. If the source grid is
        unstructured, can be one of the built-in schemes ``linear``,
        ``nearest``.  Alternatively, a `dict` that specifies generic regridding
        can be given (see below).
    lat_offset:
        Offset the grid centers of the latitude coordinate w.r.t. the pole by
        half a grid step. This argument is ignored if `target_grid` is a cube
        or file.
    lon_offset:
        Offset the grid centers of the longitude coordinate w.r.t. Greenwich
        meridian by half a grid step. This argument is ignored if
        `target_grid` is a cube or file.
    cache_weights:
        If ``True``, cache regridding weights for later usage. This can speed
        up the regridding of different datasets with similar source and target
        grids massively, but may take up a lot of memory for extremely
        high-resolution data. This option is ignored for schemes that do not
        support weights caching. More details on this are given in the section
        on :ref:`caching_regridding_weights`. To clear the cache, use
        :func:`esmvalcore.preprocessor.regrid.cache_clear`.

    Returns
    -------
    iris.cube.Cube
        Regridded cube.

    See Also
    --------
    extract_levels: Perform vertical regridding.

    Notes
    -----
    This preprocessor allows for the use of arbitrary :doc:`Iris <iris:index>`
    regridding schemes, that is anything that can be passed as a scheme to
    :meth:`iris.cube.Cube.regrid` is possible. This enables the use of further
    parameters for existing schemes, as well as the use of more advanced
    schemes for example for unstructured grids.
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
    :class:`esmf_regrid.schemes.ESMFAreaWeighted` use

    .. code-block:: yaml

        my_preprocessor:
          regrid:
            target: 1x1
            scheme:
              reference: esmf_regrid.schemes:ESMFAreaWeighted
    """
    # Load target grid and select appropriate scheme
    target_grid_cube = _get_target_grid_cube(
        cube,
        target_grid,
        lat_offset=lat_offset,
        lon_offset=lon_offset,
    )

    # Horizontal grids from source and target (almost) match
    # -> Return source cube with target coordinates
    if _horizontal_grid_is_close(cube, target_grid_cube):
        for coord in ['latitude', 'longitude']:
            cube.coord(coord).points = (
                target_grid_cube.coord(coord).core_points())
            cube.coord(coord).bounds = (
                target_grid_cube.coord(coord).core_bounds())
        return cube

    # Load scheme and reuse existing regridder if possible
    if isinstance(scheme, str):
        scheme = scheme.lower()
    regridder = _get_regridder(cube, target_grid_cube, scheme, cache_weights)

    # Rechunk and actually perform the regridding
    cube = _rechunk(cube, target_grid_cube)
    cube = regridder(cube)

    return cube


def _cache_clear():
    """Clear regridding weights cache."""
    _CACHED_REGRIDDERS.clear()


regrid.cache_clear = _cache_clear  # type: ignore


def _rechunk(cube: Cube, target_grid: Cube) -> Cube:
    """Re-chunk cube with optimal chunk sizes for target grid."""
    if not cube.has_lazy_data() or cube.ndim < 3:
        # Only rechunk lazy multidimensional data
        return cube

    lon_coord = target_grid.coord(axis='X')
    lat_coord = target_grid.coord(axis='Y')
    if lon_coord.ndim != 1 or lat_coord.ndim != 1:
        # This function only supports 1D lat/lon coordinates.
        return cube

    lon_dim, = target_grid.coord_dims(lon_coord)
    lat_dim, = target_grid.coord_dims(lat_coord)
    grid_indices = sorted((lon_dim, lat_dim))
    target_grid_shape = tuple(target_grid.shape[i] for i in grid_indices)

    if 2 * np.prod(cube.shape[-2:]) > np.prod(target_grid_shape):
        # Only rechunk if target grid is more than a factor of 2 larger,
        # because rechunking will keep the original chunk in memory.
        return cube

    data = cube.lazy_data()

    # Compute a good chunk size for the target array
    tgt_shape = data.shape[:-2] + target_grid_shape
    tgt_chunks = data.chunks[:-2] + target_grid_shape
    tgt_data = da.empty(tgt_shape, dtype=data.dtype, chunks=tgt_chunks)
    tgt_data = tgt_data.rechunk({i: "auto" for i in range(cube.ndim - 2)})

    # Adjust chunks to source array and rechunk
    chunks = tgt_data.chunks[:-2] + data.shape[-2:]
    cube.data = data.rechunk(chunks)

    return cube


def _horizontal_grid_is_close(cube1: Cube, cube2: Cube) -> bool:
    """Check if two cubes have the same horizontal grid definition.

    The result of the function is a boolean answer, if both cubes have the
    same horizontal grid definition. The function checks both longitude and
    latitude, based on extent and resolution.

    Note
    ----
    The current implementation checks if the bounds and the grid shapes are the
    same. Exits on first difference.

    Parameters
    ----------
    cube1:
        The first of the cubes to be checked.
    cube2:
        The second of the cubes to be checked.

    Returns
    -------
    bool
        ``True`` if grids are close; ``False`` if not.
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
    result = Cube(data, **kwargs)

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

    if cube.has_lazy_data():
        # Make source levels lazy if cube has lazy data.
        src_points = src_levels.lazy_points()
    else:
        src_points = src_levels.core_points()

    # Broadcast the source cube vertical coordinate to fully describe the
    # spatial extent that will be interpolated.
    src_levels_broadcast = broadcast_to_shape(
        src_points,
        shape=cube.shape,
        chunks=cube.lazy_data().chunks if cube.has_lazy_data() else None,
        dim_map=cube.coord_dims(src_levels),
    )

    # Make the target levels lazy if the input data is lazy.
    if cube.has_lazy_data() and isinstance(src_points, da.Array):
        levels = da.asarray(levels)

    # force mask onto data as nan's
    npx = get_array_module(cube.core_data())
    data = npx.ma.filled(cube.core_data(), np.nan)

    # Perform vertical interpolation.
    new_data = stratify.interpolate(
        levels,
        src_levels_broadcast,
        data,
        axis=z_axis,
        interpolation=interpolation,
        extrapolation=extrapolation,
    )

    # Calculate the mask based on the any NaN values in the interpolated data.
    new_data = npx.ma.masked_where(npx.isnan(new_data), new_data)

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


def _rechunk_aux_factory_dependencies(
    cube: iris.cube.Cube,
    coord_name: str,
) -> iris.cube.Cube:
    """Rechunk coordinate aux factory dependencies.

    This ensures that the resulting coordinate has reasonably sized
    chunks that are aligned with the cube data for optimal computational
    performance.
    """
    # Workaround for https://github.com/SciTools/iris/issues/5457
    try:
        factory = cube.aux_factory(coord_name)
    except iris.exceptions.CoordinateNotFoundError:
        return cube

    cube = cube.copy()
    cube_chunks = cube.lazy_data().chunks
    for coord in factory.dependencies.values():
        coord_dims = cube.coord_dims(coord)
        if coord_dims:
            coord = coord.copy()
            chunks = tuple(cube_chunks[i] for i in coord_dims)
            coord.points = coord.lazy_points().rechunk(chunks)
            if coord.has_bounds():
                coord.bounds = coord.lazy_bounds().rechunk(chunks + (None, ))
            cube.replace_coord(coord)
    return cube


@preserve_float_dtype
def extract_levels(
    cube: iris.cube.Cube,
    levels: np.typing.ArrayLike | da.Array,
    scheme: str,
    coordinate: Optional[str] = None,
    rtol: float = 1e-7,
    atol: Optional[float] = None,
):
    """Perform vertical interpolation.

    Parameters
    ----------
    cube:
        The source cube to be vertically interpolated.
    levels:
        One or more target levels for the vertical interpolation. Assumed
        to be in the same S.I. units of the source cube vertical dimension
        coordinate. If the requested levels are sufficiently close to the
        levels of the cube, cube slicing will take place instead of
        interpolation.
    scheme:
        The vertical interpolation scheme to use. Choose from
        'linear',
        'nearest',
        'linear_extrapolate',
        'nearest_extrapolate'.
    coordinate:
        The coordinate to interpolate. If specified, pressure levels
        (if present) can be converted to height levels and vice versa using
        the US standard atmosphere. E.g. 'coordinate = altitude' will convert
        existing pressure levels (air_pressure) to height levels (altitude);
        'coordinate = air_pressure' will convert existing height levels
        (altitude) to pressure levels (air_pressure).
    rtol:
        Relative tolerance for comparing the levels in `cube` to the requested
        levels. If the levels are sufficiently close, the requested levels
        will be assigned to the cube and no interpolation will take place.
    atol:
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
    if not isinstance(levels, da.Array):
        levels = np.array(levels, ndmin=1)

    # Try to determine the name of the vertical coordinate automatically
    if coordinate is None:
        coordinate = cube.coord(axis='z', dim_coords=True).name()

    # Add extra coordinates
    coord_names = [coord.name() for coord in cube.coords()]
    if coordinate in coord_names:
        cube = _rechunk_aux_factory_dependencies(cube, coordinate)
    else:
        # Try to calculate air_pressure from altitude coordinate or
        # vice versa using US standard atmosphere for conversion.
        if coordinate == 'air_pressure' and 'altitude' in coord_names:
            # Calculate pressure level coordinate from altitude.
            cube = _rechunk_aux_factory_dependencies(cube, 'altitude')
            add_plev_from_altitude(cube)
        if coordinate == 'altitude' and 'air_pressure' in coord_names:
            # Calculate altitude coordinate from pressure levels.
            cube = _rechunk_aux_factory_dependencies(cube, 'air_pressure')
            add_altitude_from_plev(cube)

    src_levels = cube.coord(coordinate)

    if (src_levels.shape == levels.shape and np.allclose(
            src_levels.core_points(),
            levels,
            rtol=rtol,
            atol=1e-7 *
            np.mean(src_levels.core_points()) if atol is None else atol,
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
        coord_values = {
            name: lambda cell: cell.point in set(levels)  # type: ignore
        }
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
            f"Level definition cmor_table '{cmor_table}' not available")

    if coordinate not in CMOR_TABLES[cmor_table].coords:
        raise ValueError(
            f'Coordinate {coordinate} not available for {cmor_table}')

    cmor = CMOR_TABLES[cmor_table].coords[coordinate]

    if cmor.requested:
        return [float(level) for level in cmor.requested]
    if cmor.value:
        return [float(cmor.value)]

    raise ValueError(
        f'Coordinate {coordinate} in {cmor_table} does not have requested '
        f'values')


def get_reference_levels(dataset):
    """Get level definition from a reference dataset.

    Parameters
    ----------
    dataset: esmvalcore.dataset.Dataset
        Dataset containing the reference files.

    Returns
    -------
    list[float]

    Raises
    ------
    ValueError:
        If the dataset is not defined, the coordinate does not specify any
        levels or the string is badly formatted.
    """
    dataset = dataset.copy()
    dataset.supplementaries.clear()
    dataset.files = [dataset.files[0]]
    cube = dataset.load()
    try:
        coord = cube.coord(axis='Z')
    except iris.exceptions.CoordinateNotFoundError as exc:
        raise ValueError(f'z-coord not available in {dataset.files}') from exc
    return coord.points.tolist()


@preserve_float_dtype
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
    definition : dict(str, float or array of float)
        The coordinate - values pairs to extract
    scheme : str
        The interpolation scheme. 'linear' or 'nearest'. No default.

    Returns
    -------
    iris.cube.Cube
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
