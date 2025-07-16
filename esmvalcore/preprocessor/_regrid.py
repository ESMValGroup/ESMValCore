"""Horizontal and vertical regridding module."""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import os
import re
import ssl
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import dask.array as da
import iris
import iris.coords
import numpy as np
import stratify
from geopy.geocoders import Nominatim
from iris.analysis import AreaWeighted, Linear, Nearest
from iris.cube import Cube
from iris.util import broadcast_to_shape

from esmvalcore.cmor._fixes.shared import (
    add_altitude_from_plev,
    add_plev_from_altitude,
)
from esmvalcore.cmor.table import CMOR_TABLES
from esmvalcore.iris_helpers import has_irregular_grid, has_unstructured_grid
from esmvalcore.preprocessor._shared import (
    _rechunk_aux_factory_dependencies,
    get_array_module,
    get_dims_along_axes,
    preserve_float_dtype,
)
from esmvalcore.preprocessor._supplementary_vars import (
    add_ancillary_variable,
    add_cell_measure,
)
from esmvalcore.preprocessor.regrid_schemes import (
    GenericFuncScheme,
    IrisESMFRegrid,
    UnstructuredLinear,
    UnstructuredNearest,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import ArrayLike

    from esmvalcore.dataset import Dataset

logger = logging.getLogger(__name__)

# Regular expression to parse a "MxN" cell-specification.
_CELL_SPEC = re.compile(
    r"""\A
        \s*(?P<dlon>\d+(\.\d+)?)\s*
        x
        \s*(?P<dlat>\d+(\.\d+)?)\s*
        \Z
     """,
    re.IGNORECASE | re.VERBOSE,
)

# Default fill-value.
_MDI = 1e20

# Stock cube - global grid extents (degrees).
_LAT_MIN = -90.0
_LAT_MAX = 90.0
_LAT_RANGE = _LAT_MAX - _LAT_MIN
_LON_MIN = 0.0
_LON_MAX = 360.0
_LON_RANGE = _LON_MAX - _LON_MIN

NamedPointInterpolationScheme = Literal[
    "linear",
    "nearest",
]

# Supported point interpolation schemes.
POINT_INTERPOLATION_SCHEMES = {
    "linear": Linear(extrapolation_mode="mask"),
    "nearest": Nearest(extrapolation_mode="mask"),
}

NamedHorizontalScheme = Literal[
    "area_weighted",
    "linear",
    "nearest",
]

# Supported horizontal regridding schemes for regular grids (= rectilinear
# grids; i.e., grids that can be described with 1D latitude and 1D longitude
# coordinates orthogonal to each other)
HORIZONTAL_SCHEMES_REGULAR = {
    "area_weighted": AreaWeighted(),
    "linear": Linear(extrapolation_mode="mask"),
    "nearest": Nearest(extrapolation_mode="mask"),
}

# Supported horizontal regridding schemes for irregular grids (= general
# curvilinear grids; i.e., grids that can be described with 2D latitude and 2D
# longitude coordinates with common dimensions)
HORIZONTAL_SCHEMES_IRREGULAR = {
    "area_weighted": IrisESMFRegrid(method="conservative"),
    "linear": IrisESMFRegrid(method="bilinear"),
    "nearest": IrisESMFRegrid(method="nearest"),
}

# Supported horizontal regridding schemes for meshes
# https://scitools-iris.readthedocs.io/en/stable/further_topics/ugrid/index.html
HORIZONTAL_SCHEMES_MESH = {
    "area_weighted": IrisESMFRegrid(method="conservative"),
    "linear": IrisESMFRegrid(method="bilinear"),
    "nearest": IrisESMFRegrid(method="nearest"),
}

# Supported horizontal regridding schemes for unstructured grids (i.e., grids,
# that can be described with 1D latitude and 1D longitude coordinate with
# common dimensions)
HORIZONTAL_SCHEMES_UNSTRUCTURED = {
    "linear": UnstructuredLinear(),
    "nearest": UnstructuredNearest(),
}

NamedVerticalScheme = Literal[
    "linear",
    "nearest",
    "linear_extrapolate",
    "nearest_extrapolate",
]

# Supported vertical interpolation schemes.
VERTICAL_SCHEMES: tuple[NamedVerticalScheme, ...] = (
    "linear",
    "nearest",
    "linear_extrapolate",
    "nearest_extrapolate",
)


def parse_cell_spec(spec: str) -> tuple[float, float]:
    """Parse an MxN cell specification string.

    Parameters
    ----------
    spec:
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
        emsg = "Invalid MxN cell specification for grid, got {!r}."
        raise ValueError(emsg.format(spec))

    cell_group = cell_match.groupdict()
    dlon = float(cell_group["dlon"])
    dlat = float(cell_group["dlat"])

    if (np.trunc(_LON_RANGE / dlon) * dlon) != _LON_RANGE:
        emsg = (
            "Invalid longitude delta in MxN cell specification "
            "for grid, got {!r}."
        )
        raise ValueError(emsg.format(dlon))

    if (np.trunc(_LAT_RANGE / dlat) * dlat) != _LAT_RANGE:
        emsg = (
            "Invalid latitude delta in MxN cell specification "
            "for grid, got {!r}."
        )
        raise ValueError(emsg.format(dlat))

    return dlon, dlat


def _generate_cube_from_dimcoords(
    latdata: np.ndarray | da.Array,
    londata: np.ndarray | da.Array,
    circular: bool = False,
) -> Cube:
    """Generate cube from lat/lon points.

    Parameters
    ----------
    latdata:
        List of latitudes.
    londata:
        List of longitudes.
    circular
        Wrap longitudes around the full great circle. Bounds will not be
        generated for circular coordinates.

    Returns
    -------
    iris.cube.Cube
    """
    lats = iris.coords.DimCoord(
        latdata,
        standard_name="latitude",
        units="degrees_north",
        var_name="lat",
        circular=circular,
    )

    lons = iris.coords.DimCoord(
        londata,
        standard_name="longitude",
        units="degrees_east",
        var_name="lon",
        circular=circular,
    )

    if not circular:
        # cannot guess bounds for wrapped coordinates
        lats.guess_bounds()
        lons.guess_bounds()

    # Construct the resultant stock cube, with dummy data.
    shape = (latdata.size, londata.size)
    dummy = np.empty(shape, dtype=np.int32)
    coords_spec = [(lats, 0), (lons, 1)]
    return Cube(dummy, dim_coords_and_dims=coords_spec)


@functools.lru_cache
def _global_stock_cube(
    spec: str,
    lat_offset: bool = True,
    lon_offset: bool = True,
) -> Cube:
    """Create a stock cube.

    Create a global cube with M degree-east by N degree-north regular grid
    cells.

    The longitude range is from 0 to 360 degrees. The latitude range is from
    -90 to 90 degrees. Each cell grid point is calculated as the mid-point of
    the associated MxN cell.

    Parameters
    ----------
    spec
        Specifies the 'MxN' degree cell-specification for the global grid.
    lat_offset
        Offset the grid centers of the latitude coordinate w.r.t. the
        pole by half a grid step. This argument is ignored if `target_grid`
        is a cube or file.
    lon_offset
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
        latdata = np.linspace(
            _LAT_MIN + mid_dlat,
            _LAT_MAX - mid_dlat,
            int(_LAT_RANGE / dlat),
        )
    else:
        latdata = np.linspace(_LAT_MIN, _LAT_MAX, int(_LAT_RANGE / dlat) + 1)

    # Construct the longitude coordinat, with bounds.
    if lon_offset:
        londata = np.linspace(
            _LON_MIN + mid_dlon,
            _LON_MAX - mid_dlon,
            int(_LON_RANGE / dlon),
        )
    else:
        londata = np.linspace(
            _LON_MIN,
            _LON_MAX - dlon,
            int(_LON_RANGE / dlon),
        )

    return _generate_cube_from_dimcoords(latdata=latdata, londata=londata)


def _spec_to_latlonvals(
    *,
    start_latitude: float,
    end_latitude: float,
    step_latitude: float,
    start_longitude: float,
    end_longitude: float,
    step_longitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Define lat/lon values from spec.

    Create a regional cube starting defined by the target specification.

    The latitude must be between -90 and +90. The longitude is not bounded, but
    wraps around the full great circle.

    Parameters
    ----------
    start_latitude:
        Latitude value of the first grid cell center (start point). The grid
        includes this value.
    end_latitude:
        Latitude value of the last grid cell center (end point). The grid
        includes this value only if it falls on a grid point. Otherwise, it
        cuts off at the previous value.
    step_latitude:
        Latitude distance between the centers of two neighbouring cells.
    start_longitude:
        Latitude value of the first grid cell center (start point). The grid
        includes this value.
    end_longitude:
        Longitude value of the last grid cell center (end point). The grid
        includes this value only if it falls on a grid point. Otherwise, it
        cuts off at the previous value.
    step_longitude:
        Longitude distance between the centers of two neighbouring cells.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Longitudes, Latitudes.

    """
    if step_latitude == 0:
        msg = f"Latitude step cannot be 0, got step_latitude={step_latitude}."
        raise ValueError(msg)

    if step_longitude == 0:
        msg = (
            f"Longitude step cannot be 0, got step_longitude={step_longitude}."
        )
        raise ValueError(msg)

    if (start_latitude < _LAT_MIN) or (end_latitude > _LAT_MAX):
        msg = (
            f"Latitude values must lie between {_LAT_MIN}:{_LAT_MAX}, "
            f"got start_latitude={start_latitude}:end_latitude={end_latitude}."
        )
        raise ValueError(msg)

    def get_points(start, stop, step):
        """Calculate grid points."""
        # use Decimal to avoid floating point errors
        num = int(Decimal(stop - start) // Decimal(str(step)))
        stop = start + num * step
        return np.linspace(start, stop, num + 1)

    latitudes = get_points(start_latitude, end_latitude, step_latitude)
    longitudes = get_points(start_longitude, end_longitude, step_longitude)

    return latitudes, longitudes


def _regional_stock_cube(spec: dict[str, Any]) -> Cube:
    """Create a regional stock cube.

    Returns
    -------
    iris.cube.Cube
    """
    latdata, londata = _spec_to_latlonvals(**spec)

    cube = _generate_cube_from_dimcoords(
        latdata=latdata,
        londata=londata,
        circular=True,
    )

    def add_bounds_from_step(
        coord: iris.coords.DimCoord | iris.coords.AuxCoord,
        step: float,
    ) -> None:
        """Calculate bounds from the given step."""
        bound = step / 2
        points = coord.points
        coord.bounds = np.vstack((points - bound, points + bound)).T

    add_bounds_from_step(cube.coord("latitude"), spec["step_latitude"])
    add_bounds_from_step(cube.coord("longitude"), spec["step_longitude"])

    return cube


def extract_location(
    cube: Cube,
    location: str,
    scheme: NamedPointInterpolationScheme,
) -> Cube:
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
    cube:
        The source cube to extract a point from.
    location:
        The reference location. Examples: 'mount everest',
        'romania','new york, usa'
    scheme:
        The interpolation scheme. 'linear' or 'nearest'. No default.

    Returns
    -------
    iris.cube.Cube
        Returns a cube with the extracted point, and with adjusted latitude and
        longitude coordinates.

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
        msg = (
            "Location needs to be specified."
            " Examples: 'mount everest', 'romania',"
            " 'new york, usa'"
        )
        raise ValueError(msg)
    if scheme is None:
        msg = (
            "Interpolation scheme needs to be specified."
            " Use either 'linear' or 'nearest'."
        )
        raise ValueError(msg)
    try:
        # Try to use the default SSL context, see
        # https://github.com/ESMValGroup/ESMValCore/issues/2012 for more
        # information.
        ssl_context = ssl.create_default_context()
        geolocator = Nominatim(
            user_agent="esmvalcore",
            ssl_context=ssl_context,
        )
    except ssl.SSLError:
        logger.warning(
            "ssl.create_default_context() encountered a problem, not using it.",
        )
        geolocator = Nominatim(user_agent="esmvalcore")
    geolocation = geolocator.geocode(location)
    if geolocation is None:
        msg = f"Requested location {location} can not be found."
        raise ValueError(msg)
    logger.info(
        "Extracting data for %s (%s °N, %s °E)",
        geolocation,
        geolocation.latitude,
        geolocation.longitude,
    )

    return extract_point(
        cube,
        geolocation.latitude,
        geolocation.longitude,
        scheme,
    )


def extract_point(
    cube: Cube,
    latitude: ArrayLike,
    longitude: ArrayLike,
    scheme: NamedPointInterpolationScheme,
) -> Cube:
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
    cube:
        The source cube to extract a point from.
    latitude:
        The latitude of the point.
    longitude:
        The longitude of the point.
    scheme:
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
    loaded_scheme = POINT_INTERPOLATION_SCHEMES.get(scheme.lower())
    if not loaded_scheme:
        raise ValueError(msg)

    point = [("latitude", latitude), ("longitude", longitude)]
    return cube.interpolate(point, scheme=loaded_scheme)


def is_dataset(dataset: Any) -> bool:
    """Test if something is an `esmvalcore.dataset.Dataset`."""
    # Use this function to avoid circular imports
    return hasattr(dataset, "facets")


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
        target_grid_cube = _global_stock_cube(
            target_grid,
            lat_offset,
            lon_offset,
        )
        # Align the target grid coordinate system to the source
        # coordinate system.
        src_cs = cube.coord_system()
        xcoord = target_grid_cube.coord(axis="x", dim_coords=True)
        ycoord = target_grid_cube.coord(axis="y", dim_coords=True)
        xcoord.coord_system = src_cs
        ycoord.coord_system = src_cs
    elif isinstance(target_grid, dict):
        # Generate a target grid from the provided specification,
        target_grid_cube = _regional_stock_cube(target_grid)
    else:
        target_grid_cube = target_grid

    if not isinstance(target_grid_cube, Cube):
        msg = f"Expecting a cube, got {target_grid}."
        raise TypeError(msg)

    return target_grid_cube


def _load_scheme(
    src_cube: Cube,
    tgt_cube: Cube,
    scheme: NamedHorizontalScheme | dict[str, Any],
):
    """Return scheme that can be used in :meth:`iris.cube.Cube.regrid`."""
    loaded_scheme: Any = None

    if isinstance(scheme, dict):
        # Scheme is a dict -> assume this describes a generic regridding scheme
        loaded_scheme = _load_generic_scheme(scheme)
    else:
        # Scheme is a str -> load appropriate regridding scheme depending on
        # the type of input data
        if has_irregular_grid(src_cube) or has_irregular_grid(tgt_cube):
            grid_type = "irregular"
        elif src_cube.mesh is not None or tgt_cube.mesh is not None:
            grid_type = "mesh"
        elif has_unstructured_grid(src_cube):
            grid_type = "unstructured"
        else:
            grid_type = "regular"

        schemes = globals()[f"HORIZONTAL_SCHEMES_{grid_type.upper()}"]
        if scheme not in schemes:
            msg = (
                f"Regridding scheme '{scheme}' not available for {grid_type} "
                f"data, expected one of: {', '.join(schemes)}"
            )
            raise ValueError(msg)
        loaded_scheme = schemes[scheme]

    logger.debug("Loaded regridding scheme %s", loaded_scheme)

    return loaded_scheme


def _load_generic_scheme(scheme: dict[str, Any]):
    """Load generic regridding scheme."""
    scheme = dict(scheme)  # do not overwrite original scheme

    try:
        object_ref = scheme.pop("reference")
    except KeyError as key_err:
        msg = "No reference specified for generic regridding."
        raise ValueError(msg) from key_err
    module_name, separator, scheme_name = object_ref.partition(":")
    try:
        obj: Any = importlib.import_module(module_name)
    except ImportError as import_err:
        msg = (
            f"Could not import specified generic regridding module "
            f"'{module_name}'. Please double check spelling and that the "
            f"required module is installed."
        )
        raise ValueError(msg) from import_err
    if separator:
        for attr in scheme_name.split("."):
            obj = getattr(obj, attr)

    # If `obj` is a function that requires `src_cube` and `grid_cube`, use
    # GenericFuncScheme
    scheme_args = inspect.getfullargspec(obj).args
    if "src_cube" in scheme_args and "grid_cube" in scheme_args:
        loaded_scheme = GenericFuncScheme(obj, **scheme)
    else:
        loaded_scheme = obj(**scheme)

    return loaded_scheme


_CACHED_REGRIDDERS: dict[tuple, dict] = {}


def _get_regridder(
    src_cube: Cube,
    tgt_cube: Cube,
    scheme: NamedHorizontalScheme | dict,
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
            for key, regridder in _CACHED_REGRIDDERS[name_shape_key].items():
                if key == coord_key:
                    return regridder

        # Regridder is not in cached -> return a new one and cache it
        loaded_scheme = _load_scheme(src_cube, tgt_cube, scheme)
        regridder = loaded_scheme.regridder(src_cube, tgt_cube)
        _CACHED_REGRIDDERS.setdefault(name_shape_key, {})
        _CACHED_REGRIDDERS[name_shape_key][coord_key] = regridder

    # (2) Weights caching disabled
    else:
        loaded_scheme = _load_scheme(src_cube, tgt_cube, scheme)
        regridder = loaded_scheme.regridder(src_cube, tgt_cube)

    return regridder


def _get_coord_key(
    src_cube: Cube,
    tgt_cube: Cube,
) -> tuple[iris.coords.DimCoord | iris.coords.AuxCoord, ...]:
    """Get dict key from coordinates."""
    src_lat = src_cube.coord("latitude")
    src_lon = src_cube.coord("longitude")
    tgt_lat = tgt_cube.coord("latitude")
    tgt_lon = tgt_cube.coord("longitude")
    return (src_lat, src_lon, tgt_lat, tgt_lon)


def _get_name_and_shape_key(
    src_cube: Cube,
    tgt_cube: Cube,
    scheme: NamedHorizontalScheme | dict,
) -> tuple[str, tuple[int, ...]]:
    """Get dict key from scheme name and coordinate shapes."""
    name = str(scheme)
    shapes = [c.shape for c in _get_coord_key(src_cube, tgt_cube)]
    return (name, *shapes)


@preserve_float_dtype
def regrid(
    cube: Cube,
    target_grid: Cube | Dataset | Path | str | dict,
    scheme: NamedHorizontalScheme | dict,
    lat_offset: bool = True,
    lon_offset: bool = True,
    cache_weights: bool = False,
    use_src_coords: Iterable[str] = ("latitude", "longitude"),
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
    use_src_coords:
        If there are multiple horizontal coordinates available in the source
        cube, only use horizontal coordinates with these standard names.

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
    # Remove unwanted coordinates from the source cube.
    cube = cube.copy()
    use_src_coords = set(use_src_coords)
    for axis in ("X", "Y"):
        coords = cube.coords(axis=axis)
        if len(coords) > 1:
            for coord in coords:
                if coord.standard_name not in use_src_coords:
                    cube.remove_coord(coord)

    target_grid_cube = _get_target_grid_cube(
        cube,
        target_grid,
        lat_offset=lat_offset,
        lon_offset=lon_offset,
    )

    # Horizontal grids from source and target (almost) match
    # -> Return source cube with target coordinates
    if cube.coords("latitude") and cube.coords("longitude"):
        if _horizontal_grid_is_close(cube, target_grid_cube):
            for coord in ["latitude", "longitude"]:
                is_dim_coord = cube.coords(coord, dim_coords=True)
                coord_dims = cube.coord_dims(coord)
                cube.remove_coord(coord)
                target_coord = target_grid_cube.coord(coord).copy()
                if is_dim_coord:
                    cube.add_dim_coord(target_coord, coord_dims)
                else:
                    cube.add_aux_coord(target_coord, coord_dims)
            return cube

    # Load scheme and reuse existing regridder if possible
    if isinstance(scheme, str):
        scheme = scheme.lower()  # type: ignore
    regridder = _get_regridder(cube, target_grid_cube, scheme, cache_weights)

    # Rechunk and actually perform the regridding
    cube = _rechunk(cube, target_grid_cube)
    return regridder(cube)


def _cache_clear():
    """Clear regridding weights cache."""
    _CACHED_REGRIDDERS.clear()


regrid.cache_clear = _cache_clear  # type: ignore


def _rechunk(cube: Cube, target_grid: Cube) -> Cube:
    """Re-chunk cube with optimal chunk sizes for target grid."""
    if not cube.has_lazy_data():
        # Only rechunk lazy data
        return cube

    # Extract grid dimension information from source cube
    src_grid_indices = get_dims_along_axes(cube, ["X", "Y"])
    src_grid_shape = tuple(cube.shape[i] for i in src_grid_indices)
    src_grid_ndims = len(src_grid_indices)

    # Extract grid dimension information from target cube.
    tgt_grid_indices = get_dims_along_axes(target_grid, ["X", "Y"])
    tgt_grid_shape = tuple(target_grid.shape[i] for i in tgt_grid_indices)
    tgt_grid_ndims = len(tgt_grid_indices)

    if 2 * np.prod(src_grid_shape) > np.prod(tgt_grid_shape):
        # Only rechunk if target grid is more than a factor of 2 larger,
        # because rechunking will keep the original chunk in memory.
        return cube

    # Compute a good chunk size for the target array
    # This uses the fact that horizontal dimension(s) are the last dimension(s)
    # of the input cube and also takes into account that iris regridding needs
    # unchunked data along the grid dimensions.
    data = cube.lazy_data()
    tgt_shape = data.shape[:-src_grid_ndims] + tgt_grid_shape
    tgt_chunks = data.chunks[:-src_grid_ndims] + tgt_grid_shape

    tgt_data = da.empty(tgt_shape, chunks=tgt_chunks, dtype=data.dtype)
    tgt_data = tgt_data.rechunk(
        dict.fromkeys(range(tgt_data.ndim - tgt_grid_ndims), "auto"),
    )

    # Adjust chunks to source array and rechunk
    chunks = tgt_data.chunks[:-tgt_grid_ndims] + data.shape[-src_grid_ndims:]
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
    for coord in ["latitude", "longitude"]:
        coord1 = cube1.coord(coord)
        coord2 = cube2.coord(coord)

        if coord1.shape != coord2.shape:
            return False

        if not np.allclose(coord1.bounds, coord2.bounds):
            return False

    return True


def _create_cube(
    src_cube: Cube,
    data: np.ndarray | da.Array,
    src_levels: iris.coords.DimCoord | iris.coords.AuxCoord,
    levels: np.ndarray | da.Array,
) -> Cube:
    """Generate a new cube with the interpolated data.

    The resultant cube is seeded with `src_cube` metadata and coordinates,
    excluding any source coordinates that span the associated vertical
    dimension. The `levels` of interpolation are used along with the
    associated source cube vertical coordinate metadata to add a new
    vertical coordinate to the resultant cube.

    Parameters
    ----------
    src_cube:
        The source cube that was vertically interpolated.
    data:
        The payload resulting from interpolating the source cube over the
        specified levels.
    src_levels:
        Vertical levels of the source data.
    levels:
        The vertical levels of interpolation.

    Returns
    -------
    iris.cube.Cube

    .. note::

        If there is only one level of interpolation, the resultant cube
        will be collapsed over the associated vertical dimension, and a
        scalar vertical coordinate will be added.
    """
    # Get the source cube vertical coordinate and associated dimension.
    z_coord = src_cube.coord(axis="z", dim_coords=True)
    (z_dim,) = src_cube.coord_dims(z_coord)

    if (len(levels.shape) == 1) and (data.shape[z_dim] != levels.size):
        emsg = (
            "Mismatch between data and levels for data dimension {!r}, "
            "got data shape {!r} with levels shape {!r}."
        )
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
        "standard_name": metadata.standard_name,
        "long_name": metadata.long_name,
        "var_name": metadata.var_name,
        "units": metadata.units,
        "attributes": metadata.attributes,
        "coord_system": metadata.coord_system,
        "climatological": metadata.climatological,
    }

    try:
        coord = iris.coords.DimCoord(levels, **kwargs)
        result.add_dim_coord(coord, z_dim)
    except ValueError:
        coord = iris.coords.AuxCoord(levels, **kwargs)
        result.add_aux_coord(
            coord,
            z_dim if len(levels.shape) == 1 else np.arange(len(coord.shape)),
        )

    # Collapse the z-dimension for the scalar case.
    if levels.size == 1:
        slicer: list[slice | int] = [slice(None)] * result.ndim
        slicer[z_dim] = 0
        result = result[tuple(slicer)]

    return result


def _vertical_interpolate(
    cube: Cube,
    src_levels: iris.coords.DimCoord | iris.coords.AuxCoord,
    levels: np.ndarray | da.Array,
    interpolation: str,
    extrapolation: str,
) -> Cube:
    """Perform vertical interpolation."""
    # Determine the source levels and axis for vertical interpolation.
    (z_axis,) = cube.coord_dims(cube.coord(axis="z", dim_coords=True))

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


def _preserve_fx_vars(cube: iris.cube.Cube, result: iris.cube.Cube) -> None:
    vertical_dim = set(cube.coord_dims(cube.coord(axis="z", dim_coords=True)))
    if cube.cell_measures():
        for measure in cube.cell_measures():
            measure_dims = set(cube.cell_measure_dims(measure))
            if vertical_dim.intersection(measure_dims):
                logger.warning(
                    "Discarding use of z-axis dependent cell measure %s "
                    "in variable %s, as z-axis has been interpolated",
                    measure.var_name,
                    result.var_name,
                )
            else:
                add_cell_measure(result, measure, measure.measure)
    if cube.ancillary_variables():
        for ancillary_var in cube.ancillary_variables():
            ancillary_dims = set(cube.ancillary_variable_dims(ancillary_var))
            if vertical_dim.intersection(ancillary_dims):
                logger.warning(
                    "Discarding use of z-axis dependent ancillary variable %s "
                    "in variable %s, as z-axis has been interpolated",
                    ancillary_var.var_name,
                    result.var_name,
                )
            else:
                # Create cube and add coordinates to ancillary variable
                ancillary_coords: list[
                    tuple[iris.coords.AncillaryVariable, int]
                ] = []
                for i, coord in enumerate(cube.coords()):
                    if i in ancillary_dims:
                        coord_idx = len(ancillary_coords)
                        ancillary_coords.append((coord.copy(), coord_idx))
                ancillary_cube = iris.cube.Cube(
                    ancillary_var.core_data(),
                    standard_name=ancillary_var.standard_name,
                    long_name=ancillary_var.long_name,
                    units=ancillary_var.units,
                    var_name=ancillary_var.var_name,
                    attributes=ancillary_var.attributes,
                    dim_coords_and_dims=ancillary_coords,
                )
                add_ancillary_variable(result, ancillary_cube)


def parse_vertical_scheme(scheme: NamedVerticalScheme) -> tuple[str, str]:
    """Parse the scheme provided for level extraction.

    Parameters
    ----------
    scheme:
        The vertical interpolation scheme to use. Choose from
        'linear',
        'nearest',
        'linear_extrapolate',
        'nearest_extrapolate'.

    Returns
    -------
    tuple[str, str]
        A tuple containing the interpolation and extrapolation scheme.
    """
    # Check if valid scheme is given
    if scheme not in VERTICAL_SCHEMES:
        msg = (
            f"Unknown vertical interpolation scheme, got '{scheme}', possible "
            f"schemes are {VERTICAL_SCHEMES}"
        )
        raise ValueError(msg)

    # This allows us to put level 0. to load the ocean surface.
    extrap_scheme = "nan"

    if scheme == "linear_extrapolate":
        scheme = "linear"
        extrap_scheme = "nearest"

    if scheme == "nearest_extrapolate":
        scheme = "nearest"
        extrap_scheme = "nearest"

    return scheme, extrap_scheme


@preserve_float_dtype
def extract_levels(
    cube: iris.cube.Cube,
    levels: np.typing.ArrayLike | da.Array,
    scheme: NamedVerticalScheme,
    coordinate: str | None = None,
    rtol: float = 1e-7,
    atol: float | None = None,
) -> Cube:
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
        coordinate = cube.coord(axis="z", dim_coords=True).name()

    # Add extra coordinates
    coord_names = [coord.name() for coord in cube.coords()]
    if coordinate in coord_names:
        cube = _rechunk_aux_factory_dependencies(cube, coordinate)
    else:
        # Try to calculate air_pressure from altitude coordinate or
        # vice versa using US standard atmosphere for conversion.
        if coordinate == "air_pressure" and "altitude" in coord_names:
            # Calculate pressure level coordinate from altitude.
            cube = _rechunk_aux_factory_dependencies(cube, "altitude")
            add_plev_from_altitude(cube)
        if coordinate == "altitude" and "air_pressure" in coord_names:
            # Calculate altitude coordinate from pressure levels.
            cube = _rechunk_aux_factory_dependencies(cube, "air_pressure")
            add_altitude_from_plev(cube)

    src_levels = cube.coord(coordinate)

    if src_levels.shape == levels.shape and np.allclose(
        src_levels.core_points(),
        levels,
        rtol=rtol,
        atol=1e-7 * np.mean(src_levels.core_points())
        if atol is None
        else atol,
    ):
        # Only perform vertical extraction/interpolation if the source
        # and target levels are not "similar" enough.
        result = cube
        # Set the levels to the requested values
        src_levels.points = levels
    elif (
        len(src_levels.shape) == 1
        and len(levels.shape) == 1
        and set(levels.flatten()).issubset(set(src_levels.points))
    ):
        # If all target levels exist in the source cube, simply extract them.
        name = src_levels.name()
        coord_values = {
            name: lambda cell: cell.point in set(levels),  # type: ignore
        }
        constraint = iris.Constraint(coord_values=coord_values)
        result = cube.extract(constraint)
        # Ensure the constraint did not fail.
        if not result:
            emsg = "Failed to extract levels {!r} from cube {!r}."
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


def get_cmor_levels(cmor_table: str, coordinate: str) -> list[float]:
    """Get level definition from a CMOR coordinate.

    Parameters
    ----------
    cmor_table:
        CMOR table name
    coordinate:
        CMOR coordinate name

    Returns
    -------
    list[float]

    Raises
    ------
    ValueError:
        If the CMOR table is not defined, the coordinate does not specify any
        levels or the string is badly formatted.
    """
    if cmor_table not in CMOR_TABLES:
        msg = f"Level definition cmor_table '{cmor_table}' not available"
        raise ValueError(msg)

    if coordinate not in CMOR_TABLES[cmor_table].coords:
        msg = f"Coordinate {coordinate} not available for {cmor_table}"
        raise ValueError(msg)

    cmor = CMOR_TABLES[cmor_table].coords[coordinate]

    if cmor.requested:
        return [float(level) for level in cmor.requested]
    if cmor.value:
        return [float(cmor.value)]

    msg = (
        f"Coordinate {coordinate} in {cmor_table} does not have requested "
        f"values"
    )
    raise ValueError(msg)


def get_reference_levels(dataset: Dataset) -> list[float]:
    """Get level definition from a reference dataset.

    Parameters
    ----------
    dataset:
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
        coord = cube.coord(axis="Z")
    except iris.exceptions.CoordinateNotFoundError as exc:
        msg = f"z-coord not available in {dataset.files}"
        raise ValueError(msg) from exc
    return coord.points.tolist()


@preserve_float_dtype
def extract_coordinate_points(
    cube: Cube,
    definition: dict[str, ArrayLike],
    scheme: NamedPointInterpolationScheme,
) -> Cube:
    """Extract points from any coordinate with interpolation.

    Multiple points can also be extracted, by supplying an array of
    coordinates. The resulting point cube will match the respective
    coordinates to those of the input coordinates.
    If the input coordinate is a scalar, the dimension will be a
    scalar in the output cube.

    Parameters
    ----------
    cube:
        The source cube to extract a point from.
    definition:
        The coordinate - values pairs to extract
    scheme:
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
    loaded_scheme = POINT_INTERPOLATION_SCHEMES.get(scheme.lower())
    if not loaded_scheme:
        raise ValueError(msg)
    return cube.interpolate(definition.items(), scheme=loaded_scheme)
