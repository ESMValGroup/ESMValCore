"""Area operations on data cubes.

Allows for selecting data subsets using certain latitude and longitude
bounds; selecting geographical regions; constructing area averages; etc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import fiona
import iris
import numpy as np
import shapely
import shapely.ops
from dask import array as da
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError

from esmvalcore.iris_helpers import ignore_iris_vague_metadata_warnings
from esmvalcore.preprocessor._shared import (
    apply_mask,
    get_dims_along_axes,
    get_iris_aggregator,
    get_normalized_cube,
    preserve_float_dtype,
    try_adding_calculated_cell_area,
    update_weights_kwargs,
)
from esmvalcore.preprocessor._supplementary_vars import (
    add_ancillary_variable,
    add_cell_measure,
    register_supplementaries,
    remove_supplementary_variables,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from esmvalcore.config import Session

logger = logging.getLogger(__name__)

SHAPE_ID_KEYS: tuple[str, ...] = ("name", "NAME", "Name", "id", "ID")


def extract_region(  # noqa: C901
    cube: Cube,
    start_longitude: float,
    end_longitude: float,
    start_latitude: float,
    end_latitude: float,
) -> Cube:
    """Extract a region from a cube.

    Function that subsets a cube on a box (start_longitude, end_longitude,
    start_latitude, end_latitude).

    Parameters
    ----------
    cube:
        Input data cube.
    start_longitude:
        Western boundary longitude.
    end_longitude:
        Eastern boundary longitude.
    start_latitude:
        Southern Boundary latitude.
    end_latitude:
        Northern Boundary Latitude.

    Returns
    -------
    iris.cube.Cube
        Smaller cube.

    """
    # first examine if any cell_measures are present
    cell_measures = cube.cell_measures()
    ancil_vars = cube.ancillary_variables()

    if abs(start_latitude) > 90.0:
        msg = f"Invalid start_latitude: {start_latitude}"
        raise ValueError(msg)
    if abs(end_latitude) > 90.0:
        msg = f"Invalid end_latitude: {end_latitude}"
        raise ValueError(msg)
    if cube.coord("latitude").ndim == 1:
        # Iris checks if any point of the cell is inside the region.
        # To check only the center, ignore_bounds must be set to True.
        region_subset = cube.intersection(
            longitude=(start_longitude, end_longitude),
            latitude=(start_latitude, end_latitude),
            ignore_bounds=True,
        )
    else:
        region_subset = _extract_irregular_region(
            cube,
            start_longitude,
            end_longitude,
            start_latitude,
            end_latitude,
        )

    # put back cell measures and ancillary_variables;
    # iris.Cube.cube.intersection removes them both.
    # This is a workaround resulting from opening upstream
    # https://github.com/SciTools/iris/issues/5413
    # When removing this block after iris have a fix, make sure to remove the
    # test too tests/integration/preprocessor/_extract_region/

    def _extract_region_from_dim_metadata(dim_metadata, dim_metadata_dims):
        """Extract region from dimensional metadata."""
        idx = tuple(
            slice(None) if d in dim_metadata_dims else 0
            for d in range(cube.ndim)
        )
        subcube = cube[idx].copy(dim_metadata.core_data())
        for sub_cm in subcube.cell_measures():
            subcube.remove_cell_measure(sub_cm)
        for sub_av in subcube.ancillary_variables():
            subcube.remove_ancillary_variable(sub_av)
        subcube = extract_region(
            subcube,
            start_longitude,
            end_longitude,
            start_latitude,
            end_latitude,
        )
        return dim_metadata.copy(subcube.core_data())

    # Step 1: cell measures
    if cell_measures and not region_subset.cell_measures():
        for cell_measure in cell_measures:
            cell_measure_dims = cube.cell_measure_dims(cell_measure)
            cell_measure_subset = _extract_region_from_dim_metadata(
                cell_measure,
                cell_measure_dims,
            )
            region_subset.add_cell_measure(
                cell_measure_subset,
                cell_measure_dims,
            )

    # Step 2: ancillary variables
    if ancil_vars and not region_subset.ancillary_variables():
        for ancil_var in ancil_vars:
            ancil_var_dims = cube.ancillary_variable_dims(ancil_var)
            ancil_var_subset = _extract_region_from_dim_metadata(
                ancil_var,
                ancil_var_dims,
            )
            region_subset.add_ancillary_variable(
                ancil_var_subset,
                ancil_var_dims,
            )

    return region_subset


def _extract_irregular_region(
    cube,
    start_longitude,
    end_longitude,
    start_latitude,
    end_latitude,
):
    """Extract a region from a cube on an irregular grid."""
    # Convert longitudes to valid range
    if start_longitude != 360.0:
        start_longitude %= 360.0
    if end_longitude != 360.0:
        end_longitude %= 360.0

    # Select coordinates inside the region
    lats = cube.coord("latitude").points
    lons = (cube.coord("longitude").points + 360.0) % 360.0
    if start_longitude <= end_longitude:
        select_lons = (lons >= start_longitude) & (lons <= end_longitude)
    else:
        select_lons = (lons >= start_longitude) | (lons <= end_longitude)

    if start_latitude <= end_latitude:
        select_lats = (lats >= start_latitude) & (lats <= end_latitude)
    else:
        select_lats = (lats >= start_latitude) | (lats <= end_latitude)

    selection = select_lats & select_lons

    # Crop the selection, but keep rectangular shape
    i_range, j_range = selection.nonzero()
    if i_range.size == 0:
        msg = "No data points available in selected region"
        raise ValueError(msg)
    i_min, i_max = i_range.min(), i_range.max()
    j_min, j_max = j_range.min(), j_range.max()
    i_slice, j_slice = slice(i_min, i_max + 1), slice(j_min, j_max + 1)
    cube = cube[..., i_slice, j_slice]
    selection = selection[i_slice, j_slice]
    # Mask remaining coordinates outside region
    horizontal_dims = get_dims_along_axes(cube, ["X", "Y"])
    cube.data = apply_mask(~selection, cube.core_data(), horizontal_dims)
    return cube


@preserve_float_dtype
def zonal_statistics(
    cube: Cube,
    operator: str,
    normalize: Literal["subtract", "divide"] | None = None,
    **operator_kwargs,
) -> Cube:
    """Compute zonal statistics.

    Parameters
    ----------
    cube:
        Input cube.
    operator:
        The operation. Used to determine the :class:`iris.analysis.Aggregator`
        object used to calculate the statistics. Allowed options are given in
        :ref:`this table <supported_stat_operator>`.
    normalize:
        If given, do not return the statistics cube itself, but rather, the
        input cube, normalized with the statistics cube. Can either be
        `subtract` (statistics cube is subtracted from the input cube) or
        `divide` (input cube is divided by the statistics cube).
    **operator_kwargs:
        Optional keyword arguments for the :class:`iris.analysis.Aggregator`
        object defined by `operator`.

    Returns
    -------
    iris.cube.Cube
        Zonal statistics cube or input cube normalized by statistics cube (see
        `normalize`).

    Raises
    ------
    ValueError
        Error raised if computation on irregular grids is attempted.
        Zonal statistics not yet implemented for irregular grids.

    """
    if cube.coord("longitude").points.ndim >= 2:
        msg = "Zonal statistics on irregular grids not yet implemented"
        raise ValueError(
            msg,
        )
    (agg, agg_kwargs) = get_iris_aggregator(operator, **operator_kwargs)
    with ignore_iris_vague_metadata_warnings():
        result = cube.collapsed("longitude", agg, **agg_kwargs)
    if normalize is not None:
        result = get_normalized_cube(cube, result, normalize)
    return result


@preserve_float_dtype
def meridional_statistics(
    cube: Cube,
    operator: str,
    normalize: Literal["subtract", "divide"] | None = None,
    **operator_kwargs,
) -> Cube:
    """Compute meridional statistics.

    Parameters
    ----------
    cube:
        Input cube.
    operator:
        The operation. Used to determine the :class:`iris.analysis.Aggregator`
        object used to calculate the statistics. Allowed options are given in
        :ref:`this table <supported_stat_operator>`.
    normalize:
        If given, do not return the statistics cube itself, but rather, the
        input cube, normalized with the statistics cube. Can either be
        `subtract` (statistics cube is subtracted from the input cube) or
        `divide` (input cube is divided by the statistics cube).
    **operator_kwargs:
        Optional keyword arguments for the :class:`iris.analysis.Aggregator`
        object defined by `operator`.

    Returns
    -------
    iris.cube.Cube
        Meridional statistics cube.

    Raises
    ------
    ValueError
        Error raised if computation on irregular grids is attempted.
        Zonal statistics not yet implemented for irregular grids.

    """
    if cube.coord("latitude").points.ndim >= 2:
        msg = "Meridional statistics on irregular grids not yet implemented"
        raise ValueError(
            msg,
        )
    (agg, agg_kwargs) = get_iris_aggregator(operator, **operator_kwargs)
    with ignore_iris_vague_metadata_warnings():
        result = cube.collapsed("latitude", agg, **agg_kwargs)
    if normalize is not None:
        result = get_normalized_cube(cube, result, normalize)
    return result


@register_supplementaries(
    variables=["areacella", "areacello"],
    required="prefer_at_least_one",
)
@preserve_float_dtype
def area_statistics(
    cube: Cube,
    operator: str,
    normalize: Literal["subtract", "divide"] | None = None,
    **operator_kwargs,
) -> Cube:
    """Apply a statistical operator in the horizontal plane.

    We assume that the horizontal directions are ['longitude', 'latitude'].

    :ref:`This table <supported_stat_operator>` shows a list of supported
    operators. All operators that support weights are by default weighted with
    the grid cell areas. Note that for area-weighted sums, the units of the
    resulting cube will be multiplied by m :math:`^2`.

    Parameters
    ----------
    cube:
        Input cube. The input cube should have a
        :class:`iris.coords.CellMeasure` named ``'cell_area'``, unless it has
        regular 1D latitude and longitude coordinates so the cell areas can be
        computed using :func:`iris.analysis.cartography.area_weights`.
    operator:
        The operation. Used to determine the :class:`iris.analysis.Aggregator`
        object used to calculate the statistics. Allowed options are given in
        :ref:`this table <supported_stat_operator>`.
    normalize:
        If given, do not return the statistics cube itself, but rather, the
        input cube, normalized with the statistics cube. Can either be
        `subtract` (statistics cube is subtracted from the input cube) or
        `divide` (input cube is divided by the statistics cube).
    **operator_kwargs:
        Optional keyword arguments for the :class:`iris.analysis.Aggregator`
        object defined by `operator`.

    Returns
    -------
    iris.cube.Cube
        Collapsed cube.

    Raises
    ------
    iris.exceptions.CoordinateMultiDimError
        Cube has irregular or unstructured grid but supplementary variable
        `cell_area` is not available.

    """
    has_cell_measure = bool(cube.cell_measures("cell_area"))

    # Get aggregator and correct kwargs (incl. weights)
    (agg, agg_kwargs) = get_iris_aggregator(operator, **operator_kwargs)
    agg_kwargs = update_weights_kwargs(
        operator,
        agg,
        agg_kwargs,
        "cell_area",
        cube,
        try_adding_calculated_cell_area,
    )

    with ignore_iris_vague_metadata_warnings():
        result = cube.collapsed(["latitude", "longitude"], agg, **agg_kwargs)
    if normalize is not None:
        result = get_normalized_cube(cube, result, normalize)

    # Make sure input cube has not been modified
    if not has_cell_measure and cube.cell_measures("cell_area"):
        cube.remove_cell_measure("cell_area")

    return result


def extract_named_regions(cube: Cube, regions: str | Iterable[str]) -> Cube:
    """Extract a specific named region.

    The region coordinate exist in certain CMIP datasets.
    This preprocessor allows a specific named regions to be extracted.

    Parameters
    ----------
    cube:
       Input cube.
    regions:
        A region or list of regions to extract.

    Returns
    -------
    iris.cube.Cube
        Smaller cube.

    Raises
    ------
    ValueError
        regions is not list or tuple or set.
    ValueError
        region not included in cube.

    """
    # Make sure regions is a list of strings
    if isinstance(regions, str):
        regions = [regions]

    if not isinstance(regions, (list, tuple, set)):
        msg = f'Regions "{regions}" is not an acceptable format.'
        raise TypeError(
            msg,
        )

    available_regions = set(cube.coord("region").points)
    invalid_regions = set(regions) - available_regions
    if invalid_regions:
        msg = f'Region(s) "{invalid_regions}" not in cube region(s): {available_regions}'
        raise ValueError(
            msg,
        )

    constraints = iris.Constraint(region=lambda r: r in regions)
    return cube.extract(constraint=constraints)


def _crop_cube(
    cube: Cube,
    start_longitude: float,
    start_latitude: float,
    end_longitude: float,
    end_latitude: float,
    cmor_coords: bool = True,
) -> Cube:
    """Crop cubes on a regular grid."""
    lon_coord = cube.coord(axis="X")
    lat_coord = cube.coord(axis="Y")
    if lon_coord.ndim == 1 and lat_coord.ndim == 1:
        # add a padding of one cell around the cropped cube
        lon_bound = lon_coord.core_bounds()[0]
        lon_step = lon_bound[1] - lon_bound[0]
        start_longitude -= lon_step
        if not cmor_coords:
            start_longitude = max(start_longitude, -180.0)
        elif start_longitude < 0:
            start_longitude = 0
        end_longitude += lon_step
        if not cmor_coords:
            end_longitude = min(end_longitude, 180.0)
        elif end_longitude > 360:
            end_longitude = 360.0
        lat_bound = lat_coord.core_bounds()[0]
        lat_step = lat_bound[1] - lat_bound[0]
        start_latitude -= lat_step
        if start_latitude < -90:
            start_latitude = -90.0
        end_latitude += lat_step
        end_latitude = min(end_latitude, 90.0)
        cube = extract_region(
            cube,
            start_longitude,
            end_longitude,
            start_latitude,
            end_latitude,
        )
    return cube


def _select_representative_point(
    shape,
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """Get mask to select a representative point."""
    representative_point = shape.representative_point()
    points = shapely.geometry.MultiPoint(
        np.stack((np.ravel(lon), np.ravel(lat)), axis=1),
    )
    nearest_point = shapely.ops.nearest_points(points, representative_point)[0]
    nearest_lon, nearest_lat = nearest_point.coords[0]
    return (lon == nearest_lon) & (lat == nearest_lat)


def _correct_coords_from_shapefile(
    cube: Cube,
    cmor_coords: bool,
    pad_north_pole: bool,
    pad_hawaii: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Get correct lat and lon from shapefile."""
    lon = cube.coord(axis="X").points
    lat = cube.coord(axis="Y").points
    if cube.coord(axis="X").ndim < 2:
        lon, lat = np.meshgrid(lon, lat, copy=False)

    if not cmor_coords:
        # Wrap around longitude coordinate to match data
        lon = lon.copy()  # ValueError: assignment destination is read-only
        lon[lon >= 180.0] -= 360.0

        # the NE mask may not have points at x = -180 and y = +/-90
        # so we will fool it and apply the mask at (-179, -89, 89) instead
        if pad_hawaii:
            lon = np.where(lon == -180.0, lon + 1.0, lon)
    if pad_north_pole:
        lat_0 = np.where(lat == -90.0, lat + 1.0, lat)
        lat = np.where(lat_0 == 90.0, lat_0 - 1.0, lat_0)

    return lon, lat


def _process_ids(geometries, ids: list | dict | None) -> tuple:
    """Read requested IDs and ID keys."""
    # If ids is a dict, it needs to have length 1 and all geometries needs to
    # have the requested attribute key
    if isinstance(ids, dict):
        if len(ids) != 1:
            msg = (
                f"If `ids` is given as dict, it needs exactly one entry, got "
                f"{ids}"
            )
            raise ValueError(
                msg,
            )
        key = next(iter(ids.keys()))
        for geometry in geometries:
            if key not in geometry["properties"]:
                msg = (
                    f"Geometry {dict(geometry['properties'])} does not have "
                    f"requested attribute {key}"
                )
                raise ValueError(
                    msg,
                )
        id_keys: tuple[str, ...] = (key,)
        ids = ids[key]

    # Otherwise, use SHAPE_ID_KEYS to get ID
    else:
        id_keys = SHAPE_ID_KEYS

    # IDs should be strings or None
    if not ids:
        ids = None
    if ids is not None:
        ids = [str(id_) for id_ in ids]

    return (id_keys, ids)


def _get_requested_geometries(
    geometries,
    ids: list | dict | None,
    shapefile: Path,
) -> dict[str, dict]:
    """Return requested geometries."""
    (id_keys, ids) = _process_ids(geometries, ids)

    # Iterate through all geometries and select matching elements
    requested_geometries = {}
    for reading_order, geometry in enumerate(geometries):
        for key in id_keys:
            if key in geometry["properties"]:
                geometry_id = str(geometry["properties"][key])
                break

        # If none of the attributes are available in the geometry, use reading
        # order as last resort
        else:
            geometry_id = str(reading_order)

        logger.debug("Found shape '%s'", geometry_id)

        # Select geometry if its ID is requested or all IDs are requested
        # (i.e., ids=None)
        if ids is None or geometry_id in ids:
            requested_geometries[geometry_id] = geometry

    # Check if all requested IDs have been found
    if ids is not None:
        missing = set(ids) - set(requested_geometries.keys())
        if missing:
            msg = (
                f"Requested shapes {missing} not found in shapefile "
                f"{shapefile}"
            )
            raise ValueError(
                msg,
            )

    return requested_geometries


def _get_masks_from_geometries(
    geometries: dict[str, dict],
    lon: np.ndarray,
    lat: np.ndarray,
    method: str = "contains",
    decomposed: bool = False,
) -> dict[str, np.ndarray]:
    """Get cube masks from requested regions."""
    if method not in {"contains", "representative"}:
        msg = "Invalid value for `method`. Choose from 'contains', "
        raise ValueError(
            msg,
            "'representative'.",
        )

    masks = {}
    for id_, geometry in geometries.items():
        masks[id_] = _get_single_mask(lon, lat, method, geometry)

    if not decomposed and len(masks) > 1:
        return _merge_masks(masks, lat.shape)

    return masks


def _get_bounds(
    geometries: dict[str, dict],
) -> tuple[float, float, float, float]:
    """Get bounds from given geometries.

    Parameters
    ----------
    geometries: fiona.collection.Collection
        Fiona collection of shapes (geometries).

    Returns
    -------
    lat_min, lon_min, lat_max, lon_max
        Coordinates deliminating bounding box for shape ids.

    """
    all_bounds = np.vstack(
        [fiona.bounds(geom) for geom in geometries.values()],
    )
    lon_max, lat_max = all_bounds[:, 2:].max(axis=0)
    lon_min, lat_min = all_bounds[:, :2].min(axis=0)

    return lon_min, lat_min, lon_max, lat_max


def _get_single_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    method: str,
    geometry: dict,
) -> np.ndarray:
    """Get single mask from one region."""
    shape = shapely.geometry.shape(geometry["geometry"])
    if method == "contains":
        mask = shapely.vectorized.contains(shape, lon, lat)
    if method == "representative" or not mask.any():
        mask = _select_representative_point(shape, lon, lat)
    return mask


def _merge_masks(
    masks: dict[str, np.ndarray],
    shape: tuple,
) -> dict[str, np.ndarray]:
    """Merge masks into one."""
    merged_mask = np.zeros(shape, dtype=bool)
    for mask in masks.values():
        merged_mask |= mask
    return {"0": merged_mask}


def fix_coordinate_ordering(cube: Cube) -> Cube:
    """Transpose the cube dimensions.

    This is done such that the order of dimension is in standard order, i.e.:

    [time] [shape_id] [other_coordinates] latitude longitude

    where dimensions between brackets are optional.

    Parameters
    ----------
    cube:
        Input cube.

    Returns
    -------
    iris.cube.Cube
        Cube with dimensions transposed to standard order

    """
    try:
        time_dim = cube.coord_dims("time")
    except CoordinateNotFoundError:
        time_dim = ()
    try:
        shape_dim = cube.coord_dims("shape_id")
    except CoordinateNotFoundError:
        shape_dim = ()

    other = list(range(len(cube.shape)))
    for dim in [time_dim, shape_dim]:
        for i in dim:
            other.remove(i)
    other_dims = tuple(other)

    order = time_dim + shape_dim + other_dims

    cube.transpose(new_order=order)
    return cube


def _update_shapefile_path(
    shapefile: str | Path,
    session: Session | None = None,
) -> Path:
    """Update path to shapefile."""
    shapefile = str(shapefile)
    shapefile_path = Path(shapefile)

    # Try absolute path
    logger.debug("extract_shape: Looking for shapefile %s", shapefile_path)
    if shapefile_path.exists():
        return shapefile_path

    # Try path relative to auxiliary_data_dir if session is given
    if session is not None:
        shapefile_path = session["auxiliary_data_dir"] / shapefile
        logger.debug("extract_shape: Looking for shapefile %s", shapefile_path)
        if shapefile_path.exists():
            return shapefile_path

    # Try path relative to esmvalcore/preprocessor/shapefiles/
    shapefile_path = Path(__file__).parent / "shapefiles" / shapefile
    logger.debug("extract_shape: Looking for shapefile %s", shapefile_path)
    if shapefile_path.exists():
        return shapefile_path

    # As final resort, add suffix '.shp' and try path relative to
    # esmvalcore/preprocessor/shapefiles/ again
    # Note: this will find "special" shapefiles like 'ar6'
    shapefile_path = (
        Path(__file__).parent / "shapefiles" / f"{shapefile.lower()}.shp"
    )
    if shapefile_path.exists():
        return shapefile_path

    # If no valid shapefile has been found, return original input (an error
    # will be raised at a later stage)
    return Path(shapefile)


def extract_shape(
    cube: Cube,
    shapefile: str | Path,
    method: str = "contains",
    crop: bool = True,
    decomposed: bool = False,
    ids: list | dict | None = None,
) -> Cube:
    """Extract a region defined by a shapefile using masking.

    Note that this function does not work for shapes crossing the
    prime meridian or poles.

    Parameters
    ----------
    cube:
        Input cube.
    shapefile:
        A shapefile defining the region(s) to extract. Also accepts the
        following strings to load special shapefiles:

        * ``'ar6'``:  IPCC WG1 reference regions (v4) used in Assessment Report
          6 (https://doi.org/10.5281/zenodo.5176260). Should be used in
          combination with a :obj:`dict` for the argument `ids`, e.g.,
          ``ids={'Acronym': ['GIC', 'WNA']}``.
    method:
        Select all points contained by the shape or select a single
        representative point. Choose either `'contains'` or `'representative'`.
        If `'contains'` is used, but not a single grid point is contained by
        the shape, a representative point will be selected.
    crop:
        In addition to masking, crop the resulting cube using
        :func:`~esmvalcore.preprocessor.extract_region`. Data on irregular
        grids will not be cropped.
    decomposed:
        If set to `True`, the output cube will have an additional dimension
        `shape_id` describing the requested regions.
    ids:
        Shapes to be read from the shapefile. Can be given as:

        * :obj:`list`: IDs are assigned from the attributes `name`, `NAME`,
          `Name`, `id`, or `ID` (in that priority order; the first one
          available is used). If none of these attributes are available in the
          shapefile, assume that the given `ids` correspond to the reading
          order of the individual shapes, e.g., ``ids=[0, 2]`` corresponds to
          the first and third shape read from the shapefile. Note: An empty
          list is interpreted as `ids=None`.
        * :obj:`dict`: IDs (dictionary value; :obj:`list` of :obj:`str`) are
          assigned from attribute given as dictionary key (:obj:`str`). Only
          dictionaries with length 1 are supported.
          Example: ``ids={'Acronym': ['GIC', 'WNA']}`` for ``shapefile='ar6'``.
        * `None`: select all available shapes from the shapefile.

    Returns
    -------
    iris.cube.Cube
        Cube containing the extracted region.

    See Also
    --------
    extract_region: Extract a region from a cube.

    """
    shapefile = _update_shapefile_path(shapefile)
    with fiona.open(shapefile) as geometries:
        # Get parameters specific to the shapefile (NE used case e.g.
        # longitudes [-180, 180] or latitude missing or overflowing edges)
        cmor_coords = True
        pad_north_pole = False
        pad_hawaii = False
        if geometries.bounds[0] < 0:
            cmor_coords = False
        if geometries.bounds[1] > -90.0 and geometries.bounds[1] < -85.0:
            pad_north_pole = True
        if geometries.bounds[0] > -180.0 and geometries.bounds[0] < 179.0:
            pad_hawaii = True

        requested_geometries = _get_requested_geometries(
            geometries,
            ids,
            shapefile,
        )

        # Crop cube if desired
        if crop:
            lon_min, lat_min, lon_max, lat_max = _get_bounds(
                requested_geometries,
            )
            cube = _crop_cube(
                cube,
                start_longitude=lon_min,
                start_latitude=lat_min,
                end_longitude=lon_max,
                end_latitude=lat_max,
                cmor_coords=cmor_coords,
            )

        lon, lat = _correct_coords_from_shapefile(
            cube,
            cmor_coords,
            pad_north_pole,
            pad_hawaii,
        )

        masks = _get_masks_from_geometries(
            requested_geometries,
            lon,
            lat,
            method=method,
            decomposed=decomposed,
        )

    # Mask input cube based on requested regions
    result = _mask_cube(cube, masks)

    # Remove dummy scalar coordinate if final cube is not decomposed
    if not decomposed:
        result.remove_coord("shape_id")

    return result


def _mask_cube(cube: Cube, masks: dict[str, np.ndarray]) -> Cube:
    """Mask input cube."""
    cubelist = CubeList()
    for id_, mask in masks.items():
        _cube = cube.copy()
        remove_supplementary_variables(_cube)
        _cube.add_aux_coord(
            AuxCoord(id_, units="no_unit", long_name="shape_id"),
        )
        horizontal_dims = get_dims_along_axes(cube, axes=["X", "Y"])
        _cube.data = apply_mask(~mask, _cube.core_data(), horizontal_dims)
        cubelist.append(_cube)
    result = fix_coordinate_ordering(cubelist.merge_cube())
    for measure in cube.cell_measures():
        # Cell measures that are time-dependent, with 4 dimensions and an
        # original shape of (time, depth, lat, lon), need to be broadcast to
        # the cube with 5 dimensions and shape (time, shape_id, depth, lat, lon)
        if measure.ndim > 3 and result.ndim > 4:
            data = measure.core_data()
            if result.has_lazy_data():
                # Make the cell measure lazy if the result is lazy.
                cube_chunks = cube.lazy_data().chunks
                chunk_dims = cube.cell_measure_dims(measure)
                data = da.asarray(
                    data,
                    chunks=tuple(cube_chunks[i] for i in chunk_dims),
                )
                chunks = result.lazy_data().chunks
            else:
                chunks = None
            dim_map = get_dims_along_axes(result, ["T", "Z", "Y", "X"])
            data = iris.util.broadcast_to_shape(
                data,
                result.shape,
                dim_map=dim_map,
                chunks=chunks,
            )
            measure = iris.coords.CellMeasure(  # noqa: PLW2901
                data,
                standard_name=measure.standard_name,
                long_name=measure.long_name,
                units=measure.units,
                measure=measure.measure,
                var_name=measure.var_name,
                attributes=measure.attributes,
            )
        add_cell_measure(result, measure, measure.measure)
    for ancillary_variable in cube.ancillary_variables():
        add_ancillary_variable(result, ancillary_variable)
    return result
