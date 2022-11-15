"""Area operations on data cubes.

Allows for selecting data subsets using certain latitude and longitude
bounds; selecting geographical regions; constructing area averages; etc.
"""
import logging
import warnings

import fiona
import iris
import numpy as np
import shapely
import shapely.ops
from dask import array as da
from iris.exceptions import CoordinateNotFoundError

from ._ancillary_vars import (
    add_ancillary_variable,
    add_cell_measure,
    remove_fx_variables,
)
from ._shared import (
    get_iris_analysis_operation,
    guess_bounds,
    operator_accept_weights,
)

logger = logging.getLogger(__name__)

SHAPE_ID_KEYS = ('name', 'NAME', 'Name', 'id', 'ID')


def extract_region(cube, start_longitude, end_longitude, start_latitude,
                   end_latitude):
    """Extract a region from a cube.

    Function that subsets a cube on a box (start_longitude, end_longitude,
    start_latitude, end_latitude)

    Parameters
    ----------
    cube: iris.cube.Cube
        input data cube.
    start_longitude: float
        Western boundary longitude.
    end_longitude: float
        Eastern boundary longitude.
    start_latitude: float
        Southern Boundary latitude.
    end_latitude: float
        Northern Boundary Latitude.

    Returns
    -------
    iris.cube.Cube
        smaller cube.
    """
    if abs(start_latitude) > 90.:
        raise ValueError(f"Invalid start_latitude: {start_latitude}")
    if abs(end_latitude) > 90.:
        raise ValueError(f"Invalid end_latitude: {end_latitude}")
    if cube.coord('latitude').ndim == 1:
        # Iris check if any point of the cell is inside the region
        # To check only the center, ignore_bounds must be set to
        # True (default) is False
        region_subset = cube.intersection(
            longitude=(start_longitude, end_longitude),
            latitude=(start_latitude, end_latitude),
            ignore_bounds=True,
        )
        region_subset = region_subset.intersection(longitude=(0., 360.))
    else:
        region_subset = _extract_irregular_region(
            cube,
            start_longitude,
            end_longitude,
            start_latitude,
            end_latitude,
        )
    return region_subset


def _extract_irregular_region(cube, start_longitude, end_longitude,
                              start_latitude, end_latitude):
    """Extract a region from a cube on an irregular grid."""
    # Convert longitudes to valid range
    if start_longitude != 360.:
        start_longitude %= 360.
    if end_longitude != 360.:
        end_longitude %= 360.

    # Select coordinates inside the region
    lats = cube.coord('latitude').points
    lons = (cube.coord('longitude').points + 360.) % 360.
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
        raise ValueError("No data points available in selected region")
    i_min, i_max = i_range.min(), i_range.max()
    j_min, j_max = j_range.min(), j_range.max()
    i_slice, j_slice = slice(i_min, i_max + 1), slice(j_min, j_max + 1)
    cube = cube[..., i_slice, j_slice]
    selection = selection[i_slice, j_slice]
    # Mask remaining coordinates outside region
    mask = da.broadcast_to(~selection, cube.shape)
    cube.data = da.ma.masked_where(mask, cube.core_data())
    return cube


def zonal_statistics(cube, operator):
    """Compute zonal statistics.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    operator: str, optional
        Select operator to apply.
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min',
        'max', 'rms'.

    Returns
    -------
    iris.cube.Cube
        Zonal statistics cube.

    Raises
    ------
    ValueError
        Error raised if computation on irregular grids is attempted.
        Zonal statistics not yet implemented for irregular grids.
    """
    if cube.coord('longitude').points.ndim < 2:
        operation = get_iris_analysis_operation(operator)
        cube = cube.collapsed('longitude', operation)
        cube.data = cube.core_data().astype(np.float32, casting='same_kind')
        return cube
    msg = ("Zonal statistics on irregular grids not yet implemnted")
    raise ValueError(msg)


def meridional_statistics(cube, operator):
    """Compute meridional statistics.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    operator: str, optional
        Select operator to apply.
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min',
        'max', 'rms'.

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
    if cube.coord('latitude').points.ndim < 2:
        operation = get_iris_analysis_operation(operator)
        cube = cube.collapsed('latitude', operation)
        cube.data = cube.core_data().astype(np.float32, casting='same_kind')
        return cube
    msg = ("Meridional statistics on irregular grids not yet implemented")
    raise ValueError(msg)


def compute_area_weights(cube):
    """Compute area weights."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.filterwarnings(
            'always',
            message="Using DEFAULT_SPHERICAL_EARTH_RADIUS.",
            category=UserWarning,
            module='iris.analysis.cartography',
        )
        weights = iris.analysis.cartography.area_weights(cube)
        for warning in caught_warnings:
            logger.debug(
                "%s while computing area weights of the following cube:\n%s",
                warning.message, cube)
    return weights


def area_statistics(cube, operator):
    """Apply a statistical operator in the horizontal direction.

    The average in the horizontal direction. We assume that the
    horizontal directions are ['longitude', 'latutude'].

    This function can be used to apply
    several different operations in the horizontal plane: mean, standard
    deviation, median variance, minimum and maximum. These options are
    specified using the `operator` argument and the following key word
    arguments:

    +------------+--------------------------------------------------+
    | `mean`     | Area weighted mean.                              |
    +------------+--------------------------------------------------+
    | `median`   | Median (not area weighted)                       |
    +------------+--------------------------------------------------+
    | `std_dev`  | Standard Deviation (not area weighted)           |
    +------------+--------------------------------------------------+
    | `sum`      | Area weighted sum.                               |
    +------------+--------------------------------------------------+
    | `variance` | Variance (not area weighted)                     |
    +------------+--------------------------------------------------+
    | `min`:     | Minimum value                                    |
    +------------+--------------------------------------------------+
    | `max`      | Maximum value                                    |
    +------------+--------------------------------------------------+
    | `rms`      | Area weighted root mean square.                  |
    +------------+--------------------------------------------------+

    Parameters
    ----------
        cube: iris.cube.Cube
            Input cube.
        operator: str
            The operation, options: mean, median, min, max, std_dev, sum,
            variance, rms.

    Returns
    -------
    iris.cube.Cube
        collapsed cube.

    Raises
    ------
    iris.exceptions.CoordinateMultiDimError
        Exception for latitude axis with dim > 2.
    ValueError
        if input data cube has different shape than grid area weights
    """
    original_dtype = cube.dtype
    grid_areas = None
    try:
        grid_areas = cube.cell_measure('cell_area').core_data()
    except iris.exceptions.CellMeasureNotFoundError:
        logger.debug(
            'Cell measure "cell_area" not found in cube %s. '
            'Check fx_file availability.', cube.summary(shorten=True))
        logger.debug('Attempting to calculate grid cell area...')
    else:
        grid_areas = da.broadcast_to(grid_areas, cube.shape)

    if grid_areas is None and cube.coord('latitude').points.ndim == 2:
        coord_names = [coord.standard_name for coord in cube.coords()]
        if 'grid_latitude' in coord_names and 'grid_longitude' in coord_names:
            cube = guess_bounds(cube, ['grid_latitude', 'grid_longitude'])
            cube_tmp = cube.copy()
            cube_tmp.remove_coord('latitude')
            cube_tmp.coord('grid_latitude').rename('latitude')
            cube_tmp.remove_coord('longitude')
            cube_tmp.coord('grid_longitude').rename('longitude')
            grid_areas = compute_area_weights(cube_tmp)
            logger.debug('Calculated grid area shape: %s', grid_areas.shape)
        else:
            logger.error(
                'fx_file needed to calculate grid cell area for irregular '
                'grids.')
            raise iris.exceptions.CoordinateMultiDimError(
                cube.coord('latitude'))

    coord_names = ['longitude', 'latitude']
    if grid_areas is None:
        cube = guess_bounds(cube, coord_names)
        grid_areas = compute_area_weights(cube)
        logger.debug('Calculated grid area shape: %s', grid_areas.shape)

    if cube.shape != grid_areas.shape:
        raise ValueError('Cube shape ({}) doesn`t match grid area shape '
                         '({})'.format(cube.shape, grid_areas.shape))

    operation = get_iris_analysis_operation(operator)

    # TODO: implement weighted stdev, median, s var when available in iris.
    # See iris issue: https://github.com/SciTools/iris/issues/3208

    if operator_accept_weights(operator):
        result = cube.collapsed(coord_names, operation, weights=grid_areas)
    else:
        # Many IRIS analysis functions do not accept weights arguments.
        result = cube.collapsed(coord_names, operation)

    new_dtype = result.dtype
    if original_dtype != new_dtype:
        logger.debug(
            "area_statistics changed dtype from "
            "%s to %s, changing back", original_dtype, new_dtype)
        result.data = result.core_data().astype(original_dtype)
    return result


def extract_named_regions(cube, regions):
    """Extract a specific named region.

    The region coordinate exist in certain CMIP datasets.
    This preprocessor allows a specific named regions to be extracted.

    Parameters
    ----------
    cube: iris.cube.Cube
       input cube.
    regions: str, list
        A region or list of regions to extract.

    Returns
    -------
    iris.cube.Cube
        collapsed cube.

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
        raise TypeError(
            'Regions "{}" is not an acceptable format.'.format(regions))

    available_regions = set(cube.coord('region').points)
    invalid_regions = set(regions) - available_regions
    if invalid_regions:
        raise ValueError('Region(s) "{}" not in cube region(s): {}'.format(
            invalid_regions, available_regions))

    constraints = iris.Constraint(region=lambda r: r in regions)
    cube = cube.extract(constraint=constraints)
    return cube


def _crop_cube(cube,
               start_longitude,
               start_latitude,
               end_longitude,
               end_latitude,
               cmor_coords=True):
    """Crop cubes on a cartesian grid."""
    lon_coord = cube.coord(axis='X')
    lat_coord = cube.coord(axis='Y')
    if lon_coord.ndim == 1 and lat_coord.ndim == 1:
        # add a padding of one cell around the cropped cube
        lon_bound = lon_coord.core_bounds()[0]
        lon_step = lon_bound[1] - lon_bound[0]
        start_longitude -= lon_step
        if not cmor_coords:
            if start_longitude < -180.:
                start_longitude = -180.
        else:
            if start_longitude < 0:
                start_longitude = 0
        end_longitude += lon_step
        if not cmor_coords:
            if end_longitude > 180.:
                end_longitude = 180.
        else:
            if end_longitude > 360:
                end_longitude = 360.
        lat_bound = lat_coord.core_bounds()[0]
        lat_step = lat_bound[1] - lat_bound[0]
        start_latitude -= lat_step
        if start_latitude < -90:
            start_latitude = -90.
        end_latitude += lat_step
        if end_latitude > 90.:
            end_latitude = 90.
        cube = extract_region(cube, start_longitude, end_longitude,
                              start_latitude, end_latitude)
    return cube


def _select_representative_point(shape, lon, lat):
    """Select a representative point for `shape` from `lon` and `lat`."""
    representative_point = shape.representative_point()
    points = shapely.geometry.MultiPoint(
        np.stack((np.ravel(lon), np.ravel(lat)), axis=1))
    nearest_point = shapely.ops.nearest_points(points, representative_point)[0]
    nearest_lon, nearest_lat = nearest_point.coords[0]
    select = (lon == nearest_lon) & (lat == nearest_lat)
    return select


def _correct_coords_from_shapefile(cube, cmor_coords, pad_north_pole,
                                   pad_hawaii):
    """Get correct lat and lon from shapefile."""
    lon = cube.coord(axis='X').points
    lat = cube.coord(axis='Y').points
    if cube.coord(axis='X').ndim < 2:
        lon, lat = np.meshgrid(lon, lat, copy=False)

    if not cmor_coords:
        # Wrap around longitude coordinate to match data
        lon = lon.copy()  # ValueError: assignment destination is read-only
        lon[lon >= 180.] -= 360.

        # the NE mask may not have points at x = -180 and y = +/-90
        # so we will fool it and apply the mask at (-179, -89, 89) instead
        if pad_hawaii:
            lon = np.where(lon == -180., lon + 1., lon)
    if pad_north_pole:
        lat_0 = np.where(lat == -90., lat + 1., lat)
        lat = np.where(lat_0 == 90., lat_0 - 1., lat_0)

    return lon, lat


def _get_masks_from_geometries(geometries,
                               lon,
                               lat,
                               method='contains',
                               decomposed=False,
                               ids=None):

    if method not in {'contains', 'representative'}:
        raise ValueError(
            "Invalid value for `method`. Choose from 'contains', ",
            "'representative'.")

    selections = dict()
    if ids:
        ids = [str(id_) for id_ in ids]
    for i, item in enumerate(geometries):
        for id_prop in SHAPE_ID_KEYS:
            if id_prop in item['properties']:
                id_ = str(item['properties'][id_prop])
                break
        else:
            id_ = str(i)
        logger.debug('Shape "%s" found', id_)
        if ids and id_ not in ids:
            continue
        selections[id_] = _get_shape(lon, lat, method, item)

    if ids:
        missing = set(ids) - set(selections.keys())
        if missing:
            raise ValueError(f'Shapes {" ".join(missing)!r} not found')

    if not decomposed and len(selections) > 1:
        return _merge_shapes(selections, lat.shape)

    return selections


def _geometry_matches_ids(geometry: dict, ids: list):
    """Returns True if `geometry` matches one of the `ids`."""
    props = geometry['properties']

    geom_id = [props.get(key, None) for key in SHAPE_ID_KEYS]
    geom_id = [key for key in geom_id if key is not None]

    if not geom_id:
        raise KeyError(f'{props} dict has no `name` or `id` key')

    geom_id = geom_id[0]

    return geom_id in ids


def _get_bounds(geometries, ids=None):
    """Get bounds from the subset of geometries defined by `ids`.

    Parameters
    ----------
    geometries : fiona.Collection
        Fiona collection of shapes (geometries).
    ids : tuple of str, optional
        List of ids to select from geometry collection. If None,
        return global bounds (``geometries.bounds``)

    Returns
    -------
    lat_min, lon_min, lat_max, lon_max
        Returns coordinates deliminating bounding box for shape ids.
    """
    if not ids:
        return geometries.bounds

    subset = [geom for geom in geometries if _geometry_matches_ids(geom, ids)]

    all_bounds = np.vstack([fiona.bounds(geom) for geom in subset])
    lon_max, lat_max = all_bounds[:, 2:].max(axis=0)
    lon_min, lat_min = all_bounds[:, :2].min(axis=0)

    return lon_min, lat_min, lon_max, lat_max


def _get_shape(lon, lat, method, item):
    shape = shapely.geometry.shape(item['geometry'])
    if method == 'contains':
        select = shapely.vectorized.contains(shape, lon, lat)
    if method == 'representative' or not select.any():
        select = _select_representative_point(shape, lon, lat)
    return select


def _merge_shapes(selections, shape):
    selection = np.zeros(shape, dtype=bool)
    for select in selections.values():
        selection |= select
    return {0: selection}


def fix_coordinate_ordering(cube):
    """Transpose the dimensions.

    This is done such that the order of dimension is
    in standard order, ie:

    [time] [shape_id] [other_coordinates] latitude longitude

    where dimensions between brackets are optional.

    Parameters
    ----------
    cube: iris.cube.Cube
       input cube.

    Returns
    -------
    iris.cube.Cube
        Cube with dimensions transposed to standard order
    """
    try:
        time_dim = cube.coord_dims('time')
    except CoordinateNotFoundError:
        time_dim = ()
    try:
        shape_dim = cube.coord_dims('shape_id')
    except CoordinateNotFoundError:
        shape_dim = ()

    other = list(range(len(cube.shape)))
    for dim in [time_dim, shape_dim]:
        for i in dim:
            other.remove(i)
    other = tuple(other)

    order = time_dim + shape_dim + other

    cube.transpose(new_order=order)
    return cube


def extract_shape(cube,
                  shapefile,
                  method='contains',
                  crop=True,
                  decomposed=False,
                  ids=None):
    """Extract a region defined by a shapefile.

    Note that this function does not work for shapes crossing the
    prime meridian or poles.

    Parameters
    ----------
    cube: iris.cube.Cube
       input cube.
    shapefile: str
        A shapefile defining the region(s) to extract.
    method: str, optional
        Select all points contained by the shape or select a single
        representative point. Choose either 'contains' or 'representative'.
        If 'contains' is used, but not a single grid point is contained by the
        shape, a representative point will selected.
    crop: bool, optional
        Crop the resulting cube using `extract_region()`. Note that data on
        irregular grids will not be cropped.
    decomposed: bool, optional
        Whether or not to retain the sub shapes of the shapefile in the output.
        If this is set to True, the output cube has a dimension for the sub
        shapes.
    ids: list(str), optional
        List of shapes to be read from the file. The ids are assigned from
        the attributes 'name' or 'id' (in that priority order) if present in
        the file or correspond to the reading order if not.

    Returns
    -------
    iris.cube.Cube
        Cube containing the extracted region.

    See Also
    --------
    extract_region : Extract a region from a cube.
    """
    with fiona.open(shapefile) as geometries:

        # get parameters specific to the shapefile (NE used case
        # eg longitudes [-180, 180] or latitude missing
        # or overflowing edges)
        cmor_coords = True
        pad_north_pole = False
        pad_hawaii = False
        if geometries.bounds[0] < 0:
            cmor_coords = False
        if geometries.bounds[1] > -90. and geometries.bounds[1] < -85.:
            pad_north_pole = True
        if geometries.bounds[0] > -180. and geometries.bounds[0] < 179.:
            pad_hawaii = True

        if crop:
            lon_min, lat_min, lon_max, lat_max = _get_bounds(
                geometries=geometries,
                ids=ids,
            )
            cube = _crop_cube(cube,
                              start_longitude=lon_min,
                              start_latitude=lat_min,
                              end_longitude=lon_max,
                              end_latitude=lat_max,
                              cmor_coords=cmor_coords)

        lon, lat = _correct_coords_from_shapefile(cube, cmor_coords,
                                                  pad_north_pole, pad_hawaii)

        selections = _get_masks_from_geometries(geometries,
                                                lon,
                                                lat,
                                                method=method,
                                                decomposed=decomposed,
                                                ids=ids)

    return _mask_cube(cube, selections)


def _mask_cube(cube, selections):
    cubelist = iris.cube.CubeList()
    for id_, select in selections.items():
        _cube = cube.copy()
        remove_fx_variables(_cube)
        _cube.add_aux_coord(
            iris.coords.AuxCoord(id_, units='no_unit', long_name="shape_id"))
        select = da.broadcast_to(select, _cube.shape)
        _cube.data = da.ma.masked_where(~select, _cube.core_data())
        cubelist.append(_cube)
    result = fix_coordinate_ordering(cubelist.merge_cube())
    if cube.cell_measures():
        for measure in cube.cell_measures():
            add_cell_measure(result, measure, measure.measure)
    if cube.ancillary_variables():
        for ancillary_variable in cube.ancillary_variables():
            add_ancillary_variable(result, ancillary_variable)
    return result
