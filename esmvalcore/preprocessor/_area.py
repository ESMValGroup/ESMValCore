"""Area operations on data cubes.

Allows for selecting data subsets using certain latitude and longitude
bounds; selecting geographical regions; constructing area averages; etc.
"""
import logging

import fiona
import iris
import numpy as np
import shapely
import shapely.ops
from dask import array as da
from iris.exceptions import CoordinateNotFoundError

from ._shared import (
    get_iris_analysis_operation,
    guess_bounds,
    operator_accept_weights,
)

logger = logging.getLogger(__name__)


# slice cube over a restricted area (box)
def extract_region(cube, start_longitude, end_longitude, start_latitude,
                   end_latitude):
    """Extract a region from a cube.

    Function that subsets a cube on a box (start_longitude, end_longitude,
    start_latitude, end_latitude)
    This function is a restriction of masked_cube_lonlat().

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
        return region_subset
    # Irregular grids
    lats = cube.coord('latitude').points
    lons = cube.coord('longitude').points
    # Convert longitudes to valid range
    if start_longitude != 360.:
        start_longitude %= 360.
    if end_longitude != 360.:
        end_longitude %= 360.

    if start_longitude <= end_longitude:
        select_lons = (lons >= start_longitude) & (lons <= end_longitude)
    else:
        select_lons = (lons >= start_longitude) | (lons <= end_longitude)

    if start_latitude <= end_latitude:
        select_lats = (lats >= start_latitude) & (lats <= end_latitude)
    else:
        select_lats = (lats >= start_latitude) | (lats <= end_latitude)

    selection = select_lats & select_lons
    selection = da.broadcast_to(selection, cube.shape)
    cube.data = da.ma.masked_where(~selection, cube.core_data())
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
    grid_areas = None
    try:
        grid_areas = cube.cell_measure('cell_area').core_data()
    except iris.exceptions.CellMeasureNotFoundError:
        logger.info(
            'Cell measure "cell_area" not found in cube %s. '
            'Check fx_file availability.', cube
        )
        logger.info('Attempting to calculate grid cell area...')

    if grid_areas is None and cube.coord('latitude').points.ndim == 2:
        coord_names = [coord.standard_name for coord in cube.coords()]
        if 'grid_latitude' in coord_names and 'grid_longitude' in coord_names:
            cube = guess_bounds(cube, ['grid_latitude', 'grid_longitude'])
            cube_tmp = cube.copy()
            cube_tmp.remove_coord('latitude')
            cube_tmp.coord('grid_latitude').rename('latitude')
            cube_tmp.remove_coord('longitude')
            cube_tmp.coord('grid_longitude').rename('longitude')
            grid_areas = iris.analysis.cartography.area_weights(cube_tmp)
            logger.info('Calculated grid area shape: %s', grid_areas.shape)
        else:
            logger.error(
                'fx_file needed to calculate grid cell area for irregular '
                'grids.')
            raise iris.exceptions.CoordinateMultiDimError(
                cube.coord('latitude'))

    coord_names = ['longitude', 'latitude']
    if grid_areas is None:
        cube = guess_bounds(cube, coord_names)
        grid_areas = iris.analysis.cartography.area_weights(cube)
        logger.info('Calculated grid area shape: %s', grid_areas.shape)

    if cube.shape != grid_areas.shape:
        raise ValueError('Cube shape ({}) doesn`t match grid area shape '
                         '({})'.format(cube.shape, grid_areas.shape))

    operation = get_iris_analysis_operation(operator)

    # TODO: implement weighted stdev, median, s var when available in iris.
    # See iris issue: https://github.com/SciTools/iris/issues/3208

    if operator_accept_weights(operator):
        return cube.collapsed(coord_names, operation, weights=grid_areas)

    # Many IRIS analysis functions do not accept weights arguments.
    return cube.collapsed(coord_names, operation)


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


def _get_masks_from_geometries(geometries, lon, lat, method='contains',
                               decomposed=False, ids=None):

    if method not in {'contains', 'representative'}:
        raise ValueError(
            "Invalid value for `method`. Choose from 'contains', ",
            "'representative'.")

    selections = dict()
    if ids:
        ids = [str(id_) for id_ in ids]
    for i, item in enumerate(geometries):
        for id_prop in ('name', 'NAME', 'Name', 'id', 'ID'):
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


def extract_shape(cube, shapefile, method='contains', crop=True,
                  decomposed=False, ids=None):
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
            cube = _crop_cube(cube,
                              *geometries.bounds,
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
        _cube.add_aux_coord(
            iris.coords.AuxCoord(id_, units='no_unit', long_name="shape_id"))
        select = da.broadcast_to(select, _cube.shape)
        _cube.data = da.ma.masked_where(~select, _cube.core_data())
        cubelist.append(_cube)
    return fix_coordinate_ordering(cubelist.merge_cube())
