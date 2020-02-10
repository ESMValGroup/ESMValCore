"""
Area operations on data cubes.

Allows for selecting data subsets using certain latitude and longitude bounds;
selecting geographical regions; constructing area averages; etc.
"""
import logging

import fiona
import iris
import numpy as np
import shapely
import shapely.ops
from dask import array as da
from iris.exceptions import CoordinateNotFoundError

from ._shared import (get_iris_analysis_operation, guess_bounds,
                      operator_accept_weights)

logger = logging.getLogger(__name__)


# slice cube over a restricted area (box)
def extract_region(cube, start_longitude, end_longitude, start_latitude,
                   end_latitude):
    """
    Extract a region from a cube.

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
    """
    Compute zonal statistics.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    operator: str, optional
        Select operator to apply.
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min', 'max'.

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
        return cube.collapsed('longitude', operation)
    else:
        msg = (f"Zonal statistics on irregular grids not yet implemnted")
        raise ValueError(msg)


def meridional_statistics(cube, operator):
    """
    Compute meridional statistics.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    operator: str, optional
        Select operator to apply.
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min', 'max'.

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
        return cube.collapsed('latitude', operation)
    else:
        msg = (f"Meridional statistics on irregular grids not yet implemnted")
        raise ValueError(msg)


def tile_grid_areas(cube, fx_files):
    """
    Tile the grid area data to match the dataset cube.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    fx_files: dict
        dictionary of field:filename for the fx_files

    Returns
    -------
    iris.cube.Cube
        Freshly tiled grid areas cube.
    """
    grid_areas = None
    if fx_files:
        for key, fx_file in fx_files.items():
            if fx_file is None:
                continue
            logger.info('Attempting to load %s from file: %s', key, fx_file)
            fx_cube = iris.load_cube(fx_file)

            grid_areas = fx_cube.core_data()
            if cube.ndim == 4 and grid_areas.ndim == 2:
                grid_areas = da.tile(grid_areas,
                                     [cube.shape[0], cube.shape[1], 1, 1])
            elif cube.ndim == 4 and grid_areas.ndim == 3:
                grid_areas = da.tile(grid_areas, [cube.shape[0], 1, 1, 1])
            elif cube.ndim == 3 and grid_areas.ndim == 2:
                grid_areas = da.tile(grid_areas, [cube.shape[0], 1, 1])
            else:
                raise ValueError('Grid and dataset number of dimensions not '
                                 'recognised: {} and {}.'
                                 ''.format(cube.ndim, grid_areas.ndim))
    return grid_areas


# get the area average
def area_statistics(cube, operator, fx_files=None):
    """
    Apply a statistical operator in the horizontal direction.

    The average in the horizontal direction. We assume that the
    horizontal directions are ['longitude', 'latutude'].

    This function can be used to apply
    several different operations in the horizonal plane: mean, standard
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

    Parameters
    ----------
        cube: iris.cube.Cube
            Input cube.
        operator: str
            The operation, options: mean, median, min, max, std_dev, sum,
            variance
        fx_files: dict
            dictionary of field:filename for the fx_files

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
    grid_areas = tile_grid_areas(cube, fx_files)

    if not fx_files and cube.coord('latitude').points.ndim == 2:
        logger.error(
            'fx_file needed to calculate grid cell area for irregular grids.')
        raise iris.exceptions.CoordinateMultiDimError(cube.coord('latitude'))

    coord_names = ['longitude', 'latitude']
    if grid_areas is None or not grid_areas.any():
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
    """
    Extract a specific named region.

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


def _crop_cube(cube, start_longitude, start_latitude, end_longitude,
               end_latitude):
    """Crop cubes on a cartesian grid."""
    lon_coord = cube.coord(axis='X')
    lat_coord = cube.coord(axis='Y')
    if lon_coord.ndim == 1 and lat_coord.ndim == 1:
        # add a padding of one cell around the cropped cube
        lon_bound = lon_coord.core_bounds()[0]
        lon_step = lon_bound[1] - lon_bound[0]
        start_longitude -= lon_step
        end_longitude += lon_step
        lat_bound = lat_coord.core_bounds()[0]
        lat_step = lat_bound[1] - lat_bound[0]
        start_latitude -= lat_step
        end_latitude += lat_step
        cube = extract_region(cube, start_longitude, end_longitude,
                              start_latitude, end_latitude)

    return cube


def _select_representative_point(shape, lon, lat):
    """Select a representative point for `shape` from `lon` and `lat`."""
    representative_point = shape.representative_point()
    points = shapely.geometry.MultiPoint(np.stack((lon.flat, lat.flat),
                                                  axis=1))
    nearest_point = shapely.ops.nearest_points(points, representative_point)[0]
    nearest_lon, nearest_lat = nearest_point.coords[0]
    select = (lon == nearest_lon) & (lat == nearest_lat)
    return select


def _get_masks_from_geometries(geometries,
                               lon,
                               lat,
                               method='contains',
                               decomposed=False):

    if method not in {'contains', 'representative'}:
        raise ValueError(
            "Invalid value for `method`. Choose from 'contains', ",
            "'representative'.")

    selections = dict()

    for i, item in enumerate(geometries):
        shape = shapely.geometry.shape(item['geometry'])
        if method == 'contains':
            select = shapely.vectorized.contains(shape, lon, lat)
        if method == 'representative' or not select.any():
            select = _select_representative_point(shape, lon, lat)
        if 'ID' in item['properties']:
            id_ = int(item['properties']['ID'])
        elif 'id' in item['properties']:
            id_ = int(item['properties']['id'])
        else:
            id_ = i

        selections[id_] = select

    if not decomposed and len(selections) > 1:
        selection = np.zeros(lat.shape, dtype=bool)
        for select in selections.values():
            selection |= select

        selections = {0: selection}

    return selections


def fix_coordinate_ordering(cube):
    """ transpose the dimensions such that the order of dimension is
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
                  decomposed=False):
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

    Returns
    -------
    iris.cube.Cube
        Cube containing the extracted region.

    See Also
    --------
    extract_region : Extract a region from a cube.

    """

    with fiona.open(shapefile) as geometries:

        if crop:
            cube = _crop_cube(cube, *geometries.bounds)

        lon = cube.coord(axis='X').points
        lat = cube.coord(axis='Y').points
        if cube.coord(axis='X').ndim == 1 and cube.coord(axis='Y').ndim == 1:
            lon, lat = np.meshgrid(lon.flat, lat.flat, copy=False)

        selections = _get_masks_from_geometries(geometries,
                                                lon,
                                                lat,
                                                method=method,
                                                decomposed=decomposed)

    cubelist = iris.cube.CubeList()

    for id_, select in selections.items():
        _cube = cube.copy()
        _cube.add_aux_coord(
            iris.coords.AuxCoord(id_, units='no_unit', long_name="shape_id"))

        select = da.broadcast_to(select, _cube.shape)
        _cube.data = da.ma.masked_where(~select, _cube.core_data())
        cubelist.append(_cube)

    cube = cubelist.merge_cube()

    return fix_coordinate_ordering(cube)
