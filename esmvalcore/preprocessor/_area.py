"""
Area operations on data cubes.

Allows for selecting data subsets using certain latitude and longitude bounds;
selecting geographical regions; constructing area averages; etc.
"""
import logging

import iris
from dask import array as da
import numpy as np
from ._shared import get_iris_analysis_operation, weighted_operators

logger = logging.getLogger(__name__)


# guess bounds tool
def _guess_bounds(cube, coords):
    """Guess bounds of a cube, or not."""
    # check for bounds just in case
    for coord in coords:
        if not cube.coord(coord).has_bounds():
            cube.coord(coord).guess_bounds()
    return cube


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
    # Converts Negative longitudes to 0 -> 360. standard
    start_longitude = float(start_longitude)
    end_longitude = float(end_longitude)
    start_latitude = float(start_latitude)
    end_latitude = float(end_latitude)

    if cube.coord('latitude').ndim == 1:
        region_subset = cube.intersection(
            longitude=(start_longitude, end_longitude),
            latitude=(start_latitude, end_latitude))
        region_subset = region_subset.intersection(longitude=(0., 360.))
        return region_subset

    # irregular grids
    import numpy as np
    lats = cube.coord('latitude').points
    lons = cube.coord('longitude').points
    mask = np.ma.array(cube.data).mask
    mask += np.ma.masked_where(lats < start_latitude, lats).mask
    mask += np.ma.masked_where(lats > end_latitude, lats).mask
    mask += np.ma.masked_where(lons > start_longitude, lons).mask
    mask += np.ma.masked_where(lons > end_longitude, lons).mask
    cube.data = np.ma.masked_where(mask, cube.data)
    return cube

    # irregular grids
    lats = cube.coord('latitude').points
    lons = cube.coord('longitude').points
    select_lats = start_latitude < lats < end_latitude
    select_lons = start_longitude < lons < end_longitude
    selection = select_lats & select_lons
    data = da.ma.masked_where(~selection, cube.core_data())
    return cube.copy(data)


def zonal_statistics(cube, operator, fx_files=None):
    """
    Get zonal statistics.

    Function that applies a statistical operation along the zonal
    dimension. The resulting cube will typicallly have dimensions:
    (time, z, latitude'). This function can be used to apply several different
    operations in the horizonal plane: mean, standard deviation, median,
    variance, minimum and maximum. These options are specified using the
    `operator` argument and the following key word arguments are accepted:

    +------------+--------------------------------------------------+
    | `mean`     | Area weighted mean.                              |
    +------------+--------------------------------------------------+
    | `median`   | Median (not area weighted)                       |
    +------------+--------------------------------------------------+
    | `std_dev`  | Standard Deviation (not area weighted)           |
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
        input cube.
    operator: str
        Type of analysis to use, from iris.analysis.
    fx_files: dict
        dictionary of field:filename for the fx_files
    Returns
    -------
    iris.cube.Cube
    """
    return zonal_meridional_statistics(cube, operator, 'longitude',
                                       fx_files=fx_files)


def meridional_statistics(cube, operator, fx_files=None):
    """
    Get meridional statistics.

    Function that applies and operator inb the meridional direction.
    The operation is provided by the operator variable (string).
    - 'mean' -> MEAN
    - 'median' -> MEDIAN
    - 'std_dev' -> STD_DEV
    - 'variance' -> VARIANCE
    - 'min' -> MIN
    - 'max' -> MAX

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
      | `variance` | Variance (not area weighted)                     |
      +------------+--------------------------------------------------+
      | `min`:     | Minimum value                                    |
      +------------+--------------------------------------------------+
      | `max`      | Maximum value                                    |
      +------------+--------------------------------------------------+

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    operator: str
        Type of analysis to use, from iris.analysis.

    Returns
    -------
    iris.cube.Cube
    """
    return zonal_meridional_statistics(cube, operator, 'latitude',
                                       fx_files=fx_files)


def zonal_meridional_statistics(cube, operator, coord, fx_files=None):
    """
    Function to calculate either the zonal or meriodinal statistics.

    The zonal and meriodinal statistics calculations are very similar, so they
    are writen as wrappers to this function which does the work. The only
    difference between them is which dimension to apply to iris operator:
    Longitude (Zonal) or Latitude (Meridional). This choice is provided by the
    coord variable.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    operator: str
        Type of analysis to use, from iris.analysis.
    coord: str
        Longitude (Zonal) or Latitude (Meridional)
    Returns
    -------
    iris.cube.Cube
    """

    # Load iris opertion.
    operation = get_iris_analysis_operation(operator)

    # Load fx files
    grid_volume_found = False
    grid_volume = None
    if fx_files:
        for key, fx_file in fx_files.items():
            if fx_file is None:
                continue
            logger.info('Attempting to load %s from file: %s', key, fx_file)
            fx_cube = iris.load_cube(fx_file)

            grid_volume = fx_cube.data
            grid_volume_found = True

    if grid_volume_found and operator.lower() in weighted_operators:
        return cube.collapsed(coord, operation, weights=grid_volume)

    return cube.collapsed(coord, operation)


def load_fx_files(fx_files):
    """
    Load the fx_files data.

    Assumes that there is only one fx_file requested.

    Parameters
    ----------
        fx_files: dict
            dictionary of field:filename for the fx_files

    Returns
    -------
    np.array
        The fx_file data.
    """
    for key, fx_file in fx_files.items():
        if fx_file is None:
            continue
        logger.info('Attempting to load %s from file: %s', key, fx_file)
        fx_cube = iris.load_cube(fx_file)
        fx_data = fx_cube.core_data()
        return fx_data
    return None


def tile_grid_areas(cube, grid_areas):
    """
    Tile the grid area data to match the dataset cube.

    Parameters
    ----------
    cube: iris.cube.Cube
        Input cube.
    grid_areas: iris.cube.Cube
        fx files area.

    Returns
    -------
    iris.cube.Cube
        Freshly tiled grid areas cube.
    """
    if grid_areas.shape != cube.shape[-2:]:
        raise ValueError('Fx area {} and dataset {} shapes do not match.'
                         ''.format(grid_areas.shape, cube.shape))

    if cube.ndim == grid_areas.ndim:
        return grid_areas

    # # np.tile works as iris can't accept dask arrays for weights.
    # if cube.ndim == 4 and grid_areas.ndim == 2:
    #     grid_areas = np.tile(grid_areas,
    #                          [cube.shape[0], cube.shape[1], 1, 1])
    # elif cube.ndim == 4 and grid_areas.ndim == 3:
    #     grid_areas = np.tile(grid_areas, [cube.shape[0], 1, 1, 1])
    # elif cube.ndim == 3 and grid_areas.ndim == 2:
    #     grid_areas = np.tile(grid_areas, [cube.shape[0], 1, 1])
    # else:
    #     raise ValueError('Grid and dataset number of dimensions not '
    #                      'recognised: {} and {}.'
    #                      ''.format(cube.ndim, grid_areas.ndim))

    # Using dash arary stack instead of np.tile
    if cube.ndim == grid_areas.ndim:
        return grid_areas
    elif cube.ndim == 4 and grid_areas.ndim == 2:
        for shape in [1, 0]:
            ga = [grid_areas for itr in range(cube.shape[shape])]
            grid_areas = da.stack(ga, axis=0)
    elif cube.ndim == 4 and grid_areas.ndim == 3:
        ga = [grid_areas for itr in range(cube.shape[0])]
        grid_areas = da.stack(ga, axis=0)
    elif cube.ndim == 3 and grid_areas.ndim == 2:
        ga = [grid_areas for itr in range(cube.shape[0])]
        grid_areas = da.stack(ga, axis=0)
    else:
        raise ValueError('Grid and dataset number of dimensions not '
                         'recognised: {} and {}.'
                         ''.format(cube.ndim, grid_areas.ndim))
    return grid_areas


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
            The operation, options: mean, median, min, max, std_dev, variance
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
    coord_names = ['longitude', 'latitude']
    operation = get_iris_analysis_operation(operator)

    # Many IRIS analysis functions do not accept weights arguments
    # and some operators do not require fx files.
    if operator not in weighted_operators:
        return cube.collapsed(coord_names, operation)


    if fx_files == None:
        raise ValueError("Operator {} requires fx_files. ".format(operator))

    # All weighted operators need FX files:
    if not fx_files and cube.coord('latitude').points.ndim == 2 and \
        operator in weighted_operators:
        logger.error(
            'fx_file needed to calculate grid cell area for irregular grids.')
        raise ValueError('Fx_file needed to calculate weighted area'
                         'statitics.')

    grid_areas = load_fx_files(fx_files)
    grid_areas = tile_grid_areas(cube, grid_areas)
    if cube.shape != grid_areas.shape:
        raise ValueError('Cube shape ({}) doesn`t match grid area shape '
                         '({})'.format(cube.shape, grid_areas.shape))

    #if grid_areas is None or not grid_areas.any():
#        cube = _guess_bounds(cube, coord_names)
#        grid_areas = iris.analysis.cartography.area_weights(cube)
#        logger.info('Calculated grid area shape: %s', grid_areas.shape)

    # TODO: implement weighted stdev, median, s var when available in iris.
    # See iris issue: https://github.com/SciTools/iris/issues/3208
    return cube.collapsed(coord_names,
                          operation,
                          weights=np.array(grid_areas))



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
