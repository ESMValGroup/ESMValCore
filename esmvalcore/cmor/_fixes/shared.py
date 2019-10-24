"""Shared functions for fixes."""
import logging

import dask.array as da
import iris
from cf_units import Unit

logger = logging.getLogger(__name__)


def add_scalar_depth_coord(cube, depth=0.0):
    """Add scalar coordinate 'depth' with value of `depth`m."""
    logger.debug("Adding depth coordinate (%sm)", depth)
    depth_coord = iris.coords.AuxCoord(depth,
                                       var_name='depth',
                                       standard_name='depth',
                                       long_name='depth',
                                       units=Unit('m'),
                                       attributes={'positive': 'down'})
    cube.add_aux_coord(depth_coord, ())
    return cube


def add_scalar_height_coord(cube, height=2.0):
    """Add scalar coordinate 'height' with value of `height`m."""
    logger.debug("Adding height coordinate (%sm)", height)
    height_coord = iris.coords.AuxCoord(height,
                                        var_name='height',
                                        standard_name='height',
                                        long_name='height',
                                        units=Unit('m'),
                                        attributes={'positive': 'up'})
    cube.add_aux_coord(height_coord, ())
    return cube


def add_scalar_typeland_coord(cube, value='default'):
    """Add scalar coordinate 'typeland' with value of `value`."""
    logger.debug("Adding typeland coordinate (%s)", value)
    typeland_coord = iris.coords.AuxCoord(value,
                                          var_name='type',
                                          standard_name='area_type',
                                          long_name='Land area type',
                                          units=Unit('no unit'))
    cube.add_aux_coord(typeland_coord, ())
    return cube


def add_scalar_typesea_coord(cube, value='default'):
    """Add scalar coordinate 'typesea' with value of `value`."""
    logger.debug("Adding typesea coordinate (%s)", value)
    typesea_coord = iris.coords.AuxCoord(value,
                                         var_name='type',
                                         standard_name='area_type',
                                         long_name='Ocean area type',
                                         units=Unit('no unit'))
    cube.add_aux_coord(typesea_coord, ())
    return cube


def round_coordinates(cubes, decimals=5):
    """Round all dimensional coordinates of all cubes."""
    for cube in cubes:
        for coord in cube.coords(dim_coords=True):
            coord.points = da.round(da.asarray(coord.core_points()), decimals)
            if coord.bounds is not None:
                coord.bounds = da.round(da.asarray(coord.core_bounds()),
                                        decimals)
    return cubes
