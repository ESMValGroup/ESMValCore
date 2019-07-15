"""CMORizer for certain projects."""
import datetime
import importlib.util
import logging
import os
import re

import iris
import numpy as np
from cf_units import Unit
from dask import array as da

from esmvalcore import __version__ as version
from esmvalcore.cmor.table import CMOR_TABLES

logger = logging.getLogger(__name__)

INVALID_UNITS = {
    'kg/m**2s': 'kg m-2 s-1',
}


def cmorize(cubes, variable, var_mapping, cmorizer):
    """Use project-specific CMORizer and CMORizer data."""
    cmorizer = os.path.expanduser(cmorizer)
    if not os.path.isabs(cmorizer):
        root = os.path.dirname(os.path.realpath(__file__))
        cmorizer = os.path.join(root, cmorizer)
    try:
        spec = importlib.util.spec_from_file_location('cmorizer', cmorizer)
        cmorizer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cmorizer_module)
    except Exception:
        logger.error(
            "CMORizer '%s' given in 'config-developer.yml' for project '%s' "
            "is not a valid CMORizer, make sure that it exists and that it "
            "contains a function called 'cmorize'", cmorizer,
            variable['project'])
        raise
    if not hasattr(cmorizer_module, 'cmorize'):
        raise ValueError(
            f"CMORizer {cmorizer} does not contain a function called "
            f"'cmorize'")
    cubes = iris.cube.CubeList(cubes)
    logger.debug("Successfully loaded CMORizer %s", cmorizer)
    return cmorizer_module.cmorize(cubes, variable, var_mapping)


def add_scalar_height_coord(cube, height=2.0):
    """Add scalar coordinate 'height' with value of `height`m."""
    logger.info("Adding height coordinate (%sm)", height)
    height_coord = iris.coords.AuxCoord(height,
                                        var_name='height',
                                        standard_name='height',
                                        long_name='height',
                                        units=Unit('m'),
                                        attributes={'positive': 'up'})
    cube.add_aux_coord(height_coord, ())


def convert_timeunits(cube, start_year):
    """Convert time axis from malformed Year 0."""
    if cube.coord('time').units == 'months since 0000-01-01 00:00:00':
        real_unit = 'months since {}-01-01 00:00:00'.format(str(start_year))
    elif cube.coord('time').units == 'days since 0000-01-01 00:00:00':
        real_unit = 'days since {}-01-01 00:00:00'.format(str(start_year))
    elif cube.coord('time').units == 'days since 1950-1-1':
        real_unit = 'days since 1950-1-1 00:00:00'
    else:
        real_unit = cube.coord('time').units
    cube.coord('time').units = real_unit
    return cube


def fix_coords(cube):
    """Fix the time units and values to CMOR standards."""
    # first fix any completely missing coord var names
    _fix_dim_coordnames(cube)
    # fix individual coords
    for cube_coord in cube.coords():
        # fix time
        if cube_coord.var_name == 'time':
            logger.info("Fixing time")
            cube.coord('time').convert_units(
                Unit('days since 1950-1-1 00:00:00', calendar='gregorian'))
            _fix_bounds(cube, cube.coord('time'))

        # fix longitude
        if cube_coord.var_name == 'lon':
            logger.info("Fixing longitude")
            if cube.coord('longitude').points[0] < 0. and \
                    cube.coord('longitude').points[-1] < 181.:
                cube.coord('longitude').points = \
                    cube.coord('longitude').points + 180.
                _fix_bounds(cube, cube.coord('longitude'))
                cube.attributes['geospatial_lon_min'] = 0.
                cube.attributes['geospatial_lon_max'] = 360.
                nlon = len(cube.coord('longitude').points)
                _roll_cube_data(cube, int(nlon / 2), -1)
            cube_coord.coord_system = None

        # fix latitude
        if cube_coord.var_name == 'lat':
            logger.info("Fixing latitude")
            _fix_bounds(cube, cube.coord('latitude'))
            cube_coord.coord_system = None

        # fix depth
        if cube_coord.var_name == 'lev':
            logger.info("Fixing depth")
            _fix_bounds(cube, cube.coord('depth'))

        # fix air_pressure
        if cube_coord.var_name == 'air_pressure':
            logger.info("Fixing air pressure")
            _fix_bounds(cube, cube.coord('air_pressure'))

    return cube


def fix_invalid_units(cubes):
    """Fix invalid units."""
    for cube in cubes:
        attributes = cube.attributes
        if 'invalid_units' in attributes:
            units = attributes['invalid_units']

            # Fix kg(x) cases
            if 'kg(' in units:
                units = re.sub(r'\((.*?)\)', '', units)

            # Fix other cases
            units = INVALID_UNITS.get(units, units)
            units = units.replace('**', '^')

            # Replace it
            try:
                cube.units = Unit(units)
            except ValueError:
                logger.warning("Could not fix invalid units '%s'",
                               attributes['invalid_units'])
            else:
                attributes.pop('invalid_units')


def fix_var_metadata(cube, var_info):
    """Fix var metadata from CMOR table."""
    try:
        cube.standard_name = var_info.standard_name
    except ValueError:
        cube.standard_name = None
        logger.debug("Got invalid standard_name '%s' for variable '%s'",
                     var_info.standard_name, var_info.short_name)
    cube.var_name = var_info.short_name
    cube.long_name = var_info.long_name
    _set_units(cube, var_info.units)
    return cube


def flip_dim_coord(cube, coord_name):
    """Flip (reverse) dimensional coordinate of cube."""
    logger.info("Flipping dimensional coordinate '%s'", coord_name)
    coord = cube.coord(coord_name, dim_coords=True)
    coord_idx = cube.coord_dims(coord)[0]
    coord.points = np.flip(coord.points)
    if coord.bounds is not None:
        coord.bounds = np.flip(coord.bounds, axis=0)
    cube.data = da.flip(cube.core_data(), axis=coord_idx)


def get_var_info(variable):
    """Get variable information from correct CMOR table."""
    mip = variable['mip']
    short_name = variable['short_name']
    table_entry = CMOR_TABLES[variable['project']].get_variable(
        mip, short_name)
    if table_entry is None and 'derive' in variable:
        table_entry = CMOR_TABLES['custom'].get_variable(mip, short_name)
    if table_entry is None:
        raise ValueError(
            f"Unable to load CMOR table for variable '{short_name}' with mip "
            f"'{mip}' (including custom tables)")
    return table_entry


def is_increasing(cube, coord_name):
    """Check if coordinate of cube is increasing."""
    coord_points = cube.coord(coord_name, dim_coords=True).points
    if len(coord_points) < 2:
        logger.warning(
            "Cannot check if coordinate '%s' is increasing, it only contains "
            "%i elements", coord_name, len(coord_points))
        return None
    if coord_points[1] > coord_points[0]:
        return True
    return False


def set_global_atts(cube, variable, var_info):
    """Complete the cmorized file with global metadata."""
    logger.info("Setting global metadata")
    timestamp = datetime.datetime.utcnow()
    timestamp_format = "%Y-%m-%d %H:%M:%S"
    now_time = timestamp.strftime(timestamp_format)
    attrs = {
        'title': f"CMORized output for {variable['project']}",
        'CMORized_by': f"ESMValTool {version}",
        'CMORized_on': f"{now_time} UTC",
        'project': variable['project'],
        'mip': variable['mip'],
        'dataset_id': variable['dataset'],
        'frequency': var_info.frequency,
    }
    if hasattr(var_info, 'modeling_realm'):
        attrs['modeling_realm'] = var_info.modeling_realm
    cube.attributes.update(attrs)


def _fix_bounds(cube, dim_coord):
    """Reset and fix all bounds."""
    if len(cube.coord(dim_coord).points) > 1:
        if cube.coord(dim_coord).has_bounds():
            cube.coord(dim_coord).bounds = None
        cube.coord(dim_coord).guess_bounds()

    if cube.coord(dim_coord).has_bounds():
        cube.coord(dim_coord).bounds = da.array(
            cube.coord(dim_coord).core_bounds(), dtype='float64')
    return cube


def _fix_dim_coordnames(cube):
    """Perform a check on dim coordinate names."""
    # first check for CMOR standard coord;
    for coord in cube.coords():
        # guess the CMOR-standard x, y, z and t axes if not there
        coord_type = iris.util.guess_coord_axis(coord)

        if coord_type == 'T':
            cube.coord(axis=coord_type).var_name = 'time'
            cube.coord(axis=coord_type).attributes = {}

        if coord_type == 'X':
            cube.coord(axis=coord_type).var_name = 'lon'
            cube.coord(axis=coord_type).standard_name = 'longitude'
            cube.coord(axis=coord_type).long_name = 'longitude coordinate'
            cube.coord(axis=coord_type).units = Unit('degrees')
            cube.coord(axis=coord_type).attributes = {}

        if coord_type == 'Y':
            cube.coord(axis=coord_type).var_name = 'lat'
            cube.coord(axis=coord_type).standard_name = 'latitude'
            cube.coord(axis=coord_type).long_name = 'latitude coordinate'
            cube.coord(axis=coord_type).units = Unit('degrees')
            cube.coord(axis=coord_type).attributes = {}

        if coord_type == 'Z':
            if cube.coord(axis=coord_type).var_name == 'depth':
                cube.coord(axis=coord_type).standard_name = 'depth'
                cube.coord(axis=coord_type).long_name = \
                    'ocean depth coordinate'
                cube.coord(axis=coord_type).var_name = 'lev'
                cube.coord(axis=coord_type).attributes['positive'] = 'down'
            if cube.coord(axis=coord_type).var_name == 'pressure':
                cube.coord(axis=coord_type).standard_name = 'air_pressure'
                cube.coord(axis=coord_type).long_name = 'pressure'
                cube.coord(axis=coord_type).var_name = 'air_pressure'
                cube.coord(axis=coord_type).attributes['positive'] = 'up'

    return cube


def _roll_cube_data(cube, shift, axis):
    """Roll a cube data on specified axis."""
    cube.data = da.roll(cube.core_data(), shift, axis=axis)
    return cube


def _set_units(cube, units):
    """Set units in compliance with cf_unit."""
    try:
        cube.convert_units(units)
    except ValueError:
        logger.warning("Could not convert units '%s' to CMOR units '%s'",
                       cube.units, units)
    return cube
