"""CMORizer for certain projects."""
import datetime
import importlib.util
import logging
import os

import iris
import numpy as np
from cf_units import Unit
from dask import array as da

from esmvalcore import __version__ as version
from esmvalcore._config import get_tag_value

logger = logging.getLogger(__name__)


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
            "CMORizer %s given in 'config-developer.yml' is not a valid "
            "CMORizer, make sure that it exists and that it contains a "
            "function called 'cmorize'", cmorizer)
        raise
    if not hasattr(cmorizer_module, 'cmorize'):
        raise ValueError(
            f"CMORizer {cmorizer} does not contain a function called "
            f"'cmorize'")
    cubes = iris.cube.CubeList(cubes)
    logger.debug("Successfully loaded CMORizer %s", cmorizer)
    return cmorizer_module.cmorize(cubes, variable, var_mapping)


def add_height2m(cube):
    """Add scalar coordinate 'height' with value of 2m."""
    logger.info("Adding height coordinate (2m)")
    height_coord = iris.coords.AuxCoord(
        2.0,
        var_name='height',
        standard_name='height',
        long_name='height',
        units=Unit('m'),
        attributes={'positive': 'up'})
    cube.add_aux_coord(height_coord, ())


def convert_timeunits(cube, start_year):
    """Convert time axis from malformed Year 0."""
    # TODO any more weird cases?
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

        # fix latitude
        if cube_coord.var_name == 'lat':
            logger.info("Fixing latitude")
            _fix_bounds(cube, cube.coord('latitude'))

        # fix depth
        if cube_coord.var_name == 'lev':
            logger.info("Fixing depth")
            _fix_bounds(cube, cube.coord('depth'))

        # fix air_pressure
        if cube_coord.var_name == 'air_pressure':
            logger.info("Fixing air pressure")
            _fix_bounds(cube, cube.coord('air_pressure'))

    # remove CS
    cube.coord('latitude').coord_system = None
    cube.coord('longitude').coord_system = None

    return cube


def fix_var_metadata(cube, var_info):
    """Fix var metadata from CMOR table."""
    try:
        cube.standard_name = var_info.standard_name
    except ValueError:
        cube.standard_name = None
        logger.warning("Got invalid standard_name '%s' for variable '%s'",
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
    coord.bounds = np.flip(coord.bounds, axis=0)
    cube.data = da.flip(cube.core_data(), axis=coord_idx)


def save_variable(cube, var, outdir, attrs, **kwargs):
    """Saver function."""
    # CMOR standard
    cube_time = cube.coord('time')
    reftime = Unit(cube_time.units.origin, cube_time.units.calendar)
    dates = reftime.num2date(cube_time.points[[0, -1]])
    if len(cube_time.points) == 1:
        year = str(dates[0].year)
        time_suffix = '-'.join([year + '01', year + '12'])
    else:
        date1 = str(dates[0].year) + '%02d' % dates[0].month
        date2 = str(dates[1].year) + '%02d' % dates[1].month
        time_suffix = '-'.join([date1, date2])

    file_name = '_'.join([
        'OBS',
        attrs['dataset_id'],
        attrs['modeling_realm'],
        attrs['version'],
        attrs['mip'],
        var,
        time_suffix,
    ]) + '.nc'
    file_path = os.path.join(outdir, file_name)
    logger.info('Saving: %s', file_path)
    status = 'lazy' if cube.has_lazy_data() else 'realized'
    logger.info('Cube has %s data [lazy is preferred]', status)
    iris.save(cube, file_path, fill_value=1e20, **kwargs)


def set_global_atts(cube, variable):
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
    }
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
