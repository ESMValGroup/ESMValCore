"""CMORize EMAC data."""
import logging

import iris
from cf_units import Unit

from esmvalcore._data_finder import load_mapping
from esmvalcore.preprocessor._cmorize import (add_scalar_height_coord,
                                              convert_timeunits, fix_coords,
                                              fix_var_metadata, flip_dim_coord,
                                              get_var_info, is_increasing,
                                              set_global_atts)
from esmvalcore.preprocessor._emac._derive import (get_derive_function,
                                                   var_name_constraint)
from esmvalcore.preprocessor._io import fix_cube_attributes

logger = logging.getLogger(__name__)

INVALID_UNITS = {
    'kg/m**2s': 'kg m-2 s-1',
}


def _concatenate(cubes, var_name):
    """Concatenate cubes of certain variable."""
    cubes = cubes.extract(var_name_constraint(var_name))
    if not cubes:
        return None
    _unify_metadata(cubes, var_name)
    try:
        final_cube = cubes.concatenate_cube()
    except Exception:
        logger.error("Concatenation of '%s' cubes failed", var_name)
        raise
    return final_cube


def _derive(cubes, short_name):
    """Derive variable if necessary."""
    _unify_metadata(cubes)
    derive = get_derive_function(short_name)
    if derive is None:
        logger.debug("Deriving '%s' without special function", short_name)
        if len(cubes) != 1:
            raise ValueError(
                f"Expected exactly one cube for derivation of '{short_name}' "
                f"since no derivation explicit derivation function is given "
                f"in esmvalcore.preprocessor._emac._derive, got {len(cubes)}")
        cube = cubes[0]
    else:
        logger.debug("Deriving '%s' using derivation function", short_name)
        cube = derive(cubes)
    cube.var_name = short_name
    return cube


def _extract_cubes(cubes, short_name, project, mapping_file):
    """Extract relevant cubes."""
    mapping = load_mapping(short_name, project, mapping_file)
    constraints = [var_name_constraint(var_name) for var_name in mapping]
    cubes = cubes.extract(constraints)
    out_cubes = iris.cube.CubeList()
    for var_name in mapping:
        cube = _concatenate(cubes, var_name)
        if cube is None:
            raise ValueError(
                f"{project} CMORization of '{short_name}' failed, input cubes "
                f"do not contain the necessary variable '{var_name}'")
        out_cubes.append(cube)
    return out_cubes


def _fix_invalid_units(cubes):
    """Fix invalid units."""
    for cube in cubes:
        attributes = cube.attributes
        if 'invalid_units' in attributes:
            units = attributes['invalid_units']
            units = INVALID_UNITS.get(units, units)
            units = units.replace('**', '^')
            try:
                cube.units = Unit(units)
            except ValueError:
                logger.warning("Could not fix invalid units '%s'",
                               attributes['invalid_units'])
            else:
                attributes.pop('invalid_units')


def _fix_metadata(cube, variable, var_info):
    """Fix metedata of cube according to CMOR table."""
    fix_var_metadata(cube, var_info)
    fix_coords(cube)
    convert_timeunits(cube, 1950)
    set_global_atts(cube, variable, var_info)
    if 'height2m' in var_info.dimensions:
        add_scalar_height_coord(cube, 2.0)
    if 'height10m' in var_info.dimensions:
        add_scalar_height_coord(cube, 10.0)
    try:
        if not is_increasing(cube, 'latitude'):
            flip_dim_coord(cube, 'latitude')
    except iris.exceptions.CoordinateNotFoundError:
        pass


def _unify_metadata(cubes, var_name=None):
    """Unify metadata of cubes."""
    if not cubes:
        return
    _fix_invalid_units(cubes)
    fix_cube_attributes(cubes)
    attributes = cubes[0].attributes
    for cube in cubes:
        cube.standard_name = None
        cube.long_name = None
        cube.cell_methods = ()
        cube.attributes = attributes
        if var_name is not None:
            cube.var_name = var_name


def cmorize(cubes, variable, var_mapping):
    """CMORize EMAC data."""
    project = variable['project']
    short_name = variable['short_name']
    var_info = get_var_info(variable)

    # Extract only necessary variables
    cubes = _extract_cubes(cubes, short_name, project, var_mapping)

    # Derivation
    cube = _derive(cubes, short_name)

    # Fix metadata
    _fix_metadata(cube, variable, var_info)

    return cube
