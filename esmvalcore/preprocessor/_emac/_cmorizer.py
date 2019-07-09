"""CMORize EMAC data."""
import logging

import iris
from cf_units import Unit

from esmvalcore._data_finder import load_mapping
from esmvalcore.cmor.table import CMOR_TABLES
from esmvalcore.preprocessor._cmorize import fix_var_metadata, fix_coords, \
    convert_timeunits, set_global_atts
from esmvalcore.preprocessor._emac._derive import (
    var_name_constraint, get_derive_function)
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
                f"{project} CMORization of '{short_name}' failed, the cubes\n"
                f"{cubes}\ndo not contain the necessary variable "
                f"'{var_name}'")
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
    set_global_atts(cube, variable)


def _get_var_info(variable):
    """Get variable information from correct CMOR table."""
    mip = variable['mip']
    short_name = variable['short_name']
    table_entry = CMOR_TABLES[variable['project']].get_variable(mip,
                                                                short_name)
    if table_entry is not None:
        return table_entry
    if 'derive' in variable or 'custom_cmor_table' in variable:
        table_entry = CMOR_TABLES['custom'].get_variable(mip, short_name)
    if table_entry is None:
        raise ValueError(
            f"Unable to load CMOR table for variable '{short_name}' with mip "
            f"'{mip}' (including custom tables)")
    return table_entry


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
    """CMORize EMAC data.

    Note
    ----
    At the moment, this is only a "light" CMORizer, i.e. the option
    'light_cmorizer' has to be set in `config-developer.yml` or the CMOR checks
    of the preprocessor will fail.

    """
    project = variable['project']
    short_name = variable['short_name']
    var_info = _get_var_info(variable)

    # Extract only necessary variables
    cubes = _extract_cubes(cubes, short_name, project, var_mapping)

    # Derivation
    cube = _derive(cubes, short_name)

    # Fix metadata
    _fix_metadata(cube, variable, var_info)

    return cube
