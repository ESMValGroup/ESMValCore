"""CMORize EMAC data."""
import logging

import iris

from esmvalcore._data_finder import get_start_end_year, load_mapping
from esmvalcore.cmor.table import CMOR_TABLES
from esmvalcore.preprocessor._emac._derive import (
    var_name_constraint, get_derive_function)
from esmvalcore.preprocessor._io import fix_cube_attributes

logger = logging.getLogger(__name__)


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


def _extract_cubes(in_files, short_name, project, mapping_file):
    """Extract cubes from files."""
    cubes = iris.cube.CubeList()
    mapping = load_mapping(short_name, project, mapping_file)
    constraints = [var_name_constraint(var_name) for var_name in mapping]
    for in_file in in_files:
        cubes.extend(iris.load(in_file, constraints=constraints))
    out_cubes = iris.cube.CubeList()
    for var_name in mapping:
        cube = _concatenate(cubes, var_name)
        if cube is None:
            raise ValueError(
                f"{project} CMORization of '{short_name}' failed, the files\n"
                f"{in_files}\ndo not contain the necessary variable "
                f"'{var_name}'")
        out_cubes.append(cube)
    return out_cubes


def _fix_attributes(cube, var_info):
    """Fix attributes of cube."""
    pass


def _fix_metadata(cube, var_info):
    """Fix metedata of cube according to CMOR table."""
    _fix_names(cube, var_info)
    _fix_attributes(cube, var_info)


def _fix_names(cube, var_info):
    """Fix names of cube."""
    cube.var_name = var_info.short_name
    cube.long_name = var_info.long_name
    try:
        cube.standard_name = var_info.standard_name
    except ValueError:
        logger.warning("Got invalid standard_name '%s' for variable '%s'",
                       var_info.standard_name, var_info.short_name)


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


def _group_files(in_files, variable):
    """Group files by start and end year."""
    grouped_files = {}
    for in_file in in_files:
        years = get_start_end_year(in_file, variable)
        grouped_files.setdefault(years, [])
        grouped_files[years].append(in_file)
    return grouped_files


def _unify_metadata(cubes, var_name=None):
    """Unify metadata of cubes."""
    fix_cube_attributes(cubes)
    if not cubes:
        return
    attributes = cubes[0].attributes
    for cube in cubes:
        cube.standard_name = None
        cube.long_name = None
        cube.units = None
        cube.cell_methods = ()
        cube.attributes = attributes
        if var_name is not None:
            cube.var_name = var_name


def cmorize(in_files, variable, var_mapping, output_dir):
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

    # Group input files by year
    grouped_files = _group_files(in_files, variable)

    # Extract only necessary variables and concatenate cubes
    for (years, files) in grouped_files.items():
        cubes = _extract_cubes(files, short_name, project, var_mapping)

        # Derivation
        cube = _derive(cubes, short_name)

        # Fix metadata
        cube = _fix_metadata(cube, var_info)

    return output_dir
