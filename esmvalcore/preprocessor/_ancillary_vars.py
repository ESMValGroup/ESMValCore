"""Preprocessor functions for ancillary variables and cell measures."""

import logging

import dask.array as da
import iris

from esmvalcore.cmor.check import cmor_check_data, cmor_check_metadata
from esmvalcore.cmor.fix import fix_data, fix_metadata
from esmvalcore.preprocessor._io import concatenate, concatenate_callback, load
from esmvalcore.preprocessor._time import clip_timerange

logger = logging.getLogger(__name__)


def _load_fx(var_cube, fx_info, check_level):
    """Load and CMOR-check fx variables."""
    fx_cubes = iris.cube.CubeList()

    project = fx_info['project']
    mip = fx_info['mip']
    short_name = fx_info['short_name']
    freq = fx_info['frequency']

    for fx_file in fx_info['filename']:
        loaded_cube = load(fx_file, callback=concatenate_callback)
        loaded_cube = fix_metadata(loaded_cube,
                                   check_level=check_level,
                                   **fx_info)
        fx_cubes.append(loaded_cube[0])

    fx_cube = concatenate(fx_cubes)

    if freq != 'fx':
        fx_cube = clip_timerange(fx_cube, fx_info['timerange'])

    if not _is_fx_broadcastable(fx_cube, var_cube):
        return None

    fx_cube = cmor_check_metadata(fx_cube,
                                  cmor_table=project,
                                  mip=mip,
                                  short_name=short_name,
                                  frequency=freq,
                                  check_level=check_level)

    fx_cube = fix_data(fx_cube, check_level=check_level, **fx_info)

    fx_cube = cmor_check_data(fx_cube,
                              cmor_table=project,
                              mip=mip,
                              short_name=fx_cube.var_name,
                              frequency=freq,
                              check_level=check_level)

    return fx_cube


def _is_fx_broadcastable(fx_cube, cube):
    try:
        da.broadcast_to(fx_cube.core_data(), cube.shape)
    except ValueError as exc:
        logger.debug(
            "Dimensions of %s and %s cubes do not match. "
            "Discarding use of fx_variable: %s", cube.var_name,
            fx_cube.var_name, exc)
        return False
    return True


def add_cell_measure(cube, fx_cube, measure):
    """Add fx cube as a cell_measure in the cube containing the data.

    Parameters
    ----------
    cube: iris.cube.Cube
        Iris cube with input data.
    fx_cube: iris.cube.Cube
        Iris cube with fx data.
    measure: str
        Name of the measure, can be 'area' or 'volume'.

    Returns
    -------
    iris.cube.Cube
        Cube with added ancillary variables

    Raises
    ------
    ValueError
        If measure name is not 'area' or 'volume'.
    """
    if measure not in ['area', 'volume']:
        raise ValueError(f"measure name must be 'area' or 'volume', "
                         f"got {measure} instead")
    measure = iris.coords.CellMeasure(fx_cube.core_data(),
                                      standard_name=fx_cube.standard_name,
                                      units=fx_cube.units,
                                      measure=measure,
                                      var_name=fx_cube.var_name,
                                      attributes=fx_cube.attributes)
    start_dim = cube.ndim - len(measure.shape)
    cube.add_cell_measure(measure, range(start_dim, cube.ndim))
    logger.debug('Added %s as cell measure in cube of %s.', fx_cube.var_name,
                 cube.var_name)


def add_ancillary_variable(cube, fx_cube):
    """Add fx cube as an ancillary_variable in the cube containing the data.

    Parameters
    ----------
    cube: iris.cube.Cube
        Iris cube with input data.
    fx_cube: iris.cube.Cube
        Iris cube with fx data.

    Returns
    -------
    iris.cube.Cube
        Cube with added ancillary variables
    """
    ancillary_var = iris.coords.AncillaryVariable(
        fx_cube.core_data(),
        standard_name=fx_cube.standard_name,
        units=fx_cube.units,
        var_name=fx_cube.var_name,
        attributes=fx_cube.attributes)
    start_dim = cube.ndim - len(ancillary_var.shape)
    cube.add_ancillary_variable(ancillary_var, range(start_dim, cube.ndim))
    logger.debug('Added %s as ancillary variable in cube of %s.',
                 fx_cube.var_name, cube.var_name)


def add_fx_variables(cube, fx_variables, check_level):
    """Load requested fx files, check with CMOR standards and add the fx
    variables as cell measures or ancillary variables in the cube containing
    the data.

    Parameters
    ----------
    cube: iris.cube.Cube
        Iris cube with input data.
    fx_variables: dict
        Dictionary with fx_variable information.
    check_level: CheckLevels
        Level of strictness of the checks.


    Returns
    -------
    iris.cube.Cube
        Cube with added cell measures or ancillary variables.
    """

    if not fx_variables:
        return cube
    for fx_info in fx_variables.values():
        if not fx_info:
            continue
        if isinstance(fx_info['filename'], str):
            fx_info['filename'] = [fx_info['filename']]
        fx_cube = _load_fx(cube, fx_info, check_level)

        if fx_cube is None:
            continue

        measure_name = {
            'areacella': 'area',
            'areacello': 'area',
            'volcello': 'volume'
        }

        if fx_cube.var_name in measure_name:
            add_cell_measure(cube, fx_cube, measure_name[fx_cube.var_name])
        else:
            add_ancillary_variable(cube, fx_cube)
    return cube


def remove_fx_variables(cube):
    """Remove fx variables present as cell measures or ancillary variables in
    the cube containing the data.

    Parameters
    ----------
    cube: iris.cube.Cube
        Iris cube with  data and cell measures or ancillary variables.


    Returns
    -------
    iris.cube.Cube
        Cube without cell measures or ancillary variables.
    """

    if cube.cell_measures():
        for measure in cube.cell_measures():
            cube.remove_cell_measure(measure.standard_name)
    if cube.ancillary_variables():
        for variable in cube.ancillary_variables():
            cube.remove_ancillary_variable(variable.standard_name)
    return cube
