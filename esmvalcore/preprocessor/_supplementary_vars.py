"""Preprocessor functions for ancillary variables and cell measures."""

import logging
import warnings
from pathlib import Path
from typing import Iterable

import dask.array as da
import iris.coords
import iris.cube

from esmvalcore.cmor.check import cmor_check_data, cmor_check_metadata
from esmvalcore.cmor.fix import fix_data, fix_metadata
from esmvalcore.config import CFG
from esmvalcore.config._config import get_ignored_warnings
from esmvalcore.exceptions import ESMValCoreDeprecationWarning
from esmvalcore.preprocessor._io import concatenate, load
from esmvalcore.preprocessor._time import clip_timerange

logger = logging.getLogger(__name__)

PREPROCESSOR_SUPPLEMENTARIES = {}


def register_supplementaries(variables, required):
    """Register supplementary variables required for a preprocessor function.

    Parameters
    ----------
    variables: :obj:`list` of :obj`str`
        List of variable names.
    required:
        How strong the requirement is. Can be 'require_at_least_one' if at
        least one variable must be available or 'prefer_at_least_one' if it is
        preferred that at least one variable is available, but not strictly
        necessary.
    """
    valid = ('require_at_least_one', 'prefer_at_least_one')
    if required not in valid:
        raise NotImplementedError(f"`required` should be one of {valid}")
    supplementaries = {
        'variables': variables,
        'required': required,
    }

    def wrapper(func):
        PREPROCESSOR_SUPPLEMENTARIES[func.__name__] = supplementaries
        return func

    return wrapper


def _load_fx(var_cube, fx_info, check_level):
    """Load and CMOR-check fx variables."""
    fx_cubes = iris.cube.CubeList()

    project = fx_info['project']
    mip = fx_info['mip']
    short_name = fx_info['short_name']
    freq = fx_info['frequency']

    for fx_file in fx_info['filename']:
        ignored_warnings = get_ignored_warnings(project, 'load')
        loaded_cube = load(fx_file, ignore_warnings=ignored_warnings)
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


def add_cell_measure(cube, cell_measure_cube, measure):
    """Add a cube as a cell_measure in the cube containing the data.

    Parameters
    ----------
    cube: iris.cube.Cube
        Iris cube with input data.
    cell_measure_cube: iris.cube.Cube
        Iris cube with cell measure data.
    measure: str
        Name of the measure, can be 'area' or 'volume'.

    Returns
    -------
    iris.cube.Cube
        Cube with added cell measure

    Raises
    ------
    ValueError
        If measure name is not 'area' or 'volume'.
    """
    if measure not in ['area', 'volume']:
        raise ValueError(f"measure name must be 'area' or 'volume', "
                         f"got {measure} instead")
    measure = iris.coords.CellMeasure(
        cell_measure_cube.core_data(),
        standard_name=cell_measure_cube.standard_name,
        units=cell_measure_cube.units,
        measure=measure,
        var_name=cell_measure_cube.var_name,
        attributes=cell_measure_cube.attributes,
    )
    start_dim = cube.ndim - len(measure.shape)
    cube.add_cell_measure(measure, range(start_dim, cube.ndim))
    logger.debug('Added %s as cell measure in cube of %s.',
                 cell_measure_cube.var_name, cube.var_name)


def add_ancillary_variable(cube, ancillary_cube):
    """Add cube as an ancillary variable in the cube containing the data.

    Parameters
    ----------
    cube: iris.cube.Cube
        Iris cube with input data.
    ancillary_cube: iris.cube.Cube
        Iris cube with ancillary data.

    Returns
    -------
    iris.cube.Cube
        Cube with added ancillary variables
    """
    ancillary_var = iris.coords.AncillaryVariable(
        ancillary_cube.core_data(),
        standard_name=ancillary_cube.standard_name,
        units=ancillary_cube.units,
        var_name=ancillary_cube.var_name,
        attributes=ancillary_cube.attributes)
    start_dim = cube.ndim - len(ancillary_var.shape)
    cube.add_ancillary_variable(ancillary_var, range(start_dim, cube.ndim))
    logger.debug('Added %s as ancillary variable in cube of %s.',
                 ancillary_cube.var_name, cube.var_name)


def add_fx_variables(cube, fx_variables, check_level):
    """Load requested fx files, check with CMOR standards and add the fx
    variables as cell measures or ancillary variables in the cube containing
    the data.

    .. deprecated:: 2.8.0
        This function is deprecated and will be removed in version 2.10.0.
        Please use a :class:`esmvalcore.dataset.Dataset` or
        :func:`esmvalcore.preprocessor.add_supplementary_variables`
        instead.

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
    msg = (
        "The function `add_fx_variables` has been deprecated in "
        "ESMValCore version 2.8.0 and is scheduled for removal in "
        "version 2.10.0. Use a `esmvalcore.dataset.Dataset` or the function "
        "`add_supplementary_variables` instead.")
    warnings.warn(msg, ESMValCoreDeprecationWarning)
    if not fx_variables:
        return cube
    fx_cubes = []
    for fx_info in fx_variables.values():
        if not fx_info:
            continue
        if isinstance(fx_info['filename'], (str, Path)):
            fx_info['filename'] = [fx_info['filename']]
        fx_cube = _load_fx(cube, fx_info, check_level)

        if fx_cube is None:
            continue

        fx_cubes.append(fx_cube)

    add_supplementary_variables(cube, fx_cubes)
    return cube


def add_supplementary_variables(
    cube: iris.cube.Cube,
    supplementary_cubes: Iterable[iris.cube.Cube],
) -> iris.cube.Cube:
    """Add ancillary variables and/or cell measures.

    Parameters
    ----------
    cube:
        Cube to add to.
    supplementary_cubes:
        Iterable of cubes containing the supplementary variables.

    Returns
    -------
    iris.cube.Cube
        Cube with added ancillary variables and/or cell measures.
    """
    measure_names = {
        'areacella': 'area',
        'areacello': 'area',
        'volcello': 'volume'
    }
    for supplementary_cube in supplementary_cubes:
        if (CFG['use_legacy_supplementaries']
                and not _is_fx_broadcastable(supplementary_cube, cube)):
            continue
        if supplementary_cube.var_name in measure_names:
            measure_name = measure_names[supplementary_cube.var_name]
            add_cell_measure(cube, supplementary_cube, measure_name)
        else:
            add_ancillary_variable(cube, supplementary_cube)
    return cube


def remove_supplementary_variables(cube: iris.cube.Cube):
    """Remove supplementary variables.

    Strip cell measures or ancillary variables from the cube containing the
    data.

    Parameters
    ----------
    cube: iris.cube.Cube
        Iris cube with data and cell measures or ancillary variables.

    Returns
    -------
    iris.cube.Cube
        Cube without cell measures or ancillary variables.
    """
    if cube.cell_measures():
        for measure in cube.cell_measures():
            cube.remove_cell_measure(measure)
    if cube.ancillary_variables():
        for variable in cube.ancillary_variables():
            cube.remove_ancillary_variable(variable)
    return cube


def remove_fx_variables(cube):
    """Remove fx variables present as cell measures or ancillary variables in
    the cube containing the data.

    .. deprecated:: 2.8.0
        This function is deprecated and will be removed in version 2.10.0.
        Please use
        :func:`esmvalcore.preprocessor.remove_supplementary_variables`
        instead.

    Parameters
    ----------
    cube: iris.cube.Cube
        Iris cube with data and cell measures or ancillary variables.

    Returns
    -------
    iris.cube.Cube
        Cube without cell measures or ancillary variables.
    """
    msg = ("The function `remove_fx_variables` has been deprecated in "
           "ESMValCore version 2.8.0 and is scheduled for removal in "
           "version 2.10.0. Use the function `remove_supplementary_variables` "
           "instead.")
    warnings.warn(msg, ESMValCoreDeprecationWarning)
    return remove_supplementary_variables(cube)
