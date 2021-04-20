"""Preprocessor functions for ancillary variables and cell measures."""

import logging
import iris

import dask.array as da

from esmvalcore.preprocessor._io import load, concatenate_callback, concatenate
from esmvalcore.cmor.fix import fix_metadata, fix_data
from esmvalcore.cmor.check import cmor_check_metadata, cmor_check_data

logger = logging.getLogger(__name__)


def _load_fx(fx_info, check_level):
    """Load and CMOR-check fx variables."""
    fx_cubes = iris.cube.CubeList()

    for fx_file in fx_info['filename']:
        loaded_cube = load(fx_file, callback=concatenate_callback)
        short_name = fx_info['short_name']
        project = fx_info['project']
        dataset = fx_info['dataset']
        mip = fx_info['mip']
        freq = fx_info['frequency']
        loaded_cube = fix_metadata(loaded_cube, short_name=short_name,
                                   project=project, dataset=dataset,
                                   mip=mip, frequency=freq,
                                   check_level=check_level)
        fx_cubes.append(loaded_cube[0])

    fx_cube = concatenate(fx_cubes)

    fx_cube = cmor_check_metadata(fx_cube, cmor_table=project, mip=mip,
                                  short_name=short_name, frequency=freq,
                                  check_level=check_level)

    fx_cube = fix_data(fx_cube, short_name=short_name, project=project,
                       dataset=dataset, mip=mip, frequency=freq,
                       check_level=check_level)

    fx_cube = cmor_check_data(fx_cube, cmor_table=project, mip=mip,
                              short_name=fx_cube.var_name, frequency=freq,
                              check_level=check_level)

    return fx_cube


def add_cell_measure(cube, fx_cube, measure):
    """
    Broadcast fx_cube and add it as a cell_measure in
    the cube containing the data.

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
    ValueError
        If fx_cube cannot be broadcast to cube.
    """
    if measure not in ['area', 'volume']:
        raise ValueError(f"measure name must be 'area' or 'volume', "
                         f"got {measure} instead")
    try:
        fx_data = da.broadcast_to(fx_cube.core_data(), cube.shape)
    except ValueError as exc:
        raise ValueError(f"Dimensions of {cube.var_name} and "
                         f"{fx_cube.var_name} cubes do not match. "
                         "Cannot broadcast cubes.") from exc
    measure = iris.coords.CellMeasure(
        fx_data,
        standard_name=fx_cube.standard_name,
        units=fx_cube.units,
        measure=measure,
        var_name=fx_cube.var_name,
        attributes=fx_cube.attributes)
    cube.add_cell_measure(measure, range(0, measure.ndim))
    logger.debug('Added %s as cell measure in cube of %s.',
                 fx_cube.var_name, cube.var_name)


def add_ancillary_variable(cube, fx_cube):
    """
    Broadcast fx_cube and add it as an ancillary_variable in
    the cube containing the data.

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

    Raises
    ------
    ValueError
        If fx_cube cannot be broadcast to cube.
    """
    try:
        fx_data = da.broadcast_to(fx_cube.core_data(), cube.shape)
    except ValueError as exc:
        raise ValueError(f"Dimensions of {cube.var_name} and "
                         f"{fx_cube.var_name} cubes do not match. "
                         "Cannot broadcast cubes.") from exc
    ancillary_var = iris.coords.AncillaryVariable(
        fx_data,
        standard_name=fx_cube.standard_name,
        units=fx_cube.units,
        var_name=fx_cube.var_name,
        attributes=fx_cube.attributes)
    cube.add_ancillary_variable(ancillary_var, range(0, ancillary_var.ndim))
    logger.debug('Added %s as ancillary variable in cube of %s.',
                 fx_cube.var_name, cube.var_name)


def add_fx_variables(cube, fx_variables, check_level):
    """
    Load requested fx files, check with CMOR standards and add the
    fx variables as cell measures or ancillary variables in
    the cube containing the data.

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
        fx_cube = _load_fx(fx_info, check_level)

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
    """
    Remove fx variables present as cell measures or ancillary variables in
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
