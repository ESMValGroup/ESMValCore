"""Preprocessor functions for ancillary variables and cell measures."""

import logging
from typing import Iterable

import iris.coords
import iris.cube

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
