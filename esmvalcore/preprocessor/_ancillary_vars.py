"""Preprocessor functions for ancillary variables and cell measures."""

import logging

import dask.array as da
import iris

logger = logging.getLogger(__name__)

PREPROCESSOR_ANCILLARIES = {}


def register_ancillaries(variables, required):
    """Register ancillary variables required for a preprocessor function.

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
    ancillaries = {
        'variables': variables,
        'required': required,
    }

    def wrapper(func):
        PREPROCESSOR_ANCILLARIES[func.__name__] = ancillaries
        return func

    return wrapper


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


def add_fx_variables(cube, fx_variables):
    """Load requested fx files, check with CMOR standards and add the fx
    variables as cell measures or ancillary variables in the cube containing
    the data.

    Parameters
    ----------
    cube: iris.cube.Cube
        Iris cube with input data.
    fx_variables: :obj:`list` of :obj:`iris.cube.Cube`
        Cubes containing ancillary variables.

    Returns
    -------
    iris.cube.Cube
        Cube with added cell measures or ancillary variables.
    """
    # note: backwards incompatible change to function signature
    if not fx_variables:
        return cube

    measure_name = {
        'areacella': 'area',
        'areacello': 'area',
        'volcello': 'volume'
    }

    for fx_cube in fx_variables:
        if not _is_fx_broadcastable(fx_cube, cube):
            continue

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
