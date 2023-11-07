"""Automatically derive variables."""

import importlib
import logging
from copy import deepcopy
from pathlib import Path

import iris

logger = logging.getLogger(__name__)


def _get_all_derived_variables():
    """Get all possible derived variables.

    Returns
    -------
    dict
        All derived variables with `short_name` (keys) and the associated
        python classes (values).

    """
    derivers = {}
    for path in Path(__file__).parent.glob('[a-z]*.py'):
        short_name = path.stem
        module = importlib.import_module(
            f'esmvalcore.preprocessor._derive.{short_name}')
        derivers[short_name] = getattr(module, 'DerivedVariable')
    return derivers


ALL_DERIVED_VARIABLES = _get_all_derived_variables()

__all__ = list(ALL_DERIVED_VARIABLES)


def get_required(short_name, project):
    """Return all required variables for derivation.

    Get all information (at least `short_name`) required for derivation.

    Parameters
    ----------
    short_name : str
        `short_name` of the variable to derive.
    project : str
        `project` of the variable to derive.

    Returns
    -------
    list
        List of dictionaries (including at least the key `short_name`).

    """
    if short_name not in ALL_DERIVED_VARIABLES:
        raise NotImplementedError(
            f"Cannot derive variable '{short_name}', no derivation script "
            f"available")
    DerivedVariable = ALL_DERIVED_VARIABLES[short_name]  # noqa: N806
    variables = deepcopy(DerivedVariable().required(project))
    return variables


def derive(cubes, short_name, long_name, units, standard_name=None):
    """Derive variable.

    Parameters
    ----------
    cubes: iris.cube.CubeList
        Includes all the needed variables for derivation defined in
        :func:`get_required`.
    short_name: str
        short_name
    long_name: str
        long_name
    units: str
        units
    standard_name: str, optional
        standard_name

    Returns
    -------
    iris.cube.Cube
        The new derived variable.

    """
    if short_name == cubes[0].var_name:
        return cubes[0]

    cubes = iris.cube.CubeList(cubes)

    # Derive variable
    DerivedVariable = ALL_DERIVED_VARIABLES[short_name.lower()]  # noqa: N806
    try:
        cube = DerivedVariable().calculate(cubes)
    except Exception as exc:
        msg = (f"Derivation of variable '{short_name}' failed. If you used "
               f"the option '--skip_nonexistent' for running your recipe, "
               f"this might be caused by missing input data for derivation")
        raise ValueError(msg) from exc

    # Set standard attributes
    cube.var_name = short_name
    cube.standard_name = standard_name if standard_name else None
    cube.long_name = long_name
    for temp in cubes:
        if 'source_file' in temp.attributes:
            cube.attributes['source_file'] = temp.attributes['source_file']

    # Check/convert units
    if cube.units is None or cube.units == units:
        cube.units = units
    elif cube.units.is_no_unit() or cube.units.is_unknown():
        logger.warning(
            "Units of cube after executing derivation script of '%s' are "
            "'%s', automatically setting them to '%s'. This might lead to "
            "incorrect data", short_name, cube.units, units)
        cube.units = units
    elif cube.units.is_convertible(units):
        cube.convert_units(units)
    else:
        raise ValueError(
            f"Units '{cube.units}' after executing derivation script of "
            f"'{short_name}' cannot be converted to target units '{units}'")

    return cube
