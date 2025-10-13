"""Automatically derive variables."""

import importlib
import logging
from copy import deepcopy
from pathlib import Path

from cf_units import Unit
from iris.cube import Cube, CubeList

from esmvalcore.preprocessor._derive._baseclass import DerivedVariableBase
from esmvalcore.preprocessor._units import convert_units
from esmvalcore.typing import Facets

logger = logging.getLogger(__name__)


def _get_all_derived_variables() -> dict[str, type[DerivedVariableBase]]:
    """Get all possible derived variables.

    Returns
    -------
    dict
        All derived variables with `short_name` (keys) and the associated
        python classes (values).
    """
    derivers = {}
    for path in Path(__file__).parent.glob("[a-z]*.py"):
        short_name = path.stem
        module = importlib.import_module(
            f"esmvalcore.preprocessor._derive.{short_name}",
        )
        derivers[short_name] = module.DerivedVariable
    return derivers


ALL_DERIVED_VARIABLES: dict[str, type[DerivedVariableBase]] = (
    _get_all_derived_variables()
)

__all__ = list(ALL_DERIVED_VARIABLES)


def get_required(short_name: str, project: str) -> list[Facets]:
    """Return all required variables for derivation.

    Get all information (at least ``short_name``) required for derivation.

    Parameters
    ----------
    short_name:
        Short name of the variable to derive.
    project:
        Project of the variable to derive.

    Returns
    -------
    list[esmvalcore.typing.Facets]
        List of facets (including at least the key ``short_name``).

    """
    if short_name.lower() not in ALL_DERIVED_VARIABLES:
        msg = (
            f"Cannot derive variable '{short_name}': no derivation script "
            f"available"
        )
        raise NotImplementedError(msg)
    DerivedVariable = ALL_DERIVED_VARIABLES[short_name.lower()]  # noqa: N806
    return deepcopy(DerivedVariable().required(project))


def derive(
    cubes: CubeList,
    short_name: str,
    long_name: str,
    units: str | Unit,
    standard_name: str | None = None,
) -> Cube:
    """Derive variable.

    Parameters
    ----------
    cubes:
        Includes all the needed variables for derivation defined in
        :func:`get_required`.
    short_name:
        short_name
    long_name:
        long_name
    units:
        units
    standard_name:
        standard_name

    Returns
    -------
    iris.cube.Cube
        The new derived variable.
    """
    if short_name == cubes[0].var_name:
        return cubes[0]

    cubes = CubeList(cubes)

    # Derive variable
    DerivedVariable = ALL_DERIVED_VARIABLES[short_name.lower()]  # noqa: N806
    try:
        cube = DerivedVariable().calculate(cubes)
    except Exception as exc:
        msg = (
            f"Derivation of variable '{short_name}' failed. If you used "
            f"the option '--skip_nonexistent' for running your recipe, "
            f"this might be caused by missing input data for derivation"
        )
        raise ValueError(msg) from exc

    # Set standard attributes
    cube.var_name = short_name
    cube.standard_name = standard_name if standard_name else None
    cube.long_name = long_name
    for temp in cubes:
        if "source_file" in temp.attributes:
            cube.attributes["source_file"] = temp.attributes["source_file"]

    # Check/convert units
    if cube.units is None or cube.units == units:
        cube.units = units
    elif cube.units.is_no_unit() or cube.units.is_unknown():
        logger.warning(
            "Units of cube after executing derivation script of '%s' are "
            "'%s', automatically setting them to '%s'. This might lead to "
            "incorrect data",
            short_name,
            cube.units,
            units,
        )
        cube.units = units
    else:
        try:
            convert_units(cube, units)
        except ValueError as exc:
            msg = (
                f"Units '{cube.units}' after executing derivation script of "
                f"'{short_name}' cannot be converted to target units '{units}'"
            )
            raise ValueError(msg) from exc

    return cube
