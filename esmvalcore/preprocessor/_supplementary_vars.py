"""Preprocessor functions for ancillary variables and cell measures."""

import logging
from collections.abc import Callable
from typing import Iterable, Literal

import iris.coords
from iris.cube import Cube

logger = logging.getLogger(__name__)

PREPROCESSOR_SUPPLEMENTARIES = {}


def register_supplementaries(
    variables: list[str],
    required: Literal["require_at_least_one", "prefer_at_least_one"],
) -> Callable:
    """Register supplementary variables required for a preprocessor function.

    Parameters
    ----------
    variables:
        List of variable names.
    required:
        How strong the requirement is. Can be 'require_at_least_one' if at
        least one variable must be available or 'prefer_at_least_one' if it is
        preferred that at least one variable is available, but not strictly
        necessary.
    """
    valid = ("require_at_least_one", "prefer_at_least_one")
    if required not in valid:
        raise NotImplementedError(f"`required` should be one of {valid}")
    supplementaries = {
        "variables": variables,
        "required": required,
    }

    def wrapper(func):
        PREPROCESSOR_SUPPLEMENTARIES[func.__name__] = supplementaries
        return func

    return wrapper


def add_cell_measure(
    cube: Cube,
    cell_measure_cube: Cube,
    measure: Literal["area", "volume"],
) -> None:
    """Add cell measure to cube (in-place).

    Note
    ----
    This assumes that the cell measure spans the rightmost dimensions of the
    cube.

    Parameters
    ----------
    cube:
        Iris cube with input data.
    cell_measure_cube:
        Iris cube with cell measure data.
    measure:
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
    if measure not in ["area", "volume"]:
        raise ValueError(
            f"measure name must be 'area' or 'volume', got {measure} instead"
        )
    coord_dims = tuple(
        range(cube.ndim - len(cell_measure_cube.shape), cube.ndim)
    )
    cell_measure_data = cell_measure_cube.core_data()
    if cell_measure_cube.has_lazy_data():
        cube_chunks = tuple(cube.lazy_data().chunks[d] for d in coord_dims)
        cell_measure_data = cell_measure_data.rechunk(cube_chunks)
    cell_measure = iris.coords.CellMeasure(
        cell_measure_data,
        standard_name=cell_measure_cube.standard_name,
        units=cell_measure_cube.units,
        measure=measure,
        var_name=cell_measure_cube.var_name,
        attributes=cell_measure_cube.attributes,
    )
    cube.add_cell_measure(cell_measure, coord_dims)
    logger.debug(
        "Added %s as cell measure in cube of %s.",
        cell_measure_cube.var_name,
        cube.var_name,
    )


def add_ancillary_variable(cube: Cube, ancillary_cube: Cube) -> None:
    """Add ancillary variable to cube (in-place).

    Note
    ----
    This assumes that the ancillary variable spans the rightmost dimensions of
    the cube.

    Parameters
    ----------
    cube:
        Iris cube with input data.
    ancillary_cube:
        Iris cube with ancillary data.

    Returns
    -------
    iris.cube.Cube
        Cube with added ancillary variables
    """
    coord_dims = tuple(range(cube.ndim - len(ancillary_cube.shape), cube.ndim))
    ancillary_data = ancillary_cube.core_data()
    if ancillary_cube.has_lazy_data():
        cube_chunks = tuple(cube.lazy_data().chunks[d] for d in coord_dims)
        ancillary_data = ancillary_data.rechunk(cube_chunks)
    ancillary_var = iris.coords.AncillaryVariable(
        ancillary_data,
        standard_name=ancillary_cube.standard_name,
        units=ancillary_cube.units,
        var_name=ancillary_cube.var_name,
        attributes=ancillary_cube.attributes,
    )
    cube.add_ancillary_variable(ancillary_var, coord_dims)
    logger.debug(
        "Added %s as ancillary variable in cube of %s.",
        ancillary_cube.var_name,
        cube.var_name,
    )


def add_supplementary_variables(
    cube: Cube,
    supplementary_cubes: Iterable[Cube],
) -> Cube:
    """Add ancillary variables and/or cell measures to cube (in-place).

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
    measure_names: dict[str, Literal["area", "volume"]] = {
        "areacella": "area",
        "areacello": "area",
        "volcello": "volume",
    }
    for supplementary_cube in supplementary_cubes:
        if supplementary_cube.var_name in measure_names:
            measure_name = measure_names[supplementary_cube.var_name]
            add_cell_measure(cube, supplementary_cube, measure_name)
        else:
            add_ancillary_variable(cube, supplementary_cube)
    return cube


def remove_supplementary_variables(cube: Cube) -> Cube:
    """Remove supplementary variables from cube (in-place).

    Strip cell measures or ancillary variables from the cube.

    Parameters
    ----------
    cube:
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
