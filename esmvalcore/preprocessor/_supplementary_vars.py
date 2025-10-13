"""Preprocessor functions for ancillary variables and cell measures."""

import logging
from collections.abc import Callable, Iterable
from typing import Literal

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
        msg = f"`required` should be one of {valid}"
        raise NotImplementedError(msg)
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
    cell_measure_cube: Cube | iris.coords.CellMeasure,
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
        Iris cube or cell measure coord with cell measure data.
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
        msg = f"measure name must be 'area' or 'volume', got {measure} instead"
        raise ValueError(
            msg,
        )
    coord_dims = tuple(
        range(cube.ndim - len(cell_measure_cube.shape), cube.ndim),
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


def find_matching_coord_dims(
    coord_to_match: iris.coords.DimCoord,
    cube: iris.cube.Cube,
) -> tuple[int] | None:
    """Find a matching coordinate from the ancillary variable in the cube.

    Parameters
    ----------
    coord_to_match: iris.coords.DimCoord
        Coordinate from an ancillary cube to match.
    cube: iris.cube.Cube
        Iris cube with variable data.

    Returns
    -------
    cube_dims: tuple or None
        Tuple containing the matched cube coordinate dimension for the
        coordinate from the ancillary cube in the data cube. If no match
        is found, None is returned.
    """
    cube_dims = None
    for cube_coord in cube.coords():
        if (
            (
                cube_coord.var_name == coord_to_match.var_name
                or (
                    coord_to_match.standard_name is not None
                    and cube_coord.standard_name
                    == coord_to_match.standard_name
                )
                or (
                    coord_to_match.long_name is not None
                    and cube_coord.long_name == coord_to_match.long_name
                )
            )
            and cube_coord.units == coord_to_match.units
            and cube_coord.shape == coord_to_match.shape
        ):
            cube_dims = cube.coord_dims(cube_coord)
            msg = (
                f"Found a matching coordinate for {coord_to_match.var_name}"
                f" with coordinate {cube_coord.var_name}"
                f" in the cube of variable '{cube.var_name}'."
            )
            logger.debug(msg)
            break
    return cube_dims


def get_data_dims(
    cube: Cube,
    ancillary: Cube | iris.coords.AncillaryVariable,
) -> list[None | int]:
    """Get matching data dimensions between cube and ancillary variable.

    Parameters
    ----------
    cube: iris.cube.Cube
        Iris cube with input data.
    ancillary: Cube or iris.coords.AncillaryVariable
        Iris cube or AncillaryVariable with ancillary data.

    Returns
    -------
    data_dims: list
        List with as many entries as the ancillary variable dimensions.
        The i-th entry corresponds to the i-th ancillary variable
        dimension match with the cube's dimensions. If there is no match
        between the ancillary variable and the cube for a dimension, then entry
        defaults to None.
    """
    # Match the coordinates of the ancillary cube to coordinates and
    # dimensions in the input cube before adding the ancillary variable.
    data_dims: list[None | int] = []
    if isinstance(ancillary, iris.coords.AncillaryVariable):
        start_dim = cube.ndim - len(ancillary.shape)
        data_dims = list(range(start_dim, cube.ndim))
    else:
        data_dims = [None] * ancillary.ndim
        for coord in ancillary.coords():
            try:
                cube_dims = cube.coord_dims(coord)
            except iris.exceptions.CoordinateNotFoundError:
                cube_dims = find_matching_coord_dims(coord, cube)
            if cube_dims is not None:
                for ancillary_dim, cube_dim in zip(
                    ancillary.coord_dims(coord),
                    cube_dims,
                    strict=True,
                ):
                    data_dims[ancillary_dim] = cube_dim
        if None in data_dims:
            none_dims = ", ".join(
                str(i) for i, d in enumerate(data_dims) if d is None
            )
            msg = (
                f"Failed to add\n{ancillary}\nas ancillary var "
                f"to the cube\n{cube}\n"
                f"Mismatch between ancillary cube and variable cube coordinate"
                f" {none_dims}"
            )
            logger.error(msg)
            raise iris.exceptions.CoordinateNotFoundError(msg)
    return data_dims


def add_ancillary_variable(
    cube: Cube,
    ancillary_cube: Cube | iris.coords.AncillaryVariable,
) -> None:
    """Add ancillary variable to cube (in-place).

    Parameters
    ----------
    cube:
        Iris cube with input data.
    ancillary_cube:
        Iris cube or AncillaryVariable with ancillary data.

    Returns
    -------
    iris.cube.Cube
        Cube with added ancillary variables
    """
    try:
        ancillary_var = iris.coords.AncillaryVariable(
            ancillary_cube.core_data(),
            standard_name=ancillary_cube.standard_name,
            units=ancillary_cube.units,
            var_name=ancillary_cube.var_name,
            attributes=ancillary_cube.attributes,
            long_name=ancillary_cube.long_name,
        )
    except AttributeError as err:
        msg = (
            f"Failed to add\n{ancillary_cube}\nas ancillary var to the cube\n{cube}\n"
            "ancillary_cube should be either an iris.cube.Cube or an "
            "iris.coords.AncillaryVariable object."
        )
        raise ValueError(msg) from err
    data_dims = get_data_dims(cube, ancillary_cube)
    if ancillary_cube.has_lazy_data():
        cube_chunks = tuple(cube.lazy_data().chunks[d] for d in data_dims)
        ancillary_var.data = ancillary_cube.lazy_data().rechunk(cube_chunks)
    cube.add_ancillary_variable(ancillary_var, data_dims)
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
