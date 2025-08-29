"""Preprocessor functions for ancillary variables and cell measures."""

import logging
from collections.abc import Callable, Iterable
from typing import Literal

import iris.coords
import numpy as np
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


def add_ancillary_variable(  # noqa: C901, PLR0912, PLR0915
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
            f"Failed to add {ancillary_cube.var_name} as ancillary var to the cube\n{cube}\n"
            "ancillary_cube should be either an iris.cube.Cube or an "
            "iris.coords.AncillaryVariable object."
        )
        raise ValueError(msg) from err
    # Match the coordinates of the ancillary cube to coordinates and
    # dimensions in the input cube before adding the ancillary variable.
    data_dims: list[int | None] = []
    if isinstance(ancillary_cube, iris.coords.AncillaryVariable):
        start_dim = cube.ndim - len(ancillary_var.shape)
        data_dims = list(range(start_dim, cube.ndim))
    else:
        data_dims = [None] * ancillary_cube.ndim
        for coord in ancillary_cube.coords():
            logger.debug("Matching coordinate...\n%s", coord)
            try:
                for ancillary_dim, cube_dim in zip(
                    ancillary_cube.coord_dims(coord),
                    cube.coord_dims(coord),
                    strict=False,
                ):
                    data_dims[ancillary_dim] = cube_dim
                logger.debug("Matched coordinate.")
            except iris.exceptions.CoordinateNotFoundError:
                logger.debug(
                    "No exact match from ancillary cube in cube coords.",
                )
                # Try with casting coordinate
                try:
                    # Cast coordinates to float32 and back to float64
                    if coord.dtype == np.float64:
                        logger.debug("Trying to cast coordinates...")
                        coord.points = (
                            coord.core_points()
                            .astype(np.float32)
                            .astype(np.float64)
                        )
                    for ancillary_dim, cube_dim in zip(
                        ancillary_cube.coord_dims(coord),
                        cube.coord_dims(coord),
                        strict=False,
                    ):
                        data_dims[ancillary_dim] = cube_dim
                    logger.debug("Matched casted coordinate.")
                except iris.exceptions.CoordinateNotFoundError:
                    ancillary_dims = ancillary_cube.coord_dims(coord)
                    # Look for numerically matching coordinate if previous attempt failed
                    logger.debug(
                        "Trying to find a close match in the cube coords...",
                    )
                    cube_dims = []
                    for cube_coord in cube.coords():
                        if (
                            cube_coord.dtype == np.float64
                            and coord.standard_name == cube_coord.standard_name
                            and coord.shape == cube_coord.shape
                        ):
                            cube_coord.points = (
                                cube_coord.core_points()
                                .astype(np.float32)
                                .astype(np.float64)
                            )
                            acceptable_relative_difference = 0.1
                            if cube_coord.points.ndim > 1:
                                dy = np.abs(
                                    np.diff(
                                        cube_coord.core_points(),
                                        axis=0,
                                    ),
                                ).flatten()
                                dx = np.abs(
                                    np.diff(
                                        cube_coord.core_points(),
                                        axis=1,
                                    ),
                                ).flatten()
                                diffs = np.concatenate((dy, dx))
                                diffs = diffs[diffs > 0]
                                min_diff = np.min(diffs)
                            else:
                                min_diff = np.abs(
                                    np.diff(
                                        cube_coord.core_points(),
                                    ),
                                ).min()
                            atol = acceptable_relative_difference * min_diff
                            if np.allclose(
                                coord.points,
                                cube_coord.points,
                                rtol=0,
                                atol=atol,
                            ):
                                cube_dims = cube.coord_dims(cube_coord)
                                logger.debug("Found a close coordinate.")
                    for ancillary_dim, cube_dim in zip(
                        ancillary_dims,
                        cube_dims,
                        strict=False,
                    ):
                        data_dims[ancillary_dim] = cube_dim
        if None in data_dims:
            none_dims = ", ".join(
                str(i) for i, d in enumerate(data_dims) if d is None
            )
            msg = (
                f"Failed to add {ancillary_cube.var_name} as ancillary var "
                f"to the cube\n{cube}\n"
                f"No coordinate associated with ancillary cube dimensions "
                f"{none_dims}"
            )
            logger.info(msg)
            raise iris.exceptions.CoordinateNotFoundError
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
