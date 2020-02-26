"""Weighting preprocessor module."""

import logging

import iris

logger = logging.getLogger(__name__)


def _get_land_fraction(cube, fx_files):
    """Extract land fraction as :mod:`dask.array`."""
    land_fraction = None
    errors = []
    if not fx_files:
        errors.append("No fx files given.")
        return (land_fraction, errors)
    for (fx_var, fx_path) in fx_files.items():
        if not fx_path:
            errors.append(f"File for '{fx_var}' not found.")
            continue
        fx_cube = iris.load_cube(fx_path)
        if not _shape_is_broadcastable(fx_cube.shape, cube.shape):
            errors.append(
                f"Cube '{fx_var}' with shape {fx_cube.shape} not "
                f"broadcastable to cube '{cube.var_name}' with shape "
                f"{cube.shape}.")
            continue
        if fx_var == 'sftlf':
            land_fraction = fx_cube.core_data() / 100.0
            break
        if fx_var == 'sftof':
            land_fraction = 1.0 - fx_cube.core_data() / 100.0
            break
        errors.append(
            f"Cannot calculate land fraction from '{fx_var}', expected "
            f"'sftlf' or 'sftof'.")
    return (land_fraction, errors)


def _shape_is_broadcastable(shape_1, shape_2):
    """Check if two :mod:`numpy.array' shapes are broadcastable."""
    return all((m == n) or (m == 1) or (n == 1)
               for (m, n) in zip(shape_1[::-1], shape_2[::-1]))


def weighting_landsea_fraction(cube, fx_files, area_type):
    """Weight fields using land or sea fraction.

    This preprocessor function weights a field with its corresponding land or
    sea area fraction (value between 0 and 1). The application of this is
    important for most carbon cycle variables (and other land-surface outputs),
    which are e.g. reported in units of `kgC m-2`. This actually refers to 'per
    square meter of land/sea' and NOT 'per square meter of gridbox'. So in
    order to integrate these globally or regionally one has to both area-weight
    the quantity but also weight by the land/sea fraction.

    Parameters
    ----------
    cube : iris.cube.Cube
        Data cube to be weighted.
    fx_files : dict
        Dictionary holding ``var_name`` (keys) and full paths (values) to the
        fx files as ``str`` or empty ``list`` (if not available).
    area_type : str
        Use land (``'land'``) or sea (``'sea'``) fraction for weighting.

    Returns
    -------
    iris.cube.Cube
        Land/sea fraction weighted cube.

    Raises
    ------
    TypeError
        ``area_type`` is not ``'land'`` or ``'sea'``.
    ValueError
        Land/sea fraction variables ``sftlf`` or ``sftof`` not found or shape
        of them is not broadcastable to ``cube``.

    """
    if area_type not in ('land', 'sea'):
        raise TypeError(
            f"Expected 'land' or 'sea' for area_type, got '{area_type}'")
    (land_fraction, errors) = _get_land_fraction(cube, fx_files)
    if land_fraction is None:
        raise ValueError(
            f"Weighting of '{cube.var_name}' with '{area_type}' fraction "
            f"failed because of the following errors: {' '.join(errors)}")
    core_data = cube.core_data()
    if area_type == 'land':
        cube.data = core_data * land_fraction
    elif area_type == 'sea':
        cube.data = core_data * (1.0 - land_fraction)
    return cube
