"""Weighting preprocessor module."""

import logging

import iris

logger = logging.getLogger(__name__)


def _get_land_fraction(cube):
    """Extract land fraction as :mod:`dask.array`."""
    fx_cube = None
    land_fraction = None
    errors = []
    try:
        fx_cube = cube.ancillary_variable('land_area_fraction')
    except iris.exceptions.AncillaryVariableNotFoundError:
        try:
            fx_cube = cube.ancillary_variable('sea_area_fraction')
        except iris.exceptions.AncillaryVariableNotFoundError:
            errors.append(
                'Ancillary variables land/sea area fraction '
                'not found in cube. Check fx_file availability.')
            return (land_fraction, errors)

    if fx_cube.var_name == 'sftlf':
        land_fraction = fx_cube.core_data() / 100.0
    if fx_cube.var_name == 'sftof':
        land_fraction = 1.0 - fx_cube.core_data() / 100.0

    return (land_fraction, errors)


def weighting_landsea_fraction(cube, area_type):
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
        Land/sea fraction variables ``sftlf`` or ``sftof`` not found.

    """
    if area_type not in ('land', 'sea'):
        raise TypeError(
            f"Expected 'land' or 'sea' for area_type, got '{area_type}'")
    (land_fraction, errors) = _get_land_fraction(cube)
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
