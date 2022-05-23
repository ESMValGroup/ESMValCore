"""Metadata operations on data cubes.

Allows for unit conversions.
"""
import logging

import numpy as np
import iris

logger = logging.getLogger(__name__)


def convert_units(cube, units):
    """Convert the units of a cube to new ones.

    This converts units of a cube.

    Arguments
    ---------
        cube: iris.cube.Cube
            input cube

        units: str
            new units in udunits form

    Returns
    -------
    iris.cube.Cube
        converted cube.
    """
    cube.convert_units(units)
    return cube


def flux_to_total(cube):
    """Convert flux to aggregated values.

    Flux should have units of X s-1 or any compatible unit.

    Parameters
    ----------
    cube : iris.cube.Cube
        Data cube for the flux

    Returns
    -------
    iris.cube.Cube
        Cube with the aggregated data

    Raises
    ------
    ValueError
        If the units are not supported by the operator
    """
    if 's-1' not in cube.units.format():
        raise ValueError(
            f'Units {cube.units} do not contain a supported flux definition.'
            )
    coord_name = 'time'
    coord = cube.coord(coord_name)
    factor = iris.coords.AuxCoord(
        np.diff(coord.bounds)[..., -1],
        var_name=coord.var_name,
        long_name=coord.long_name,
        units=coord.units,
    )
    result = cube * factor
    unit = result.units.format().split(' ')[-1]
    result.convert_units(unit)
    result.long_name = f"{cube.long_name} * {factor.long_name}"
    return result
