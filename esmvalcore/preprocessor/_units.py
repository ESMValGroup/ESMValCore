"""Metadata operations on data cubes.

Allows for unit conversions.
"""
import logging

import numpy as np

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
    # We set units to days since ... so time_span will always be in days
    time = cube.coord('time')
    time_span = time.bounds[:, 1] - time.bounds[:, 0]

    # Separate scalar factor from actual units. day-1 units are automatically
    # converted to CONSTANT * s-1, so we must deal with this
    units = cube.units.definition.split(' ')
    if len(units) == 2:
        factor = float(units[0])
    else:
        factor = 1.

    # Separate unit components
    units = units[-1].split('.')
    if 's-1' in units:
        factor *= 24. * 3600.
        units.remove('s-1')
    else:
        raise ValueError(
            f'Units {cube.units} do not contains a supported flux definition')

    time_span = time_span.astype(np.float32) * factor
    dims = list(range(cube.ndim))
    dims.remove(cube.coord_dims(time)[0])
    time_span = np.expand_dims(time_span, dims)
    cube = cube.copy(cube.core_data() * time_span)
    cube.units = ' '.join(units)
    return cube
