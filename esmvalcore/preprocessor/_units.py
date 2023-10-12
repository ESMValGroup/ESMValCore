"""Metadata operations on data cubes.

Allows for unit conversions.
"""
import logging

from cf_units import Unit
import iris
import numpy as np

logger = logging.getLogger(__name__)


# List containing special cases for convert_units. Each list item is another
# list. Each of these sublists defines one special conversion. Each element in
# the sublists is a tuple (standard_name, units). Note: All units for a single
# special case need to be "physically identical", e.g., 1 kg m-2 s-1 "equals" 1
# mm s-1 for precipitation
SPECIAL_CASES = [
    [
        ('precipitation_flux', 'kg m-2 s-1'),
        ('lwe_precipitation_rate', 'mm s-1'),
    ],
]


def _try_special_conversions(cube, units):
    """Try special conversion."""
    for special_case in SPECIAL_CASES:
        for (std_name, special_units) in special_case:
            # Special unit conversion only works if all of the following
            # criteria are met:
            # - the cube's standard_name is one of the supported
            #   standard_names
            # - the cube's units are convertible to the ones defined for
            #   that given standard_name
            # - the desired target units are convertible to the units of
            #   one of the other standard_names in that special case

            # Step 1: find suitable source name and units
            if (cube.standard_name == std_name and
                    cube.units.is_convertible(special_units)):
                for (target_std_name, target_units) in special_case:
                    if target_std_name == std_name:
                        continue

                    # Step 2: find suitable target name and units
                    if Unit(units).is_convertible(target_units):
                        cube.standard_name = target_std_name

                        # In order to avoid two calls to cube.convert_units,
                        # determine the conversion factor between the cube's
                        # units and the source units first and simply add this
                        # factor to the target units (remember that the source
                        # units and the target units should be "physically
                        # identical").
                        factor = cube.units.convert(1.0, special_units)
                        cube.units = f"{factor} {target_units}"
                        cube.convert_units(units)
                        return True

    # If no special case has been detected, return False
    return False


def convert_units(cube, units):
    """Convert the units of a cube to new ones.

    This converts units of a cube.

    Note
    ----
    Allows special unit conversions which transforms one quantity to another
    (physically related) quantity. These quantities are identified via their
    ``standard_name`` and their ``units`` (units convertible to the ones
    defined are also supported). For example, this enables conversions between
    precipitation fluxes measured in ``kg m-2 s-1`` and precipitation rates
    measured in ``mm day-1`` (and vice versa).

    Currently, the following special conversions are supported:

        * ``precipitation_flux`` (``kg m-2 s-1``) --
          ``lwe_precipitation_rate`` (``mm day-1``)

    Names in the list correspond to ``standard_names`` of the input data.
    Conversions are allowed from each quantity to any other quantity given in a
    bullet point. The corresponding target quantity is inferred from the
    desired target units. In addition, any other units convertible to the ones
    given are also supported (e.g., instead of ``mm day-1``, ``m s-1`` is also
    supported).

    Note that for precipitation variables, a water density of ``1000 kg m-3``
    is assumed.

    Arguments
    ---------
    cube: iris.cube.Cube
        Input cube.
    units: str
        New units in udunits form.

    Returns
    -------
    iris.cube.Cube
        converted cube.

    """
    try:
        cube.convert_units(units)
    except ValueError:
        if not _try_special_conversions(cube, units):
            raise

    return cube


def accumulate_coordinate(cube, coordinate):
    """Weight data using the bounds from a given coordinate.

    The resulting cube will then have units given by
    ``cube_units * coordinate_units``.

    Parameters
    ----------
    cube : iris.cube.Cube
        Data cube for the flux

    coordinate: str
        Name of the coordinate that will be used as weights.

    Returns
    -------
    iris.cube.Cube
        Cube with the aggregated data

    Raises
    ------
    ValueError
        If the coordinate is not found in the cube.

    NotImplementedError
        If the coordinate is multidimensional.
    """
    try:
        coord = cube.coord(coordinate)
    except iris.exceptions.CoordinateNotFoundError as err:
        raise ValueError(
            "Requested coordinate %s not found in cube %s",
            coordinate, cube.summary(shorten=True)) from err

    if coord.ndim > 1:
        raise NotImplementedError(
            f'Multidimensional coordinate {coord} not supported.')

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
