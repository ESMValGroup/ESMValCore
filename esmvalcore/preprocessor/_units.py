"""
Metadata operations on data cubes.

Allows for unit conversions.
"""
import logging

from cf_units import Unit

logger = logging.getLogger(__name__)


def convert_units(cube, units):
    """
    Convert the units of a cube to new ones.

    This converts units of a cube.

    Note
    ----
    Allows special unit conversions which are usually not allowed by
    :mod:`cf_units` for certain variables (based on matching
    ``standard_name``). This also adapts the ``standard_name`` of the data so
    that is consistent with the CF conventions (names in brackets below). Note
    that any other units convertible to the ones given below also work, e.g.,
    for ``precipitation_flux``, ``[kg m-2 yr-1]`` --> ``[m s-1]`` would also
    work.

        - ``precipitation_flux`` (--> ``lwe_precipitation_rate``): ``[kg m-2
          s-1]`` --> ``[mm day-1]``

    For precipitation variables, a water density of 1000 kg m-3 is assumed.

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
    # Dictionary containing special cases
    # Dictionary keys: (expected standard_name, target standard_name)
    # Dictionary values: (special source units, special target units) --> both
    #                    need to be "identical", e.g., 1 kg m-2 s-1 "equals" 1
    #                    mm s-1 for precipitation
    special_cases = {
        ('precipitation_flux', 'lwe_precipitation_rate'): ('kg m-2 s-1',
                                                           'mm s-1'),
    }
    try:
        cube.convert_units(units)
    except ValueError:
        for (special_names, special_units) in special_cases.items():
            # Special unit conversion only works if all of the following
            # criteria are met:
            # - the units in the cube are convertible to the special source
            #   units
            # - the target units desired by the user are convertible to the
            #   special target units
            # - the cube has the correct standard_name
            is_special_case = all([
                cube.units.is_convertible(special_units[0]),
                Unit(units).is_convertible(special_units[1]),
                cube.standard_name == special_names[0],
            ])
            if is_special_case:
                cube.standard_name = special_names[1]
                cube.convert_units(special_units[0])
                cube.units = special_units[1]
                cube.convert_units(units)
                break

        # If no special case has been detected, raise the original error
        else:
            raise

    return cube
