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
    Allows some special unit conversions which are usually not allowed by
    :mod:`cf_units` for certain variables (based on matching ``var_name`` or
    patterns in ``standard_name`` or ``long_name``, see brackets below). Note
    that any other units convertible to the ones given below also work, e.g.,
    for precipitation ``[kg m-2 yr-1]`` --> ``[m s-1]`` would also work:

        - Variable ``pr`` (``precipitation``): ``[kg m-2 s-1]`` --> ``[mm
          day-1]``

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
    # Dictionary keys: (expected var_name, expected pattern which needs to
    #                   appear in standard_name or long_name)
    # Dictionary values: (special source units, special target units) --> both
    #                    need to be "identical", e.g., 1 kg m-2 s-1 "equals" 1
    #                    mm s-1 for precipitation
    special_cases = {
        ('pr', 'precipitation'): ('kg m-2 s-1', 'mm s-1'),
    }
    try:
        cube.convert_units(units)
    except ValueError:
        var_name = '' if cube.var_name is None else cube.var_name
        std_name = '' if cube.standard_name is None else cube.standard_name
        long_name = '' if cube.long_name is None else cube.long_name
        for (special_names, special_units) in special_cases.items():
            # Special unit conversion only works if all of the following
            # criteria are met:
            # - the units in the cubes are convertible to the special source
            #   units
            # - the target units given by the user are convertible to the
            #   special target units
            # - the cube has the correct names, i.e., either the cube's
            #   var_name matches the expected var_name or the expected pattern
            #   appears in the cube's standard_name or long_name
            is_special_case = (
                cube.units.is_convertible(special_units[0]) and
                Unit(units).is_convertible(special_units[1]) and
                any([
                    var_name == special_names[0],
                    special_names[1] in std_name.lower(),
                    special_names[1] in long_name.lower(),
                ])
            )
            if is_special_case:
                cube.convert_units(special_units[0])
                cube.units = special_units[1]
                cube.convert_units(units)
                break

        # If no special case has been detected, raise the original error
        else:
            raise

    return cube
