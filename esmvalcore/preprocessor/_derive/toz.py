"""Derivation of variable `toz`."""

import cf_units
import iris
from scipy import constants

from ._baseclass import DerivedVariableBase
from ._shared import pressure_level_widths

# Constants
AVOGADRO_CONST = constants.value('Avogadro constant')
AVOGADRO_CONST_UNIT = constants.unit('Avogadro constant')
STANDARD_GRAVITY = constants.value('standard acceleration of gravity')
STANDARD_GRAVITY_UNIT = constants.unit('standard acceleration of gravity')
MW_AIR = 29
MW_AIR_UNIT = cf_units.Unit('g mol^-1')
MW_O3 = 48
MW_O3_UNIT = cf_units.Unit('g mol^-1')
DOBSON_UNIT = cf_units.Unit('2.69e20 m^-2')


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `toz`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        if project == 'CMIP6':
            required = [{'short_name': 'o3'}, {'short_name': 'ps'}]
        else:
            required = [{'short_name': 'tro3'}, {'short_name': 'ps'}]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute total column ozone.

        Note
        ----
        The surface pressure is used as a lower integration bound. A fixed
        upper integration bound of 0 Pa is used.

        """
        tro3_cube = cubes.extract_cube(
            iris.Constraint(name='mole_fraction_of_ozone_in_air'))
        ps_cube = cubes.extract_cube(
            iris.Constraint(name='surface_air_pressure'))

        p_layer_widths = pressure_level_widths(tro3_cube,
                                               ps_cube,
                                               top_limit=0.0)
        toz_cube = (tro3_cube * p_layer_widths / STANDARD_GRAVITY * MW_O3 /
                    MW_AIR)
        toz_cube = toz_cube.collapsed('air_pressure', iris.analysis.SUM)
        toz_cube.units = (tro3_cube.units * p_layer_widths.units /
                          STANDARD_GRAVITY_UNIT * MW_O3_UNIT / MW_AIR_UNIT)

        # Convert from kg m^-2 to Dobson unit (2.69e20 m^-2 )
        toz_cube = toz_cube / MW_O3 * AVOGADRO_CONST
        toz_cube.units = toz_cube.units / MW_O3_UNIT * AVOGADRO_CONST_UNIT
        toz_cube.convert_units(DOBSON_UNIT)
        toz_cube.units = 'DU'

        return toz_cube
