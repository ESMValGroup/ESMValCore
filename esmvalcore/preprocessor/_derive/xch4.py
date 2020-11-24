"""Derivation of variable `xch4`."""

import iris
from iris import Constraint
import numpy as np
from scipy import constants

from ._baseclass import DerivedVariableBase
from ._shared import pressure_level_widths

# Constants
FAIR_COR = 3.0825958e-6
MW_AIR = 28.9644e-3
AVOGADRO_CONST = constants.value('Avogadro constant')


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `xch4`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {'short_name': 'ch4'},
            {'short_name': 'hus'},
            {'short_name': 'zg'},
            {'short_name': 'ps'},
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Calculate the average-column atmospheric CH4 [1e-9].

        The calculation follows the method described in the obs4MIPs
        technical note "Merged SCIMACHY/ENVISAT and TANSO-FTS/GOSAT
        atmospheric column-average dry-air mole fraction of CH4 (XCH4)"
        by M. Buchwitz and M. Reuter, available at:
        http://esgf-data1.ceda.ac.uk/thredds/fileServer/esg_obs4mips/
              esacci/ghg/data/obs4mips/crdp_3/CH4/v100/
              TechNote_SCIAGOSAT_L3_CRDP3_001_XCH4_FINAL.pdf
        """
        ch4_cube = cubes.extract_strict(
            Constraint(name='mole_fraction_of_methane_in_air'))
        hus_cube = cubes.extract_strict(Constraint(name='specific_humidity'))
        zg_cube = cubes.extract_strict(Constraint(name='geopotential_height'))
        ps_cube = cubes.extract_strict(Constraint(name='surface_air_pressure'))

        # level thickness (note: Buchwitz & Reuter use hPa but we use Pa;
        # in fact, this does not matter as units cancel out when calculating
        # xch4)
        p_layer_widths = pressure_level_widths(ch4_cube, ps_cube,
                                               top_limit=0.0)

        # latitudes (1-dim array)
        lat = ch4_cube.coord('latitude').points

        # gravitational acceleration g_0 on the geoid approximated by the
        # international gravity formula depending only on the latitude
        g_0 = np.array(lat)
        g_0 = 9.780327 * (1. + 0.0053024 * (np.sin(lat / 180. * np.pi))**2
                          - 0.0000058 * (np.sin(2. * lat / 180. * np.pi))**2)

        # approximation of the gravitational acceleration including the
        # free air correction
        # note: the formula for g given in Buchwitz & Reuter contains a typo
        #       and should read: g_0**2 - 2*f*zg (i.e. minus instead of +)
        g_4d_array = iris.util.broadcast_to_shape(g_0**2, zg_cube.shape, [2])
        g_4d_array = np.sqrt(g_4d_array.data - 2. * FAIR_COR * zg_cube.data)

        # number of dry air particles (air molecules excluding water vapor)
        # within each layer
        n_dry = ((hus_cube * -1.0 + 1.0) * AVOGADRO_CONST *
                 p_layer_widths.data / (MW_AIR * g_4d_array))

        # number of CH4 molecules per layer
        ch4_cube = ch4_cube * n_dry

        # column-average CH4
        xch4_cube = (
            ch4_cube.collapsed('air_pressure', iris.analysis.SUM) /
            n_dry.collapsed('air_pressure', iris.analysis.SUM))
        xch4_cube.units = ch4_cube.units
        xch4_cube.convert_units("1")

        return xch4_cube
