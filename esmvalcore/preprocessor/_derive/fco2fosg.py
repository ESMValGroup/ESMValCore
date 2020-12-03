"""Derivation of variable ``fco2fosg``."""
import iris

from esmvalcore.iris_helpers import var_name_constraint

from ._baseclass import DerivedVariableBase
from .dco2mass import DerivedVariable as DCo2Mass


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable ``fco2fosg``.

    Derive ``Global Carbon Mass Flux into Atmosphere Due to Fossil Fuel
    Emissions of CO2 [kgC s-1]`` by adding carbon fluxes into atmosphere, ocean
    and land and considering emissions from land use changes.

    Fossil fuel emissions (E) and land use emissions (L) partly stay in the
    atmosphere (dC_A), partly are absorbed by the terrestrial biosphere (dC_L)
    and partly are absorbed by the ocean reservoir (dC_O):

    E + L = dC_A + dC_L + dC_O

    This can be rewritten as:

    E = dC_A + dC_L - L + dC_O

    dC_A is given by the derived variable ``dco2mass``, dC_L - L is given by
    the CMOR variable ``nbp`` (net biome productivity) and dC_O is given by the
    CMOR variable ``fgco2``. Thus,

    fco2fosg = dco2mass + nbp + fgco2.

    """

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {'short_name': 'co2mass', 'mip': 'Amon'},
            {'short_name': 'nbp', 'mip': 'Lmon'},
            {'short_name': 'fgco2', 'mip': 'Omon'},
            {'short_name': 'areacella', 'mip': 'fx'},
            {'short_name': 'areacello', 'mip': 'Ofx'},
            {'short_name': 'sftlf', 'mip': 'fx'},
            {'short_name': 'sftof', 'mip': 'Ofx'},
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute carbon mass flux due to fossil fuel emissions of CO2."""
        dco2mass_cube = DCo2Mass().calculate(cubes)
        nbp_cube = cubes.extract_strict(var_name_constraint('nbp'))
        fgco2_cube = cubes.extract_strict(var_name_constraint('fgco2'))
        areacella_cube = cubes.extract_strict(var_name_constraint('areacella'))
        areacello_cube = cubes.extract_strict(var_name_constraint('areacello'))
        sftlf_cube = cubes.extract_strict(var_name_constraint('sftlf'))
        sftof_cube = cubes.extract_strict(var_name_constraint('sftof'))

        # Concert kgCO2 into kgC for dco2mass
        dco2mass_cube.data = dco2mass_cube.core_data() * 12.011 / 44.01

        # Spatial aggregation of nbp
        nbp_cube.data = nbp_cube.core_data() * sftlf_cube.core_data()
        nbp_cube = nbp_cube.collapsed(['latitude', 'longitude'],
                                      iris.analysis.SUM,
                                      weights=areacella_cube.core_data())

        # Spatial aggregation of fgco2
        fgco2_cube.data = fgco2_cube.core_data() * sftof_cube.core_data()
        fgco2_cube = fgco2_cube.collapsed(['latitude', 'longitude'],
                                          iris.analysis.SUM,
                                          weights=areacello_cube.core_data())

        # Calculate anthropogenic fossil fuel emissions
        fco2fosg_cube = dco2mass_cube.copy()
        fco2fosg_cube.data = (dco2mass_cube.core_data() +
                              nbp_cube.core_data() + fgco2_cube.core_data())
        fco2fosg_cube.units = 'kg s-1'
        print(fco2fosg_cube)
        return fco2fosg_cube
