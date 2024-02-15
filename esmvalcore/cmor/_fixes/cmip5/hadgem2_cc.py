"""Fix HadGEM2_CC."""
from ..fix import Fix
from .hadgem2_es import AllVars as BaseAllVars


AllVars = BaseAllVars


class O2(Fix):
    """Fixes for o2."""

    def fix_metadata(self, cubes):
        """
        Fix standard and long names.

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        std = 'mole_concentration_of_dissolved_molecular_oxygen_in_sea_water'
        long_name = 'Dissolved Oxygen Concentration'

        cubes[0].long_name = long_name
        cubes[0].standard_name = std
        return cubes
