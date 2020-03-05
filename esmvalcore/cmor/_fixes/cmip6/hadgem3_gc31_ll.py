"""Fixes for CMIP6 HadGEM-GC31-LL."""
from ..fix import Fix
from .ukesm1_0_ll import Cl as BaseCl


class AllVars(Fix):
    """Fixes for all vars."""

    def fix_metadata(self, cubes):
        """
        Fix parent time units.

        Parameters
        ----------
        cube : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

        """
        parent_units = 'parent_time_units'
        bad_value = 'days since 1850-01-01-00-00-00'
        for cube in cubes:
            try:
                if cube.attributes[parent_units] == bad_value:
                    cube.attributes[parent_units] = 'days since 1850-01-01'
            except AttributeError:
                pass
        return cubes


class Cl(BaseCl):
    """Fixes for ``cl``."""


class Clw(Cl):
    """Fixes for ``clw (same as for cl)``."""


class Cli(Cl):
    """Fixes for ``cli (same as for cl)``."""
