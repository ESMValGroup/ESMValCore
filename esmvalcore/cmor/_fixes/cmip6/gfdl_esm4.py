"""Fixes for GFDL-ESM4 model."""
from ..fix import Fix
import iris


class Siconc(Fix):
    """Fixes for siconc."""

    def fix_metadata(self, cubelist):
        """
        Fix missing type.

        Parameters
        ----------
        cubelist: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        typesi = iris.coords.AuxCoord(
            'siconc',
            standard_name='area_type',
            long_name='Sea Ice area type',
            var_name='type',
            units='1',
            bounds=None)
        for cube in cubelist:
            cube.add_aux_coord(typesi)
        return cubelist
