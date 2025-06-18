"""Fixes for GISS-E2-1-G model."""

from esmvalcore.cmor._fixes.common import ClFixHybridPressureCoord
from esmvalcore.cmor._fixes.fix import Fix

Cl = ClFixHybridPressureCoord

Cli = ClFixHybridPressureCoord

Clw = ClFixHybridPressureCoord


class Tos(Fix):
    """Fix tos."""

    def fix_metadata(self, cubes):
        """Correct units.

        The files for ssp126 r1i1p5f1 gn v20200115 have wrong units:
        units in the files are 'degC', but the values are in 'K'.
        """
        for cube in cubes:
            if (
                cube.units == "degC"
                and cube.core_data().ravel()[:1000].max() > 100.0
            ):
                cube.units = "K"
                cube.convert_units(self.vardef.units)
        return cubes
