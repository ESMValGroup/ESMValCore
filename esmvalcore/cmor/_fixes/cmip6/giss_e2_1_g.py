"""Fixes for GISS-E2-1-G model."""
import dask.array as da

from ..common import ClFixHybridPressureCoord
from ..fix import Fix

Cl = ClFixHybridPressureCoord

Cli = ClFixHybridPressureCoord

Clw = ClFixHybridPressureCoord


class Tos(Fix):

    def fix_metadata(self, cubes):
        for cube in cubes:
            if (cube.units == 'degC'
                    and da.any(cube.core_data().ravel()[:1000] > 100.)):
                cube.units = 'K'
                cube.convert_units(self.vardef.units)
        return cubes
