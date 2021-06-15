"""Fixes for BCC-CSM2-MR model."""
from ..common import ClFixHybridPressureCoord, OceanFixGrid
from ..fix import Fix

Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord


Tos = OceanFixGrid


Siconc = OceanFixGrid

uo = OceanFixGrid

#class Omon(Fix):
#    """Fixes for ocean variables."""
#
#    def fix_metadata(self, cubes):
#        """Fix ocean depth coordinate.
#
#        Parameters
#        ----------
#        cubes: iris CubeList
#            List of cubes to fix
#
#        Returns
#        -------
#        iris.cube.CubeList
#
#        """
#        cubes = OceanFixGrid.fix_metadata(cubes)
#
#        for cube in cubes:
#            if cube.coords('latitude'):
#                cube.coord('latitude').var_name = 'lat'
#            if cube.coords('longitude'):
#                cube.coord('longitude').var_name = 'lon'
#
#            if cube.coords(axis='Z'):
#                z_coord = cube.coord(axis='Z')
#                if z_coord.var_name == 'olevel':
#                    fix_ocean_depth_coord(cube)
#        return cubes

Sos = OceanFixGrid
