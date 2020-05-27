"""Fixes for BCC-CSM2-MR model."""
from ..cmip5.bcc_csm1_1 import Tos as BaseTos
from ..common import ClFixHybridPressureCoord


Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord


class Tos(BaseTos):
    """Fixes for tos."""

    def fix_metadata(self, cubes):
        """Rename ``var_name`` of 1D-``latitude`` and 1D-``longitude``.
        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.
        Returns
        -------
        iris.cube.CubeList
        """
        cube = self.get_cube_from_list(cubes)
        lat_coord = cube.coord('latitude', dimensions=(1, ))
        lon_coord = cube.coord('longitude', dimensions=(2, ))
        lat_coord.standard_name = None
        lat_coord.long_name = 'grid_latitude'
        lat_coord.var_name = 'i'
        lat_coord.units = '1'
        lon_coord.standard_name = None
        lon_coord.long_name = 'grid_longitude'
        lon_coord.var_name = 'j'
        lon_coord.units = '1'
        lon_coord.circular = False
        return cubes


class Siconc(BaseTos):
    """Fixes for siconc."""

    def fix_metadata(self, cubes):
        """Rename ``var_name`` of 1D-``latitude`` and 1D-``longitude``.
        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.
        Returns
        -------
        iris.cube.CubeList
        """
        cube = self.get_cube_from_list(cubes)
        lat_coord = cube.coord('latitude', dimensions=(1, ))
        lon_coord = cube.coord('longitude', dimensions=(2, ))
        lat_coord.standard_name = None
        lat_coord.long_name = 'grid_latitude'
        lat_coord.var_name = 'i'
        lat_coord.units = '1'
        lon_coord.standard_name = None
        lon_coord.long_name = 'grid_longitude'
        lon_coord.var_name = 'j'
        lon_coord.units = '1'
        lon_coord.circular = False
        return cubes
