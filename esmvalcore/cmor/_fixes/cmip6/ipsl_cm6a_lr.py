"""Fixes for IPSL-CM6A-LR model."""
import iris
from iris.cube import CubeList
from iris.coords import AuxCoord
from iris.exceptions import ConstraintMismatchError
import cf_units

from ..fix import Fix
import numpy as np

class AllVars(Fix):
    """Fixes for All vars."""

    def fix_metadata(self, cubes):
        """
        Fix cell_area coordinate.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        try:
            cell_area = cubes.extract_strict('cell_area')
        except ConstraintMismatchError:
            return cubes

        cell_area = AuxCoord(
            cell_area.data,
            standard_name=cell_area.standard_name,
            long_name=cell_area.long_name,
            var_name=cell_area.var_name,
            units=cell_area.units,
        )
        new_list = CubeList()
        for cube in cubes:
            if cube.name() == 'cell_area':
                continue

            cube.add_aux_coord(cell_area, cube.coord_dims('latitude'))
            cube.coord('latitude').var_name = 'lat'
            cube.coord('longitude').var_name = 'lon'
            new_list.append(cube)
        return CubeList(new_list)


class msftyz(Fix):
    """Fix msftyz."""

    def fix_metadata(self, cubes):
        """
        Problems:
         basin has incorrect long name, var.
         Dimensions are also wrong.

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        new_cubes = []
        for i, cube in enumerate(cubes):
            # Remove unity length longitude coordinate
            # note that squeeze doesn't occur in place. cube is no longer
            # the same cube as in the function argument `cubes`. 
            cube = iris.util.squeeze(cube)

            # Change depth
            depth = cube.coord('Vertical W levels')
            depth.var_name = 'depth'
            depth.standard_name = 'depth'
            depth.long_name = 'depth'

            # Rename latitude to grid_latitude
            gridlat = cube.coord('latitude')
            gridlat.var_name = 'rlat'
            gridlat.standard_name='grid_latitude'
            gridlat.units=cf_units.Unit('degrees')
            gridlat.long_name='Grid Latitude'


            values = np.array(['global_ocean', 'atlantic_arctic_ocean',
                               'indian_pacific_ocean'], dtype='<U21')
            basin_coord = iris.coords.AuxCoord(
                values,
                standard_name=u'region',
                units=cf_units.Unit('no_unit'),
                long_name=u'ocean basin',
                var_name='basin')

            # remove the wrong sub-basin mask DimCoord.
            cube.remove_coord(cube.coord("Sub-basin mask (1=Global 2=Atlantic 3=Indo-Pacific)"))

            # Replace broken coord with correct one.
            cube.add_aux_coord(basin_coord, data_dims=1)

            # squeeze makes a duplicate of the cube, so it is no longer in cubes.
            cubes[i] = cube
        return cubes
