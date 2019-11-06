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
        for cube in cubes:
            # Change depth
            depth = cube.coord('Vertical W levels')
            depth.var_name = 'depth'

            # Rename latitude to grid_latitude
            gridlat = cube.coord('latitude')
            gridlat.var_name = 'rlat'
            gridlat.standard_name='grid_latitude'
            gridlat.units=cf_units.Unit('degrees')
            gridlat.long_name='Grid Latitude'

            # Remove unity length longitude coordinate
            cube = iris.util.squeeze(cube)
        #
            basin = iris.coords.AuxCoord(
                ['global_ocean', 'atlantic_arctic_ocean', 'indian_pacific_ocean'],
                long_name='basin',
                units='1',
                var_name='basin',
                standard_name='region',
                )
        #    # basin.points = ['global_ocean', 'atlantic_arctic_ocean', 'indian_pacific_ocean']
            cube = cube.add_aux_coord(basin, data_dims=3)
            #coords = [c.name for c in cube.coords()]
            #print(coords)

            #print('region:', cube.coord('region'))
            # print('basin:', cube.coord('basin'))
            # assert 0

        return cubes
