"""Fixes for IPSL-CM6A-LR model."""
import iris
from iris.cube import CubeList
from iris.coords import AuxCoord
from iris.exceptions import ConstraintMismatchError
import cf_units

from ..fix import Fix
import numpy as np

class AllVars(Fix):
    """Fixes for thetao."""

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
            # Rename latitude to grid_latitude
            gridlat = cube.coord('latitude')
            gridlat.var_name = 'rlat'
            gridlat.standard_name='grid_latitude'
            gridlat.units=cf_units.Unit('degrees')
            gridlat.long_name='Grid Latitude'
            print(gridlat.points.shape)
            #print(cube.dimensions)

            # Remove unity length longitude coordinate
            cube = cube.collapsed('longitude', iris.analysis.MEAN)

            #gridlat.points = gridlat.points.mean(axis=1)
            print(gridlat.points.shape)
        #     basin = iris.coords.AuxCoord(
        #         1,
        #         #points=['global_ocean', 'atlantic_arctic_ocean', 'indian_pacific_ocean'],
        #         long_name='ocean basin',
        #         units='1',
        #         var_name='basin',
        #         standard_name='region',
        #         )
        #    # basin.points = ['global_ocean', 'atlantic_arctic_ocean', 'indian_pacific_ocean']
        #     cube.add_aux_coord(basin)
        #     print(cube)
            #assert 0
            basin = cube.coord('Sub-basin mask (1=Global 2=Atlantic 3=Indo-Pacific)')
            #basin.dtype = type('<U21')
            #cube.add_aux_coord('region')
            basin.standard_name='region'
            # basin.units=Unit('1')
            basin.long_name='ocean basin'
            basin.var_name='basin'
            print(basin)
            #basin.points = np.array(['global_ocean', 'atlantic_arctic_ocean', 'indian_pacific_ocean'], dtype='U21')

        return cubes
