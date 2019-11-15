"""Fixes for BCC-ESM1 model."""

from ..fix import Fix


class AllVars(Fix):

    def fix_metadata(self, cubes):
        """
        Fix latitude & longitude coordinates.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            print(cube.coords())
            for coord in cube.coords():
                if coord.var_name=='lat':
                    coord.standard_name='lat'
                if coord.var_name=='lon':
                    coord.standard_name='lon'

            #cube.coord('latitude').var_name = 'lat'
            #cube.coord('longitude').var_name = 'lon'
            #print('fixing: ', cube)
        #assert 0
        return cubes
