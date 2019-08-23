from ..fix import Fix

class AllVars(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Fix parent time units

        Parameters
        ----------
        cube : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

        """
        parent_units = 'parent_time_units'
        for cube in cubes:
            try:
                if (cube.attributes[parent_units] ==
                    'days since 1850-01-01-00-00-00'):
                    cube.attributes[parent_units] = 'days since 1850-01-01'
            except AttributeError:
                pass
        return cubes
