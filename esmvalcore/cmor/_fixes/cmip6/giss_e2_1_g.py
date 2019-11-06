"""Fixes for GISS-E2-1-G."""
from ..fix import Fix
import cf_units



class msftyz(Fix):
    """Fix msftyz."""

    def fix_metadata(self, cubes):
        """
        Fix standard and long name.

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        print("Note that I had to link these data by hand.")
        print("This means that msftyz_Omon_GISS-E2-1-G_historical_r1i1p1f1_gn_185001-190012.nc")
        print("is just a point to the msftmyz_Omon_GISS-E2-1-G_historical_r1i1p1f1_gn_185001-190012.nc file.")

        for cube in cubes:
            # Fix parts where the wrong variable was provided
            cube.standard_name = "ocean_y_overturning_mass_streamfunction"
            cube.long_name = "Ocean Y Overturning Mass Streamfunction"
            cube.variable_id = 'msftyz'

            # Fix part where the wrong dimension was provided
            gridlat = cube.coord('latitude')
            gridlat.var_name = 'rlat'
            gridlat.standard_name='grid_latitude'
            gridlat.units=cf_units.Unit('degrees')
            gridlat.long_name='Grid Latitude'

            # Fix part where the wrong basin was provided.
            basin = cube.coord('region')
            basin.var_name = 'basin'
        return cubes
