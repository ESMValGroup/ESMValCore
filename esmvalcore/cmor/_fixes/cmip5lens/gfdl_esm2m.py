"""Fixes for the ACCESS1-0 model."""
import logging
import iris
from cf_units import Unit
from iris import coord_categorisation
import numpy as np

from ..fix import Fix

logger = logging.getLogger(__name__)

class AllVars(Fix):
    """Common fixes to all vars."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes wrong calendar 'gregorian' instead of 'proleptic_gregorian'.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes which need to be fixed.

        Returns
        -------
        iris.cube.CubeList

        """
        # import IPython
        # from traitlets.config import get_config
        # c = get_config()
        # c.InteractiveShellEmbed.colors = "Linux"

        # IPython.embed(config=c)

        newcubes = []
        for cube in cubes:
            logger.info("Reinitializing broken time coordinate")
            time_raw = cube.coord('time')
            n_years = len(time_raw.points) // 12
            n_add_mon = len(time_raw.points) % 12

            time_range = cube.attributes['source_file'].split('/')[-1]
            time_range = time_range.split('_')[-1][:-3]
            start_yearmon, end_yearmon = time_range.split('-')
            start_year = int(start_yearmon[:4])

            if time_raw.units.calendar != '365_day':
                logger.error("Wrong calendar type, please addapt")
                raise ValueError

            n_days = (n_years + n_add_mon / 12) * 365 + 50
            times = iris.coords.DimCoord(
                        np.arange(int(n_days), dtype=float)+0.5,
                        var_name='time',
                        standard_name='time',
                        long_name='time',
                        units=Unit(f'days since {start_year}-01-01 00:00:00',
                        calendar=time_raw.units.calendar))
            times.guess_bounds()

            # init a dummy cube to enable coord_categorisation
            dummycube = iris.cube.Cube(np.zeros(int(n_days), np.int),
                                       dim_coords_and_dims=[(times, 0)])
            coord_categorisation.add_year(dummycube, 'time', name='year')
            coord_categorisation.add_month_number(dummycube, 'time',
                                                  name='month')

            dummycube = dummycube.aggregated_by(['year', 'month'],
                                                iris.analysis.MEAN)
            dummycube = dummycube[:(n_years * 12 + n_add_mon)]
            time_new = dummycube.coord('time')

            # change to the new time coordinates
            cube.remove_coord('time')
            cube.add_dim_coord(time_new, 0)

            newcubes.append(cube)

        return newcubes
