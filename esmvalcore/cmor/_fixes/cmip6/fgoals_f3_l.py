"""Fixes for CMIP6 FGOALS-f3-L."""
import iris
import numpy as np
from iris import coord_categorisation

from esmvalcore.preprocessor import extract_time

from ..fix import Fix


class AllVars(Fix):
    """Fixes for all vars."""
    def fix_metadata(self, cubes):
        """Fix parent time units.

        FGOALS-f3-L Amon data may have a bad time bounds spanning 20 days.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        def _get_correct_time_coordinate(wrong_time):
            # get daily time points
            time_points = np.arange(int(wrong_time.points[0] - 17),
                                    int(wrong_time.points[-1] + 18),
                                    1,
                                    dtype=float) + 0.5
            time = iris.coords.DimCoord(time_points,
                                        var_name=wrong_time.var_name,
                                        standard_name=wrong_time.standard_name,
                                        long_name=wrong_time.long_name,
                                        units=wrong_time.units)
            time.guess_bounds()

            # init a dummy cube to enable coord_categorisation
            dummy_cube = iris.cube.Cube(np.zeros(len(time_points), np.int),
                                        dim_coords_and_dims=[(time, 0)])
            coord_categorisation.add_year(dummy_cube, 'time', name='year')
            coord_categorisation.add_month_number(dummy_cube,
                                                  'time',
                                                  name='month')

            dummy_cube = dummy_cube.aggregated_by(['year', 'month'],
                                                  iris.analysis.MEAN)

            # get start and end of the wrong time
            dates = time.units.num2date(wrong_time.points)
            start_year = dates[0].year
            start_month = dates[0].month
            end_year = dates[-1].year
            end_month = dates[-1].month

            dummy_cube = extract_time(dummy_cube, start_year, start_month, 1,
                                      end_year, end_month, 31)

            return dummy_cube.coord('time')

        for cube in cubes:
            if cube.attributes['table_id'] == 'Amon':
                # check if dim is present
                if 'time' in [coord.standard_name for coord in cube.coords()]:
                    time = cube.coord('time')
                    mismatch = ~(time.bounds[1:, 0] == time.bounds[:-1, 1])
                    if any(mismatch):
                        correct_time = _get_correct_time_coordinate(time)
                        if all(time.points == correct_time.points):
                            time.bounds = correct_time.bounds
                        else:
                            raise ValueError("Wrong time bounds")

        return cubes
