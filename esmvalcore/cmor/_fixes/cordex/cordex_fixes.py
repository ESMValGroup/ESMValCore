"""Fixes that are shared between datasets and drivers."""
import logging
from cf_units import Unit
import cordex as cx
import numpy as np
import iris

from iris.coord_systems import RotatedGeogCS
from esmvalcore.cmor.check import CMORCheck
from esmvalcore.cmor.fix import Fix
from esmvalcore.exceptions import RecipeError
logger = logging.getLogger(__name__)


class MOHCHadREM3GA705(Fix):
    """General fix for MOHC-HadREM3-GA7-05."""

    def fix_metadata(self, cubes):
        """Fix time long_name, and latitude and longitude var_name.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            cube.coord('latitude').var_name = 'lat'
            cube.coord('longitude').var_name = 'lon'
            cube.coord('time').long_name = 'time'

        return cubes


class TimeLongName(Fix):
    """Fixes for time coordinate."""

    def fix_metadata(self, cubes):
        """
        Fix time long_name.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            cube.coord('time').long_name = 'time'

        return cubes


class CLMcomCCLM4817(Fix):
    """Fixes for CLMcom-CCLM4-8-17."""

    def fix_metadata(self, cubes):
        """Fix calendars.

        Set calendar to 'proleptic_gregorian' to avoid
        concatenation issues between historical and
        scenario runs.

        Fix dtype value of coordinates and coordinate bounds.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            if cube.coord('time').units.calendar == 'standard':
                cube.coord('time').units = Unit(
                    'days since 1850-1-1 00:00:00',
                    calendar='proleptic_gregorian'
                )
            for coord in cube.coords():
                if coord.dtype in ['>f8', '>f4']:
                    coord.points = coord.core_points().astype(
                        np.float64, casting='same_kind')
                    if coord.bounds is not None:
                        coord.bounds = coord.core_bounds().astype(
                            np.float64, casting='same_kind')
        return cubes


class AllVars(Fix):
    """General CORDEX grid fix."""

    _grid_diff_msg = ('Maximum difference between original {} '
                      'points and standard {} domain points '
                      'for dataset {} and driver {} is: {}')

    def _check_grid_differences(self, old_coord, new_coord):
        """Check differences between coords."""
        diff = np.max(np.abs(old_coord.points - new_coord.points))
        logger.debug(
            self._grid_diff_msg,
            new_coord.var_name,
            self.extra_facets['domain'],
            self.extra_facets['dataset'],
            self.extra_facets['driver'],
            diff
        )

        if diff > 10e-4:
            raise RecipeError(
                "Differences between the original grid and the "
                f"standarised grid are above 10e-4 {new_coord.units}.",
            )

    def _fix_rotated_coords(self, cube):
        """Fix rotated coordinates."""
        data_domain = self.extra_facets['domain']
        domain = cx.cordex_domain(data_domain, add_vertices=True)
        domain_info = cx.domain_info(data_domain)
        for dim_coord in ['rlat', 'rlon']:
            old_coord = cube.coord(domain[dim_coord].standard_name)
            old_coord_dims = old_coord.cube_dims(cube)
            points = domain[dim_coord].data
            coord_system = iris.coord_systems.RotatedGeogCS(
                grid_north_pole_latitude=domain_info['pollat'],
                grid_north_pole_longitude=domain_info['pollon']
            )
            new_coord = iris.coords.DimCoord(
                points,
                var_name=dim_coord,
                standard_name=domain[dim_coord].standard_name,
                long_name=domain[dim_coord].long_name,
                units=Unit('degrees'),
                coord_system=coord_system,
            )
            self._check_grid_differences(old_coord, new_coord)
            new_coord.guess_bounds()
            cube.remove_coord(old_coord)
            cube.add_dim_coord(new_coord, old_coord_dims)

    def _fix_geographical_coords(self, cube):
        """Fix geographical coordinates."""
        data_domain = self.extra_facets['domain']
        domain = cx.cordex_domain(data_domain, add_vertices=True)
        for aux_coord in ['lat', 'lon']:
            old_coord = cube.coord(domain[aux_coord].standard_name)
            cube.remove_coord(old_coord)
            points = domain[aux_coord].data
            bounds = domain[f'{aux_coord}_vertices'].data
            new_coord = iris.coords.AuxCoord(
                points,
                var_name=aux_coord,
                standard_name=domain[aux_coord].standard_name,
                long_name=domain[aux_coord].long_name,
                units=Unit(domain[aux_coord].units),
                bounds=bounds
            )
            self._check_grid_differences(old_coord, new_coord)
            aux_coord_dims = (
                cube.coord(var_name='rlat').cube_dims(cube) +
                cube.coord(var_name='rlon').cube_dims(cube)
            )
            cube.add_aux_coord(
                new_coord,
                aux_coord_dims)

    def fix_metadata(self, cubes):
        """Fix CORDEX rotated grids.

        Set rotated and geographical coordinates to the
        values given by each domain specification.

        The domain specifications are retrieved from the
        py-cordex package.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            coord_system = cube.coord_system()
            if isinstance(coord_system, RotatedGeogCS):
                self._fix_rotated_coords(cube)
                self._fix_geographical_coords(cube)

        return cubes
