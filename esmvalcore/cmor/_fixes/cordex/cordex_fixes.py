"""Fixes that are shared between datasets and drivers."""
import logging
from functools import lru_cache

import cordex as cx
import iris
import numpy as np
from cf_units import Unit
from iris.coord_systems import LambertConformal, RotatedGeogCS

from esmvalcore.cmor.fix import Fix
from esmvalcore.exceptions import RecipeError

logger = logging.getLogger(__name__)


@lru_cache
def _get_domain(data_domain):
    domain = cx.cordex_domain(data_domain, add_vertices=True)
    return domain


@lru_cache
def _get_domain_info(data_domain):
    domain_info = cx.domain_info(data_domain)
    return domain_info


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
        """Fix time long_name.

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
            time_unit = cube.coord('time').units
            if time_unit.calendar == 'standard':
                new_unit = time_unit.change_calendar('proleptic_gregorian')
                cube.coord('time').units = new_unit
            for coord in cube.coords():
                if coord.dtype in ['>f8', '>f4']:
                    coord.points = coord.core_points().astype(
                        np.float64, casting='same_kind')
                    if coord.has_bounds():
                        coord.bounds = coord.core_bounds().astype(
                            np.float64, casting='same_kind')
        return cubes


class AllVars(Fix):
    """General CORDEX grid fix."""

    def _check_grid_differences(self, old_coord, new_coord):
        """Check differences between coords."""
        diff = np.max(np.abs(old_coord.points - new_coord.points))
        logger.debug(
            "Maximum difference between original %s"
            "points and standard %s domain points  "
            "for dataset %s and driver %s is: %s.", new_coord.var_name,
            self.extra_facets['domain'], self.extra_facets['dataset'],
            self.extra_facets['driver'], str(diff))

        if diff > 10e-4:
            raise RecipeError(
                "Differences between the original grid and the "
                f"standardised grid are above 10e-4 {new_coord.units}.")

    def _fix_rotated_coords(self, cube, domain, domain_info):
        """Fix rotated coordinates."""
        for dim_coord in ['rlat', 'rlon']:
            old_coord = cube.coord(domain[dim_coord].standard_name)
            old_coord_dims = old_coord.cube_dims(cube)
            points = domain[dim_coord].data
            coord_system = iris.coord_systems.RotatedGeogCS(
                grid_north_pole_latitude=domain_info['pollat'],
                grid_north_pole_longitude=domain_info['pollon'])
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

    def _fix_geographical_coords(self, cube, domain):
        """Fix geographical coordinates."""
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
                bounds=bounds,
            )
            if aux_coord == 'lon' and new_coord.points.min() < 0.:
                lon_inds = (new_coord.points < 0.) & (old_coord.points > 0.)
                old_coord.points[lon_inds] = old_coord.points[lon_inds] - 360.

            self._check_grid_differences(old_coord, new_coord)
            aux_coord_dims = (cube.coord(var_name='rlat').cube_dims(cube) +
                              cube.coord(var_name='rlon').cube_dims(cube))
            cube.add_aux_coord(new_coord, aux_coord_dims)

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
        data_domain = self.extra_facets['domain']
        domain = _get_domain(data_domain)
        domain_info = _get_domain_info(data_domain)
        for cube in cubes:
            coord_system = cube.coord_system()
            if isinstance(coord_system, RotatedGeogCS):
                self._fix_rotated_coords(cube, domain, domain_info)
                self._fix_geographical_coords(cube, domain)
            elif isinstance(coord_system, LambertConformal):
                logger.warning(
                    "Support for CORDEX datasets in a Lambert Conformal "
                    "coordinate system is ongoing. Certain preprocessor "
                    "functions may fail.")
            else:
                raise RecipeError(
                    f"Coordinate system {coord_system.grid_mapping_name} "
                    "not supported in CORDEX datasets. Must be "
                    "rotated_latitude_longitude or lambert_conformal_conic.")

        return cubes
