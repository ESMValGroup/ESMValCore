"""Fixes that are shared between datasets and drivers."""
import json
import logging
from functools import lru_cache
from pathlib import Path

import cordex as cx
import iris
import numpy as np
from cf_units import Unit
from iris.coord_systems import GeogCS, LambertConformal, RotatedGeogCS
from iris.fileformats.pp import EARTH_RADIUS
from pyproj import Transformer

from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import CMOR_TABLES
from esmvalcore.exceptions import RecipeError
from esmvalcore.preprocessor import extract_region

logger = logging.getLogger(__name__)
GEOG_SYSTEM = GeogCS(EARTH_RADIUS).as_cartopy_crs()


@lru_cache
def _get_domain(data_domain):
    domain = cx.cordex_domain(data_domain, add_vertices=True)
    return domain


@lru_cache
def _get_domain_info(data_domain):
    domain_info = cx.domain_info(data_domain)
    return domain_info


@lru_cache
def _get_domain_boundaries(data_domain):
    json_file = Path(__file__).parent / 'nrp_rcm_domain_boundaries.json'
    with open(json_file, mode="r", encoding='utf-8') as file:
        domain_bndrs = json.loads(file.read())
    return domain_bndrs[data_domain]


def _transform_points(x_data, y_data, crs_from, crs_to):
    """Transform points between one pyproj.crs.CRS to another."""
    res = np.empty(x_data.shape + (2, ))
    res[..., 0], res[..., 1] = Transformer.from_crs(
        crs_from=crs_from,
        crs_to=crs_to,
        always_xy=True,
    ).transform(x_data, y_data, errcheck=True)
    return res


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
                    if coord.bounds is not None:
                        coord.bounds = coord.core_bounds().astype(
                            np.float64, casting='same_kind')
        return cubes


class LambertGrid(Fix):
    """General grid fix for CORDEX datasets in a LCC grid."""

    def _fix_projection_coords(self, cube):
        """Use geographical coords to fix projection coords."""
        cube_system = cube.coord_system().as_cartopy_crs()
        lon_pts = cube.coord("longitude").points
        lat_pts = cube.coord("latitude").points

        # Most datasets have the projection coordinates shifted.
        # Transforming them from geographical coordinates that fit
        # within the standard domain gives quite a good approximation
        # of the projection coordinate values for some datasets.
        projection_points = _transform_points(x_data=lon_pts,
                                              y_data=lat_pts,
                                              crs_from=GEOG_SYSTEM,
                                              crs_to=cube_system)

        for index, dim_coord in enumerate(['x', 'y']):
            coord_info = CMOR_TABLES['CORDEX'].coords[dim_coord]
            old_coord = cube.coord(coord_info.standard_name)
            old_coord_dims = old_coord.cube_dims(cube)
            points = np.average(projection_points[:, :, index], axis=index)
            new_coord = iris.coords.DimCoord(
                points,
                var_name=dim_coord,
                standard_name=coord_info.standard_name,
                long_name=coord_info.long_name,
                units=Unit(coord_info.units),
                coord_system=cube.coord_system(),
            )
            new_coord.guess_bounds()
            cube.remove_coord(old_coord)
            cube.add_dim_coord(new_coord, old_coord_dims)

    def _fix_geographical_coords_lambert(self, cube):
        """Fix geographical coordinate metadata and guess bounds."""
        bounds_x = cube.coord('projection_x_coordinate').bounds
        bounds_y = cube.coord('projection_y_coordinate').bounds

        # Transform projection bounds to geographical bounds
        geo_bounds = _transform_points(
            *np.meshgrid(bounds_x, bounds_y),
            crs_from=cube.coord_system().as_cartopy_crs(),
            crs_to=GEOG_SYSTEM,
        )

        for index, aux_coord in enumerate(['longitude', 'latitude']):
            coord_info = CMOR_TABLES['CORDEX'].coords[aux_coord]
            coord = cube.coord(coord_info.standard_name)
            coord.long_name = coord_info.long_name
            coord.units = Unit(coord_info.units)
            # Arrange vertices
            vertices = geo_bounds[..., index]
            bounds = np.array([
                vertices[0::2, 0::2], vertices[0::2, 1::2],
                vertices[1::2, 1::2], vertices[1::2, 0::2]
            ])
            bounds = np.moveaxis(bounds, 0, -1)
            coord.bounds = bounds

    def fix_metadata(self, cubes):
        """Fix CORDEX lambert conformal conic grids.

        Set projection coordinates taking into account the
        geographical coordinate values and guess bounds
        for both sets of coordinates.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        for cube in cubes:
            self._fix_projection_coords(cube)
            self._fix_geographical_coords_lambert(cube)
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

    def _fix_geographical_coords_rotated(self, cube, domain):
        """Fix geographical coordinates in cubes with rotated coord system."""
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
            self._check_grid_differences(old_coord, new_coord)
            aux_coord_dims = (cube.coord(var_name='rlat').cube_dims(cube) +
                              cube.coord(var_name='rlon').cube_dims(cube))
            cube.add_aux_coord(new_coord, aux_coord_dims)

    def _check_geographical_domain(self, cube, domain_bounds):
        """Check geographical domain fits with the standard domain."""
        dom_lons = np.array(domain_bounds['lons'])
        dom_lats = np.array(domain_bounds['lats'])
        # Crops cube on the standard domain.
        cropped = extract_region(cube,
                                 start_longitude=dom_lons.min(),
                                 end_longitude=dom_lons.max(),
                                 start_latitude=dom_lats.min(),
                                 end_latitude=dom_lats.max())
        # Check: <15% diff in size between original cube and cropped.
        diff = (np.sum(cube.shape) / np.sum(cropped.shape))
        logger.debug(
            "Standard %s domain and data domain "
            "for dataset %s and driver %s present differences of %s %%",
            self.extra_facets['domain'], self.extra_facets['dataset'],
            self.extra_facets['driver'], str((diff - 1.) * 100.))
        cube = cropped
        if diff > 1.15:
            raise RecipeError(
                'Differences between standard geographical domain '
                'and the domain of the cube are above 15%')

    def fix_metadata(self, cubes):
        """Fix CORDEX rotated grids.

        Set rotated coordinates to the
        values given by each domain specification.
        The domain specifications are retrieved from the
        py-cordex package.

        Check that the geographical domain
        for datasets in a Lamber Conformal Conic system
        is within the domain given by the specifications.

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
        domain_bounds = _get_domain_boundaries(data_domain[:3])
        for cube in cubes:
            coord_system = cube.coord_system()
            if isinstance(coord_system, RotatedGeogCS):
                self._fix_rotated_coords(cube, domain, domain_info)
                self._fix_geographical_coords_rotated(cube, domain)
            elif isinstance(coord_system, LambertConformal):
                self._check_geographical_domain(cube, domain_bounds)
                logger.debug(
                    "Further coordinate fixes for Lambert Conformal Conic "
                    "coordinate systems are applied at dataset level.")
            else:
                raise RecipeError(
                    f"Coordinate system {coord_system.grid_mapping_name} "
                    "not supported in CORDEX datasets. Must be "
                    "rotated_latitude_longitude or lambert_conformal_conic.")

        return cubes
