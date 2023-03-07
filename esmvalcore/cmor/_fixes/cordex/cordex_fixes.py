"""Fixes that are shared between datasets and drivers."""
import logging
import json
from functools import lru_cache
from pathlib import Path

from cf_units import Unit
import cordex as cx
import iris
from iris.coord_systems import LambertConformal, RotatedGeogCS, GeogCS
from iris.fileformats.pp import EARTH_RADIUS
import numpy as np
import dask.array as da
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
            self._check_grid_differences(old_coord, new_coord)
            aux_coord_dims = (cube.coord(var_name='rlat').cube_dims(cube) +
                              cube.coord(var_name='rlon').cube_dims(cube))
            cube.add_aux_coord(new_coord, aux_coord_dims)

    @staticmethod
    def _replace_coord(cube, old_coord, new_coord):
        """Replace old_coord by new_coord in cube."""
        dim = old_coord.cube_dims(cube)
        cube.remove_coord(old_coord)
        if isinstance(old_coord, iris.coords.DimCoord):
            cube.add_dim_coord(new_coord, dim)
        elif isinstance(old_coord, iris.coords.AuxCoord):
            cube.add_aux_coord(new_coord, dim)

    @staticmethod
    def _make_projection_coord(points, coord_info, crs):
        """Create a projection coordinate from the specifications."""
        coord = iris.coords.DimCoord(
            points,
            var_name=coord_info.name,
            standard_name=coord_info.standard_name,
            long_name=coord_info.long_name,
            units=Unit(coord_info.units),
            coord_system=crs
        )
        coord.guess_bounds()
        return coord

    @staticmethod
    def _make_geographical_coord(points, coord_info, bounds):
        """Create a geographical coordinate from the specifications."""
        return iris.coords.AuxCoord(
            points,
            var_name=coord_info.name,
            standard_name=coord_info.standard_name,
            long_name=coord_info.long_name,
            units=Unit(coord_info.units),
            bounds=bounds
        )

    @staticmethod
    def _transform_points(crs_from, crs_to, x_data, y_data, lazy=False):
        """Transform points between one pyproj.crs.CRS to another."""
        out_shape = x_data.shape + (2,)
        res = da.empty(out_shape) if lazy else np.empty(out_shape)
        res[..., 0], res[..., 1] = Transformer.from_crs(
            crs_from=crs_from,
            crs_to=crs_to,
            always_xy=True
        ).transform(x_data, y_data, errcheck=True)
        return res

    def _check_lambert_conformal_proj_coords(self, cube, domain_bounds):
        """Check lambert conformal projection coordinates."""
        # Transform std domain points in lambert to compare.
        lambert_system = cube.coord_system()
        lambert_bounds = self._transform_points(
            GEOG_SYSTEM, lambert_system.as_cartopy_crs(),
            np.array(domain_bounds['lons']),
            np.array(domain_bounds['lats']))

        proj_x = cube.coord(var_name='x')
        proj_y = cube.coord(var_name='y')
        proj_x.convert_units('m')
        proj_y.convert_units('m')

        xmin, xmax = proj_x.points.min(), proj_x.points.max()
        ymin, ymax = proj_y.points.min(), proj_y.points.max()

        # Compare the 4 corners of cube domain vs. the standard domain ones.
        cube_corners = np.take(lambert_bounds, indices=[0, 2, 6, 8], axis=0)
        domain_corners = np.array([
            [xmin, ymax], [xmax, ymax],
            [xmin, ymin], [xmax, ymin]])
        # Check: euclidean distance between cube domain and std domain < 1e6.
        check = np.mean(
            np.linalg.norm(cube_corners - domain_corners, axis=1)) < 1e6

        # Check passes -> proj coords are good -> fix and plug metadata.
        if check:
            var_infos = CMOR_TABLES["CORDEX"].coords
            new_proj_x = self._make_projection_coord(
                proj_x.points, var_infos["x"], lambert_system)
            new_proj_y = self._make_projection_coord(
                proj_y.points, var_infos["y"], lambert_system)
            self._replace_coord(cube, proj_x, new_proj_x)
            self._replace_coord(cube, proj_y, new_proj_y)
        return check

    def _check_lambert_conformal_geog_coords(self, cube, domain_bounds):
        """Check lambert conformal geographical coordinates."""
        dom_lons = np.array(domain_bounds['lons'])
        dom_lats = np.array(domain_bounds['lats'])
        # Check without time coordinate if present.
        cube2d = cube[0] if 'time' in self.vardef.dimensions else cube
        # Crops cube on the standard domain.
        cube_x_std = extract_region(cube2d,
                                    start_longitude=dom_lons.min(),
                                    end_longitude=dom_lons.max(),
                                    start_latitude=dom_lats.min(),
                                    end_latitude=dom_lats.max())
        # Check: <15% diff in size between original cube and cropped.
        check = (np.sum(cube2d.shape) / np.sum(cube_x_std.shape)) < 1.15
        # Check passes -> geog coords are good -> fix and plug metadata.
        if check:
            var_infos = CMOR_TABLES["CORDEX"].coords
            for var_name in ["longitude", "latitude"]:
                old_coord = cube.coord(var_name)
                new_coord = self._make_geographical_coord(
                    old_coord.points, var_infos[var_name], old_coord.bounds)
                self._replace_coord(cube, old_coord, new_coord)
        return check

    def _fix_lambert_projection_coord_using_geographical(self, cube):
        """Use geographical coords to retrieve projection coords."""
        # Retrieve infos from cube.
        lambert_system = cube.coord_system().as_cartopy_crs()
        lon_pts = cube.coord("longitude").points
        lat_pts = cube.coord("latitude").points

        # lonlat are in geographical and Dim coords are in lambert system.
        lonlat_lambert = self._transform_points(crs_from=GEOG_SYSTEM,
                                                crs_to=lambert_system,
                                                x_data=lon_pts,
                                                y_data=lat_pts)

        # Derived from the way aux_coords are built.
        x_points = np.average(lonlat_lambert[:, :, 0], axis=0)
        y_points = np.average(lonlat_lambert[:, :, 1], axis=1)

        # For each coord, make new and replace.
        var_infos = CMOR_TABLES["CORDEX"].coords
        for var_name, points in zip(['x', 'y'], [x_points, y_points]):
            new_coord = self._make_projection_coord(points,
                                                    var_infos[var_name],
                                                    cube.coord_system())
            self._replace_coord(cube, cube.coord(var_name=var_name), new_coord)

    def _fix_geographical_coord_using_projection(self, cube):
        """Use projection coords to retrieve geographical coords."""
        proj_x = cube.coord("projection_x_coordinate")
        proj_y = cube.coord("projection_y_coordinate")

        # AuxCoords are built in 2D from the coords points.
        points_grid = da.meshgrid(proj_x.points, proj_y.points)
        bounds_grid = da.meshgrid(proj_x.bounds, proj_y.bounds)

        # Dim coords are in lambert system and lonlat are in geographical.
        lonlat = self._transform_points(
            crs_from=cube.coord_system().as_cartopy_crs(), crs_to=GEOG_SYSTEM,
            x_data=points_grid[0], y_data=points_grid[1], lazy=True)
        lonlat_bounds = self._transform_points(
            crs_from=cube.coord_system().as_cartopy_crs(), crs_to=GEOG_SYSTEM,
            x_data=bounds_grid[0], y_data=bounds_grid[1], lazy=True)

        # For each coord, make new and replace.
        for i, var_name in enumerate(["longitude", "latitude"]):
            bounds = lonlat_bounds[:, :, i]
            bounds = [bounds[::2, ::2], bounds[::2, 1::2],
                      bounds[1::2, 1::2], bounds[1::2, ::2]]
            bounds = da.moveaxis(da.array(bounds), 0, -1)

            new_coord = self._make_geographical_coord(
                lonlat[..., i], CMOR_TABLES["CORDEX"].coords[var_name], bounds)
            self._replace_coord(cube, cube.coord(var_name), new_coord)

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
        domain_bounds = _get_domain_boundaries(data_domain[:3])
        for cube in cubes:
            coord_system = cube.coord_system()
            if isinstance(coord_system, RotatedGeogCS):
                self._fix_rotated_coords(cube, domain, domain_info)
                self._fix_geographical_coords(cube, domain)
            elif isinstance(coord_system, LambertConformal):
                proj_is_good = self._check_lambert_conformal_proj_coords(
                    cube, domain_bounds)
                geog_is_good = self._check_lambert_conformal_geog_coords(
                    cube, domain_bounds)

                if geog_is_good and not proj_is_good:
                    self._fix_lambert_projection_coord_using_geographical(cube)

                if proj_is_good and not geog_is_good:
                    self._fix_geographical_coord_using_projection(cube)

                if not proj_is_good and not geog_is_good:
                    raise RecipeError(
                        "Both projection and geographical "
                        "coordinates of the cube seems to present large "
                        "differences with the standard domain.")
            else:
                raise RecipeError(
                    f"Coordinate system {coord_system.grid_mapping_name} "
                    "not supported in CORDEX datasets. Must be "
                    "rotated_latitude_longitude or lambert_conformal_conic.")

        return cubes
