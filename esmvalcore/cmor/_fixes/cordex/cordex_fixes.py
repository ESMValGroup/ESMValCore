"""Fixes that are shared between datasets and drivers."""
import json
import logging
from functools import lru_cache
from pathlib import Path

import cordex as cx
import dask.array as da
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
        coord = iris.coords.DimCoord(points,
                                     var_name=coord_info.name,
                                     standard_name=coord_info.standard_name,
                                     long_name=coord_info.long_name,
                                     units=Unit(coord_info.units),
                                     coord_system=crs
                                     )
        coord.guess_bounds()
        return coord

    @staticmethod
    def _make_geographical_coord(points, coord_info, bounds, lazy=True):
        """Create a geographical coordinate from the specifications."""
        return iris.coords.AuxCoord(
            points if not lazy else da.array(points),
            var_name=coord_info.name,
            standard_name=coord_info.standard_name,
            long_name=coord_info.long_name,
            units=Unit(coord_info.units),
            bounds=bounds if not lazy or bounds is None else da.array(bounds)
        )

    @staticmethod
    def _transform_points(x_data, y_data, crs_from, crs_to):
        """Transform points between one pyproj.crs.CRS to another."""
        res = np.empty(x_data.shape + (2, ))
        res[..., 0], res[..., 1] = Transformer.from_crs(
            crs_from=crs_from,
            crs_to=crs_to,
            always_xy=True
        ).transform(x_data, y_data, errcheck=True)
        return res

    def _check_lambert_conformal_proj_coords(self, cube, domain_bounds):
        """Check lambert conformal projection coordinates."""
        # Transform std domain points in lambert to compare.
        lambert_bounds = self._transform_points(
            np.array(domain_bounds['lons']),
            np.array(domain_bounds['lats']),
            crs_from=GEOG_SYSTEM,
            crs_to=cube.coord_system().as_cartopy_crs()
        )

        proj_x = cube.coord(var_name='x')
        proj_y = cube.coord(var_name='y')
        proj_x.convert_units('m')
        proj_y.convert_units('m')

        # Compare the 4 corners of cube domain vs. the standard domain ones.
        cube_corners = np.take(lambert_bounds, indices=[0, 2, 6, 8], axis=0)
        domain_corners = np.array([
            [proj_x.points.min(), proj_y.points.max()],
            [proj_x.points.max(), proj_y.points.max()],
            [proj_x.points.min(), proj_y.points.min()],
            [proj_x.points.max(), proj_y.points.min()]
        ])
        # Check: euclidean distance between cube domain and std domain < 1e6.
        check = np.mean(
            np.linalg.norm(cube_corners - domain_corners, axis=1)) < 1e6

        # Check passes -> proj coords domain is good -> fix and plug metadata.
        if check:
            # We remake the points to account for possible non monotonicity.
            for vname, coord in zip(("x", "y"), (proj_x, proj_y)):
                self._replace_coord(
                    cube, coord,
                    self._make_projection_coord(
                        np.linspace(
                            coord.points.min(),
                            coord.points.max(),
                            coord.shape[0]),
                        CMOR_TABLES["CORDEX"].coords[vname],
                        cube.coord_system()
                    ))
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
        # Check passes -> geog coords domain is good -> fix and plug metadata.
        if check:
            for vname in ("longitude", "latitude"):
                old_coord = cube.coord(vname)
                self._replace_coord(
                    cube, old_coord,
                    self._make_geographical_coord(
                        old_coord.points,
                        CMOR_TABLES["CORDEX"].coords[vname],
                        old_coord.bounds
                    ))
        return check

    def _fix_lambert_projection_coord_using_geographical(self, cube):
        """Use geographical coords to retrieve projection coords."""
        lambert_system = cube.coord_system().as_cartopy_crs()
        lon_pts = cube.coord("longitude").points
        lat_pts = cube.coord("latitude").points

        # lonlat are in geographical and Dim coords are in lambert system.
        lonlat_lambert = self._transform_points(x_data=lon_pts,
                                                y_data=lat_pts,
                                                crs_from=GEOG_SYSTEM,
                                                crs_to=lambert_system
                                                )

        # Derived from the way aux_coords are built.
        x_points = np.average(lonlat_lambert[:, :, 0], axis=0)
        y_points = np.average(lonlat_lambert[:, :, 1], axis=1)

        # For each coord, make new and replace.
        for vname, points in zip(('x', 'y'), (x_points, y_points)):
            self._replace_coord(
                cube, cube.coord(var_name=vname),
                self._make_projection_coord(
                    points,
                    CMOR_TABLES["CORDEX"].coords[vname],
                    cube.coord_system()
                    ))

    @classmethod
    def _make_geog_bounds_from_proj_bounds(cls, bounds_x, bounds_y, crs_from):
        """Use projection coords bounds to make geographical coords bounds."""
        grid_geog = cls._transform_points(
            *np.meshgrid(bounds_x,
                         bounds_y),
            crs_from=crs_from,
            crs_to=GEOG_SYSTEM
            )

        def format_bnds(bounds):
            bounds = np.array([
                bounds[0::2, 0::2], bounds[0::2, 1::2],
                bounds[1::2, 1::2], bounds[1::2, 0::2]
            ])
            return np.moveaxis(bounds, 0, -1)

        return format_bnds(grid_geog[:, :, 0]), format_bnds(grid_geog[:, :, 1])

    def _fix_geographical_coord_using_projection(self, cube):
        """Use projection coords to retrieve geographical coords."""
        lambert_system = cube.coord_system().as_cartopy_crs()
        proj_x = cube.coord("projection_x_coordinate")
        proj_y = cube.coord("projection_y_coordinate")

        # Dim coords are in lambert system and lonlat are in geographical.
        lonlat = self._transform_points(
            *np.meshgrid(proj_x.points,
                         proj_y.points),
            crs_from=lambert_system,
            crs_to=GEOG_SYSTEM
            )

        lonlat_bounds = self._make_geog_bounds_from_proj_bounds(
            proj_x.bounds, proj_y.bounds, lambert_system)

        for vname, points, bounds in zip(("longitude", "latitude"),
                                         (lonlat[:, :, 0], lonlat[:, :, 1]),
                                         lonlat_bounds):
            self._replace_coord(
                cube, cube.coord(vname),
                self._make_geographical_coord(
                    points,
                    CMOR_TABLES["CORDEX"].coords[vname],
                    bounds,
                    lazy=True
                    ))

    def _check_lonlat_bounds(self, cube):
        """Make sure longitude and latitude have bounds."""
        lambert_system = cube.coord_system().as_cartopy_crs()
        lon = cube.coord("longitude")
        lat = cube.coord("latitude")
        if lon.bounds is None or lat.bounds is None:
            bounds_x = cube.coord("projection_x_coordinate").bounds
            bounds_y = cube.coord("projection_y_coordinate").bounds
            lon.bounds, lat.bounds = self._make_geog_bounds_from_proj_bounds(
                bounds_x, bounds_y, lambert_system)

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
                        "coordinates of the cube seem to present large "
                        "differences with the standard domain.")

                # Not necessary for the fixes, but important for later.
                self._check_lonlat_bounds(cube)

            else:
                raise RecipeError(
                    f"Coordinate system {coord_system.grid_mapping_name} "
                    "not supported in CORDEX datasets. Must be "
                    "rotated_latitude_longitude or lambert_conformal_conic.")

        return cubes
