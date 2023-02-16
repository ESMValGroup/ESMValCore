"""Fixes that are shared between datasets and drivers."""
import logging
import os
import json
from functools import lru_cache

from cf_units import Unit
import cordex as cx
import dask.array as da
import iris
from iris.coord_systems import LambertConformal, RotatedGeogCS
from iris.fileformats.pp import EARTH_RADIUS
import numpy as np
from pyproj import Transformer

from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import CMOR_TABLES
from esmvalcore.exceptions import RecipeError

logger = logging.getLogger(__name__)


@lru_cache
def _get_domains_boundaries():
    """Load from the corresponding JSON file."""
    location = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # Build file path assuming it is located in the same directory as this file
    fpath = os.path.join(location, 'nrp_rcm_domain_boundaries.json')
    with open(fpath, "r") as f:
        data = json.loads(f.read())
    return data


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
    def _lazy_transformation(transformer, x, y):
        """cartopy.CRS.transform_points in our case using dask API"""
        result_shape = x.shape + (3, )
        x, y = x.flatten(), y.flatten()
        res = da.empty([x.shape[0], 3], dtype=np.double)
        res[:, 0], res[:, 1], res[:, 2] = transformer.transform(
            x, y, da.zeros_like(x), errcheck=True)
        return res.reshape(result_shape)

    @staticmethod
    def _make_projection_coord(start, stop, size, vardef, crs):
        """Create a projection coordinate from the specifications."""
        coord = iris.coords.DimCoord(
            np.linspace(start, stop, size),
            var_name=vardef.name,
            standard_name=vardef.standard_name,
            long_name=vardef.long_name,
            units=vardef.units,
            coord_system=crs
        )
        coord.guess_bounds()
        return coord

    @staticmethod
    def _make_geographical_coord(grid, bounds, vardef, dim):
        """Create a geographical coordinate from the specifications."""
        bounds = bounds[:, :, dim]
        bounds = [bounds[::2, ::2], bounds[::2, 1::2],
                  bounds[1::2, 1::2], bounds[1::2, ::2]]
        bounds = da.array(bounds)
        bounds = da.moveaxis(bounds, 0, -1)
        return iris.coords.AuxCoord(
            grid[:, :, dim],
            var_name=vardef.name,
            standard_name=vardef.standard_name,
            long_name=vardef.long_name,
            units=vardef.units,
            bounds=bounds,
        )

    def _fix_lambert_conformal_coords(self, cube, data_domain):
        """Fix lambert conformal coordinates."""
        # Load domain boundaries.
        boundaries = _get_domains_boundaries()[data_domain[:3]]
        lats, lons = np.array(boundaries["lats"]), np.array(boundaries["lons"])

        # Load cmor grid for cordex.
        cordex_tables = CMOR_TABLES['CORDEX']
        cordex_tables._load_table(cordex_tables._cmor_folder + "/CORDEX_grids")

        # Compute coord indices.
        has_time = 'time' in self.vardef.dimensions
        xc_idx, yc_idx = (2, 1) if has_time else (1, 0)

        # Check dim validity.
        theoretical = len(self.vardef.dimensions)
        if cube.ndim < theoretical:
            raise RecipeError(
                f"Missing dimension in cube: expected {theoretical} "
                f"got: {cube.ndim}")

        # Domain boundaries coordinates are in actual coordinates.
        # So we transform the points in Lambert.
        lambert_system = cube.coord_system().as_cartopy_crs()
        geog_system = iris.coord_systems.GeogCS(EARTH_RADIUS).as_cartopy_crs()

        lambert_bounds = lambert_system.transform_points(
            src_crs=geog_system, x=lons, y=lats)
        start, end = lambert_bounds[:, 0].min(), lambert_bounds[:, 0].max()

        # Then build coords passing the boundaries and the specifications.
        coord_x = self._make_projection_coord(
            start, end, cube.shape[xc_idx],
            cordex_tables.coords["x"], cube.coord_system())

        coord_y = self._make_projection_coord(
            start, end, cube.shape[yc_idx],
            cordex_tables.coords["y"], cube.coord_system())

        # AuxCoords are built in 2D from the coords.
        gx, gy = da.meshgrid(coord_x.points, coord_y.points)

        gx_bounds, gy_bounds = da.meshgrid(coord_x.bounds, coord_y.bounds)

        # Inverse transformation is applied.
        transformer = Transformer.from_crs(
            crs_from=lambert_system, crs_to=geog_system, always_xy=True)

        lonlat = self._lazy_transformation(transformer, gx, gy)
        lonlat_bounds = self._lazy_transformation(
            transformer, gx_bounds, gy_bounds)

        coord_lon = self._make_geographical_coord(
            lonlat, lonlat_bounds, cordex_tables.coords["longitude"], 0)

        coord_lat = self._make_geographical_coord(
            lonlat, lonlat_bounds, cordex_tables.coords["latitude"], 1)

        # Remove all old coords attached to cube axis.
        for coord in cube.coords(axis="x") + cube.coords(axis="y"):
            cube.remove_coord(coord)

        # Plug in new coords.
        cube.add_dim_coord(coord_x, xc_idx)
        cube.add_dim_coord(coord_y, yc_idx)
        cube.add_aux_coord(coord_lon, (yc_idx, xc_idx))
        cube.add_aux_coord(coord_lat, (yc_idx, xc_idx))

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
                self._fix_lambert_conformal_coords(cube, data_domain)
            else:
                raise RecipeError(
                    f"Coordinate system {coord_system.grid_mapping_name} "
                    "not supported in CORDEX datasets. Must be "
                    "rotated_latitude_longitude or lambert_conformal_conic.")

        return cubes
