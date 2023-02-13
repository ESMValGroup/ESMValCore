"""Fixes that are shared between datasets and drivers."""
import logging
from functools import lru_cache

import cordex as cx
import iris
import json
import re
import os

# import xarray as xr 
# from esmf_regrid.esmf_regridder import Regridder
# from cartopy.crs import RotatedGeodetic

import numpy as np
from cf_units import Unit
from iris.coord_systems import LambertConformal, RotatedGeogCS

from esmvalcore.cmor.fix import Fix
from esmvalcore.exceptions import RecipeError

logger = logging.getLogger(__name__)


@lru_cache
def _get_domains_boundaries():
    """ load from the corresponding JSON file """
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # Build file path assuming it is located in the same directory as this file
    fpath = os.path.join(location, 'nrp_rcm_domain_boundaries.json')
    # JSON file contains comments which are not supported by json base package
    with open(fpath, "r") as f:
        data = json.loads("".join(re.split(r"(?://|#).*(?=\n)", f.read())).strip())
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

    def _fix_lambert_conformal_coords(self, cube, data_domain):
        """Fix lambert conformal coordinates"""
        # Load domain boundaries 
        #  and get corresponding data from the data_domain 3 first letters (e.g. 'EUR-11' -> 'EUR')
        boundaries = _get_domains_boundaries()[data_domain[:3]]
        lats, lons = boundaries["lats"], boundaries["lons"]

        # load cmor grid for cordex
        from .. . import table as ct
        table = ct.CMOR_TABLES['CORDEX']
        table._load_table(table._cmor_folder + "/CORDEX_grids")
        
        # compute coord indices
        has_time = 'time' in self.vardef.dimensions
        xc_ix, yc_ix = (2, 1) if has_time else (1, 0)
        ndim = len(cube.shape)
        if ndim < (2 + (1 if has_time else 0)):
            raise RecipeError(
                f"Missing coordinate: expected a least 2 coordinates (+time for cubes concerned)"
                 "corresponding to x and y. Got: {ndim}")
    
        # Domain boundaries coordinates are in actual coordinates so we transform the points in Lambert
        lambert_system = cube.coord_system().as_cartopy_crs()
        geog_system = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS).as_cartopy_crs()
        xyz = lambert_system.transform_points(geog_system, np.array(lons), np.array(lats))

        # dicts to store variables for loops
        dims = {"x": {"idx": xc_ix, "coord": None}, "y": {"idx": yc_ix, "coord": None}}
        auxi = {"longitude": {"idx": 0}, "latitude": {"idx": 1}}

        # DimCoords
        for coord in cube.coords():
            for key, data in dims.items():
                data['standard'] = np.linspace(xyz[:,0].min(),
                                               xyz[:,0].max(),
                                               cube.shape[data['idx']])
                data['coord'] = iris.coords.DimCoord(
                    data['standard'],
                    var_name=table.coords[key].name,
                    standard_name=table.coords[key].standard_name,
                    long_name=table.coords[key].long_name,
                    units=table.coords[key].units,
                    coord_system=cube.coord_system()
                )
                data['coord'].guess_bounds()

        # 
        gx, gy = np.meshgrid(dims['x']['standard'], dims['y']['standard'])
        lonlat = geog_system.transform_points(lambert_system, gx, gy)
        
        gx_bounds, gy_bounds = np.meshgrid(dims['x']['coord'].bounds, dims['y']['coord'].bounds)
        lonlat_bounds = geog_system.transform_points(lambert_system, gx_bounds, gy_bounds)

        # AuxCoords
        for key, data in auxi.items():
            bounds = lonlat_bounds[:, :, data["idx"]]
            bounds = [bounds[::2, ::2]  , bounds[::2, 1::2],
                      bounds[1::2, 1::2], bounds[1::2, ::2]]
            bounds = np.array(bounds)
            bounds = np.moveaxis(bounds, 0, -1)
            data["coord"] = iris.coords.AuxCoord(
                lonlat[:, :, data["idx"]], # Points
                var_name=table.coords[key].name,
                standard_name=table.coords[key].standard_name,
                long_name=table.coords[key].long_name,
                units=table.coords[key].units,
                bounds=bounds,
            )

        # For testing purposes, TODO: remove in final version
        diff_pxc = cube.coord('projection_x_coordinate').points - dims['x']['coord'].points
        diff_pyc = cube.coord('projection_y_coordinate').points - dims['y']['coord'].points
        diff_lon = cube.coord('longitude').points - auxi['longitude']['coord'].points
        diff_lat = cube.coord('latitude').points - auxi['latitude']['coord'].points
        infos = ["sum", "mean", "std", "min", "max"]
        infos_pxc = [diff_pxc.sum(), diff_pxc.mean(), diff_pxc.std(), diff_pxc.min(), diff_pxc.max()]   
        infos_pyc = [diff_pyc.sum(), diff_pyc.mean(), diff_pyc.std(), diff_pyc.min(), diff_pyc.max()]
        infos_lon = [diff_lon.sum(), diff_lon.mean(), diff_lon.std(), diff_lon.min(), diff_lon.max()]
        infos_lat = [diff_lat.sum(), diff_lat.mean(), diff_lat.std(), diff_lat.min(), diff_lat.max()]
        for info in [infos_pxc, infos_pyc, infos_lon, infos_lat]:
            for name, val in zip(infos, info):
                print(name, val),
            print("")

        # remove all old coords attached to axis
        for coord in cube.coords(axis="x") + cube.coords(axis="y"):
            cube.remove_coord(coord)
        
        # plug in new coords
        cube.add_dim_coord(dims['x']['coord'], xc_ix)
        cube.add_dim_coord(dims['y']['coord'], yc_ix)
        cube.add_aux_coord(auxi['longitude']['coord'], (yc_ix, xc_ix))
        cube.add_aux_coord(auxi['latitude']['coord'], (yc_ix, xc_ix))

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
