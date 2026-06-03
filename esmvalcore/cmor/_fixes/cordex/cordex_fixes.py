"""Fixes that are shared between datasets and drivers."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import cordex as cx
import iris
import iris.coords
import iris.cube
import numpy as np
import pyproj
from cf_units import Unit
from iris.coord_systems import GeogCS, LambertConformal, RotatedGeogCS
from iris.fileformats.pp import EARTH_RADIUS

from esmvalcore.cmor.fix import Fix

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import xarray as xr


logger = logging.getLogger(__name__)


@lru_cache
def _get_domain(data_domain: str) -> xr.Dataset:
    return cx.cordex_domain(data_domain, bounds=True)


@lru_cache
def _get_domain_info(data_domain: str) -> dict[str, str | float]:
    return cx.domain_info(data_domain)


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
            cube.coord("latitude").var_name = "lat"
            cube.coord("longitude").var_name = "lon"
            for coord in cube.coords("time"):
                coord.long_name = "time"

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
            cube.coord("time").long_name = "time"

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
            # Fix the calendar if there is a "time" coordinate.
            for coord in cube.coords("time"):
                time_unit = coord.units
                if time_unit.calendar == "standard":
                    new_unit = time_unit.change_calendar("proleptic_gregorian")
                    coord.units = new_unit
            # Fix the endianness of the data.
            if cube.core_data().dtype == np.dtype(">f4"):
                cube.data = cube.core_data().astype(
                    np.float32,
                    casting="same_kind",
                )
            for coord in cube.coords():
                if coord.dtype in [">f8", ">f4"]:
                    coord.points = coord.core_points().astype(
                        np.float64,
                        casting="same_kind",
                    )
                    if coord.has_bounds():
                        coord.bounds = coord.core_bounds().astype(
                            np.float64,
                            casting="same_kind",
                        )
        return cubes


class AllVars(Fix):
    """General CORDEX grid fix."""

    def _check_grid_differences(
        self,
        old_cube: iris.cube.Cube,
        new_cube: iris.cube.Cube,
        coordinates: Iterable[str],
        max_diff: float,
    ) -> None:
        """Check differences between coords."""
        for coord_name in coordinates:
            old_coord = old_cube.coord(coord_name)
            new_coord = new_cube.coord(coord_name)
            diff = np.max(np.abs(old_coord.points - new_coord.points))
            log = logger.warning if diff > max_diff else logger.debug
            log(
                "Maximum difference between original %s "
                "points and standard %s domain points "
                "for variable %s from dataset %s and driver %s is: %s %s.",
                new_coord.standard_name,
                self.extra_facets["domain"],
                self.extra_facets["short_name"],
                self.extra_facets["dataset"],
                self.extra_facets["driver"],
                str(diff),
                new_coord.units,
            )
            # TODO: Should we check bounds too?
            # TODO: Handle 360 degree longitude wrap-around for longitude bounds?

    def _fix_rotated_coords(
        self,
        cube: iris.cube.Cube,
        domain: xr.Dataset,
        domain_info: dict[str, str | float],
    ) -> None:
        """Fix rotated coordinates."""
        for dim_coord in ["rlat", "rlon"]:
            old_coord = cube.coord(domain[dim_coord].standard_name)
            old_coord_dims = old_coord.cube_dims(cube)
            points = domain[dim_coord].data
            coord_system = RotatedGeogCS(
                grid_north_pole_latitude=domain_info["pollat"],
                grid_north_pole_longitude=domain_info["pollon"],
            )
            new_coord = iris.coords.DimCoord(
                points,
                var_name=dim_coord,
                standard_name=domain[dim_coord].standard_name,
                long_name=domain[dim_coord].long_name,
                units=Unit("degrees"),
                coord_system=coord_system,
            )
            new_coord.guess_bounds()
            cube.remove_coord(old_coord)
            cube.add_dim_coord(new_coord, old_coord_dims)

    def _fix_geographical_coords(
        self,
        cube: iris.cube.Cube,
        domain: xr.Dataset,
    ) -> None:
        """Fix geographical coordinates."""
        for aux_coord in ["lat", "lon"]:
            old_coord = cube.coord(domain[aux_coord].standard_name)
            cube.remove_coord(old_coord)
            points = domain[aux_coord].data
            bounds = domain[f"{aux_coord}_vertices"].data
            new_coord = iris.coords.AuxCoord(
                points,
                var_name=aux_coord,
                standard_name=domain[aux_coord].standard_name,
                long_name=domain[aux_coord].long_name,
                units=Unit(domain[aux_coord].units),
                bounds=bounds,
            )
            if aux_coord == "lon" and new_coord.points.min() < 0.0:
                lon_inds = (new_coord.points < 0.0) & (old_coord.points > 0.0)
                old_coord.points[lon_inds] = old_coord.points[lon_inds] - 360.0

            aux_coord_dims = cube.coord(var_name="rlat").cube_dims(
                cube,
            ) + cube.coord(var_name="rlon").cube_dims(cube)
            cube.add_aux_coord(new_coord, aux_coord_dims)

    def _use_standard_lambert_conformal_grid(
        self,
        cube: iris.cube.Cube,
    ) -> None:
        """Use standard Lambert Conformal grid.

        Parameters
        ----------
        cube :
            Apply the standard Lambert Conformal grid to this cube.

        """
        domain_step = {
            "11": 12500,
            "22": 25000,
            "44": 50000,
        }

        # Update the projection coordinates.
        domain_resolution = self.extra_facets["domain"].split("-")[-1]
        if domain_resolution not in domain_step:
            logger.warning(
                "Unable to use standard grid for dataset %s and driver %s "
                "because domain resolution %s is not supported, choose from %s.",
                self.extra_facets["dataset"],
                self.extra_facets["driver"],
                domain_resolution,
                ", ".join(domain_step.keys()),
            )
            return

        step = domain_step[domain_resolution]

        # Update the projection coordinates and bounds.
        x_coord = cube.coord("projection_x_coordinate")
        x_size = x_coord.shape[0]
        x_coord.points = step * np.linspace(
            -0.5 * (x_size - 1),
            0.5 * (x_size - 1),
            x_size,
        )
        x_coord.units = "m"
        x_coord.guess_bounds()

        y_coord = cube.coord("projection_y_coordinate")
        y_size = y_coord.shape[0]
        y_coord.points = step * np.linspace(
            -0.5 * (y_size - 1),
            0.5 * (y_size - 1),
            y_size,
        )
        y_coord.units = "m"
        y_coord.guess_bounds()

        # Define the transformation from projection coordinates to
        # geographic coordinates.
        transformer = pyproj.Transformer.from_crs(
            crs_from=x_coord.coord_system.as_cartopy_crs(),
            crs_to=GeogCS(EARTH_RADIUS).as_cartopy_crs(),
            always_xy=True,
        )

        # Update the latitude and longitude points.
        cube.coord("longitude").points, cube.coord("latitude").points = (
            transformer.transform(
                *np.meshgrid(x_coord.points, y_coord.points),
                errcheck=True,
            )
        )

        # Update the latitude and longitude bounds.
        #
        # Compute the bounds of the grid by indexing the bounds arrays
        # according to
        # https://cfconventions.org/Data/cf-conventions/cf-conventions-1.13/cf-conventions.html#cell-boundaries
        # and transforming the resulting bounds to lat/lon coordinates.
        x_bounds = np.concatenate(
            [x_coord.bounds[:1, 0], x_coord.bounds[:, 1]],
        )
        y_bounds = np.concatenate(
            [y_coord.bounds[:1, 0], y_coord.bounds[:, 1]],
        )
        n_vertices = 4
        x_idx = np.arange(x_size).reshape(1, x_size).repeat(
            y_size,
            axis=0,
        ).reshape(x_size, y_size, 1) + np.array([0, 1, 1, 0]).reshape(
            (1, 1, n_vertices),
        )
        y_idx = np.arange(y_size).reshape(y_size, 1).repeat(
            x_size,
            axis=1,
        ).reshape(x_size, y_size, 1) + np.array([0, 0, 1, 1]).reshape(
            (1, 1, n_vertices),
        )
        x_vertices = x_bounds[x_idx]
        y_vertices = y_bounds[y_idx]
        cube.coord("longitude").bounds, cube.coord("latitude").bounds = (
            transformer.transform(
                x_vertices,
                y_vertices,
                errcheck=True,
            )
        )

    def fix_metadata(
        self,
        cubes: Sequence[iris.cube.Cube],
    ) -> Sequence[iris.cube.Cube]:
        """Fix CORDEX rotated grids.

        Set rotated and geographical coordinates to the
        values given by each domain specification.

        The domain specifications are retrieved from the
        py-cordex package.

        Parameters
        ----------
        cubes :
            Input cubes.

        Returns
        -------
        :
            Fixed cubes.
        """
        data_domain = self.extra_facets["domain"]
        domain = _get_domain(data_domain)
        domain_info = _get_domain_info(data_domain)

        for cube in cubes:
            coord_system = cube.coord_system()
            if isinstance(coord_system, LambertConformal):
                logger.warning(
                    "Support for CORDEX datasets in a Lambert Conformal "
                    "coordinate system is ongoing. Certain preprocessor "
                    "functions may fail.",
                )
        if not self.extra_facets.get("use_standard_grid"):
            return cubes

        result = []
        for cube in cubes:
            coord_system = cube.coord_system()
            updated_cube = cube.copy()
            if isinstance(coord_system, RotatedGeogCS):
                self._fix_rotated_coords(updated_cube, domain, domain_info)
                self._fix_geographical_coords(updated_cube, domain)
                self._check_grid_differences(
                    cube,
                    updated_cube,
                    coordinates=[
                        "grid_latitude",
                        "grid_longitude",
                        "latitude",
                        "longitude",
                    ],
                    max_diff=10e-4,  # degrees
                )
            elif isinstance(coord_system, LambertConformal):
                self._use_standard_lambert_conformal_grid(updated_cube)
                self._check_grid_differences(
                    cube,
                    updated_cube,
                    coordinates=[
                        "projection_x_coordinate",
                        "projection_y_coordinate",
                    ],
                    max_diff=1000.0,  # meter
                )
                self._check_grid_differences(
                    cube,
                    updated_cube,
                    coordinates=["latitude", "longitude"],
                    max_diff=10e-4,  # degrees
                )
            result.append(updated_cube)

        return result
