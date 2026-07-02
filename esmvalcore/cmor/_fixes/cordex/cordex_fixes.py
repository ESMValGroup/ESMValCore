"""Fixes that are shared between datasets and drivers."""

from __future__ import annotations

import contextlib
import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import cordex as cx
import iris
import iris.coords
import iris.cube
import iris.exceptions
import iris.util
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


_PROJECTION_STEP_IN_METERS = {
    "11": 12500,
    "12": 12500,
    "22": 25000,
    "25": 25000,
    "44": 50000,
    "50": 50000,
}


def _get_projection_step(domain: str) -> float | None:
    """Get the projection step in meters for a given CORDEX domain."""
    domain_resolution = domain.rsplit("-", maxsplit=1)[-1]
    return _PROJECTION_STEP_IN_METERS.get(domain_resolution)


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
            new_coord = new_cube.coord(coord_name)
            old_coord = old_cube.coord(coord_name).copy()
            with contextlib.suppress(
                ValueError,
                iris.exceptions.UnitConversionError,
            ):
                # Try to convert old_coord to the same units as new_coord for a
                # a more informative log message.
                old_coord.convert_units(new_coord.units)
            if coord_name == "longitude":
                # Avoid issuing a warning due to the new coordinate being on the
                # other side of the zero meridian compared to the old coordinate
                # as described in https://github.com/ESMValGroup/ESMValCore/issues/2036.
                # Check adapted from: https://github.com/ESMValGroup/ESMValCore/pull/2334.
                lon_inds = (new_coord.points < 0.0) & (old_coord.points > 0.0)
                old_coord.points[lon_inds] = old_coord.points[lon_inds] - 360.0
            diff = np.max(np.abs(old_coord.points - new_coord.points))
            log = logger.warning if diff > max_diff else logger.debug
            log(
                "Maximum difference between original %s "
                "points and standard %s domain points "
                "for variable %s from dataset %s is %s %s. "
                "Consider visualizing the data on a map and comparing with "
                "recognizable features such as coastlines to check that the "
                "grid is correct.",
                new_coord.standard_name,
                self.extra_facets["domain"],
                new_cube.var_name,
                self.extra_facets["dataset"],
                str(diff),
                new_coord.units,
            )
            # TODO: Should we check bounds too?

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
        # Update the projection coordinates.
        step = _get_projection_step(self.extra_facets["domain"])
        if step is None:
            logger.warning(
                "Unable to use standard Lambert Conformal grid for domain %s, "
                "choose a domain ending with %s.",
                self.extra_facets["domain"],
                ", ".join(_PROJECTION_STEP_IN_METERS.keys()),
            )
            return

        # Update the projection coordinates and bounds.
        x_coord = cube.coord("projection_x_coordinate")
        x_coord.var_name = "x"
        x_coord.long_name = "x coordinate of projection"
        x_coord.units = "m"
        x_size = x_coord.shape[0]
        x_coord.points = step * np.linspace(
            -0.5 * (x_size - 1),
            0.5 * (x_size - 1),
            x_size,
        )
        x_coord.guess_bounds()

        y_coord = cube.coord("projection_y_coordinate")
        y_coord.var_name = "y"
        y_coord.long_name = "y coordinate of projection"
        y_coord.units = "m"
        y_size = y_coord.shape[0]
        y_coord.points = step * np.linspace(
            -0.5 * (y_size - 1),
            0.5 * (y_size - 1),
            y_size,
        )
        y_coord.guess_bounds()

        # If the original coordinate was not monotonic, it has been downgraded
        # to an auxiliary coordinate, so promote it back to a dimension
        # coordinate.
        iris.util.promote_aux_coord_to_dim_coord(
            cube,
            "projection_x_coordinate",
        )
        iris.util.promote_aux_coord_to_dim_coord(
            cube,
            "projection_y_coordinate",
        )

        # Define the transformation from projection coordinates to
        # geographic coordinates.
        transformer = pyproj.Transformer.from_crs(
            crs_from=x_coord.coord_system.as_cartopy_crs(),
            crs_to=GeogCS(EARTH_RADIUS).as_cartopy_crs(),
            always_xy=True,
        )

        # Update the latitude and longitude points.
        lon_coord = cube.coord("longitude")
        lat_coord = cube.coord("latitude")
        lon_coord.var_name = "lon"
        lat_coord.var_name = "lat"
        lon_coord.long_name = "longitude"
        lat_coord.long_name = "latitude"
        lon_coord.units = "degrees_east"
        lat_coord.units = "degrees_north"
        lon_coord.points, lat_coord.points = transformer.transform(
            *np.meshgrid(x_coord.points, y_coord.points),
            errcheck=True,
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
        lon_coord.bounds, lat_coord.bounds = transformer.transform(
            x_vertices,
            y_vertices,
            errcheck=True,
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
            # Set the maximum allowed difference between the original and
            # standard grid points to 10% of the grid spacing:
            max_tolerance_degrees = 0.1 * domain_info["dlon"]  # type: ignore[operator]
            max_tolerance_meters = 0.1 * (
                _get_projection_step(self.extra_facets["domain"]) or 10000.0
            )
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
                    max_diff=max_tolerance_degrees,  # degrees
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
                    max_diff=max_tolerance_meters,  # meter
                )
                self._check_grid_differences(
                    cube,
                    updated_cube,
                    coordinates=["latitude", "longitude"],
                    max_diff=max_tolerance_degrees,  # degrees
                )
            result.append(updated_cube)

        return result
