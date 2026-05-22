"""Fixes for rcm ALADIN63 driven by CNRM-CERFACS-CNRM-CM5."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris.coord_systems
import numpy as np
import pyproj
from iris.fileformats.pp import EARTH_RADIUS

from esmvalcore.cmor._fixes.cordex.cordex_fixes import TimeLongName as BaseFix
from esmvalcore.cmor._fixes.shared import add_scalar_height_coord
from esmvalcore.cmor.fix import Fix

if TYPE_CHECKING:
    from collections.abc import Sequence

    from iris.cube import Cube


class AllVars(Fix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes: Sequence[Cube]) -> Sequence[Cube]:
        domain_step = {
            "11": 12500,
            "22": 25000,
            "44": 50000,
        }

        for cube in cubes:
            # Correct the projection coordinates.
            domain_resolution = self.extra_facets["domain"].split("-")[-1]
            step = domain_step[domain_resolution]

            for coord_name in [
                "projection_x_coordinate",
                "projection_y_coordinate",
            ]:
                coord = cube.coord(coord_name)
                n_steps = coord.shape[0]
                coord.points = step * np.linspace(
                    -(n_steps - 1) / 2,
                    (n_steps - 1) / 2,
                    n_steps,
                )
                coord.units = "m"
                coord.guess_bounds()

            # Correct the latitude and longitude coordinates.
            latlon_crs = iris.coord_systems.GeogCS(EARTH_RADIUS)
            projected_crs = cube.coord(
                "projection_x_coordinate",
            ).coord_system.as_cartopy_crs()
            transformer = pyproj.Transformer.from_crs(
                crs_from=projected_crs,
                crs_to=latlon_crs.as_cartopy_crs(),
                always_xy=True,
            )

            # Update the points.
            lon_points, lat_points = transformer.transform(
                *np.meshgrid(
                    cube.coord("projection_x_coordinate").points,
                    cube.coord("projection_y_coordinate").points,
                ),
                errcheck=True,
            )
            cube.coord("latitude").points = lat_points
            cube.coord("longitude").points = lon_points
            cube.coord("latitude").coord_system = latlon_crs
            cube.coord("longitude").coord_system = latlon_crs

            # Update the bounds.
            x_bounds = np.concatenate(
                [
                    cube.coord("projection_x_coordinate").bounds[:1, 0],
                    cube.coord("projection_x_coordinate").bounds[:, 1],
                ],
            )
            y_bounds = np.concatenate(
                [
                    cube.coord("projection_y_coordinate").bounds[:1, 0],
                    cube.coord("projection_y_coordinate").bounds[:, 1],
                ],
            )
            x_size = cube.coord("projection_x_coordinate").shape[0]
            y_size = cube.coord("projection_y_coordinate").shape[0]
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
            lon_bounds, lat_bounds = transformer.transform(
                x_vertices,
                y_vertices,
                errcheck=True,
            )
            cube.coord("longitude").bounds = lon_bounds
            cube.coord("latitude").bounds = lat_bounds

        return cubes


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """Add height (2m) coordinate and correct long_name for time.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        for cube in cubes:
            add_scalar_height_coord(cube)
            if cube.coord("height").points != 2.0:
                cube.coord("height").points = np.ma.array([2.0])
            cube.coord("time").long_name = "time"

        return cubes


Pr = BaseFix
