"""Common fixes used for multiple datasets."""

import logging

import iris
import numpy as np
from scipy.ndimage import map_coordinates

from .fix import Fix
from .shared import add_scalar_areatype_coord, fix_bounds

logger = logging.getLogger(__name__)


class ClFixHybridPressureCoord(Fix):
    """Fixes for ``cl`` in files that do not define the coordinate correctly.

    Use this for files where the hybrid sigma pressure coordinate cannot be
    read by :mod:`iris`. The coordinates ``ap``, ``b`` and ``ps`` need to be
    added to the cube before calling :meth:`fix_metadata`.
    """

    def fix_metadata(self, cubes):
        """Add the bounds and the derived hybrid sigma pressure coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes which need to be fixed.

        Returns
        -------
        iris.cube.CubeList
        """
        cube = self.get_cube_from_list(cubes)
        fix_bounds(cube, cubes, ("ap", "b"))
        cube.add_aux_factory(
            iris.aux_factory.HybridPressureFactory(
                delta=cube.coord(var_name="ap"),
                sigma=cube.coord(var_name="b"),
                surface_air_pressure=cube.coord(var_name="ps"),
            ),
        )
        return iris.cube.CubeList([cube])


class OceanFixGrid(Fix):
    """Fixes for tos, siconc in FGOALS-g3."""

    def fix_metadata(self, cubes):
        """Fix ``latitude`` and ``longitude`` (metadata and bounds).

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        if cube.ndim != 3:
            logger.warning(
                "OceanFixGrid is designed to work on any data with an "
                "irregular ocean grid, but it was only tested on 3D (time, "
                "latitude, longitude) data so far; got %dD data",
                cube.ndim,
            )

        try:
            cube.coord("longitude", dim_coords=False)
            cube.coord("latitude", dim_coords=False)
        except iris.exceptions.CoordinateNotFoundError:
            return cubes
        # Get dimensional coordinates. Note:
        # - First dimension i -> X-direction (= longitude)
        # - Second dimension j -> Y-direction (= latitude)
        (j_dim, i_dim) = sorted(
            set(
                cube.coord_dims(cube.coord("latitude", dim_coords=False))
                + cube.coord_dims(cube.coord("longitude", dim_coords=False)),
            ),
        )
        try:
            cube.coord(dim_coords=True, dimensions=i_dim)
            cube.coord(dim_coords=True, dimensions=j_dim)
        except iris.exceptions.CoordinateNotFoundError:
            cube.add_dim_coord(
                iris.coords.DimCoord(
                    np.arange(cube.shape[i_dim]),
                    var_name="i",
                ),
                i_dim,
            )
            cube.add_dim_coord(
                iris.coords.DimCoord(
                    np.arange(cube.shape[j_dim]),
                    var_name="j",
                ),
                j_dim,
            )

        i_coord = cube.coord(dim_coords=True, dimensions=i_dim)
        j_coord = cube.coord(dim_coords=True, dimensions=j_dim)

        # Fix metadata of coordinate i
        i_coord.var_name = "i"
        i_coord.standard_name = None
        i_coord.long_name = "cell index along first dimension"
        i_coord.units = "1"
        i_coord.circular = False

        # Fix metadata of coordinate j
        j_coord.var_name = "j"
        j_coord.standard_name = None
        j_coord.long_name = "cell index along second dimension"
        j_coord.units = "1"

        # Fix points and bounds of index coordinates i and j
        for idx_coord in (i_coord, j_coord):
            idx_coord.points = np.arange(len(idx_coord.points))
            idx_coord.bounds = None
            idx_coord.guess_bounds()

        # Calculate latitude/longitude vertices by interpolation.
        # Following the CF conventions (see
        # https://cfconventions.org/cf-conventions/cf-conventions.html#cell-boundaries)
        # we go counter-clockwise around the cells and construct a grid of
        # index values which are in turn used to interpolate longitudes and
        # latitudes in the midpoints between the cell centers.
        lat_vertices = []
        lon_vertices = []
        for j, i in [(0, 0), (0, 1), (1, 1), (1, 0)]:
            (j_v, i_v) = np.meshgrid(
                j_coord.bounds[:, j],
                i_coord.bounds[:, i],
                indexing="ij",
            )
            lat_vertices.append(
                map_coordinates(
                    cube.coord("latitude").points,
                    [j_v, i_v],
                    mode="nearest",
                ),
            )
            lon_vertices.append(
                map_coordinates(
                    cube.coord("longitude").points,
                    [j_v, i_v],
                    mode="wrap",
                ),
            )
        lat_vertices = np.array(lat_vertices)
        lon_vertices = np.array(lon_vertices)
        lat_vertices = np.moveaxis(lat_vertices, 0, -1)
        lon_vertices = np.moveaxis(lon_vertices, 0, -1)

        # Copy vertices to cube
        cube.coord("latitude").bounds = lat_vertices
        cube.coord("longitude").bounds = lon_vertices

        return iris.cube.CubeList([cube])


class SiconcFixScalarCoord(Fix):
    """Fixes for siconc."""

    def fix_metadata(self, cubes):
        """Add typesi coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_areatype_coord(
            cube,
            long_name="Sea Ice area type",
            value="sea_ice",
        )
        return iris.cube.CubeList([cube])
