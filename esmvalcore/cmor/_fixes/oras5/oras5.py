"""On-the-fly CMORizer for ORAS5."""

import logging

import dask.array as da
import iris
import numpy as np
from iris import Constraint
from iris.coords import DimCoord
from iris.cube import CubeList

from ..icon.icon import AllVars as AllVars_ICON
from ..shared import fix_ocean_depth_coord
from ._base_fixes import Oras5Fix

logger = logging.getLogger(__name__)


class AllVars(Oras5Fix, AllVars_ICON):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cubes = self.add_additional_cubes(cubes)
        cube = self.get_cube(cubes)

        cube = self._fix_cube(cube)

        # Fix time
        if self.vardef.has_coord_with_standard_name("time"):
            cube = self._fix_time(cube, cubes)

        # Fix depth
        self._fix_depth(cube)

        # Fix latitude
        if self.vardef.has_coord_with_standard_name("latitude"):
            lat_idx = self._fix_lat(cube)
        else:
            lat_idx = None

        # Fix longitude
        if self.vardef.has_coord_with_standard_name("longitude"):
            lon_idx = self._fix_lon(cube)
        else:
            lon_idx = None

        # Fix unstructured mesh of unstructured grid if present
        if self._is_unstructured_grid(lat_idx, lon_idx):
            self._fix_mesh(cube, lat_idx)

        # Fix metadata of variable
        self.fix_var_metadata(cube)

        return CubeList([cube])

    def _fix_cube(self, cube):
        """Remove redundant cells and predetermine how to handle grid."""
        # Remove redundant cells
        cube = cube[..., :-1, 1:-1]

        # Predetermine how to handle grid
        make_unstructured = self.extra_facets.get("make_unstructured", False)
        u_grid = self.extra_facets.get("ugrid", False)

        # Grid is kept irregular and bounds are added from file
        if not make_unstructured and not u_grid:
            if "bounds" in self._horizontal_grids:
                logger.debug("Reusing lat/lon bounds.")
                lon_bounds = self._horizontal_grids["bounds"][0]
                lat_bounds = self._horizontal_grids["bounds"][1]
                cube.coord("longitude").bounds = lon_bounds
                cube.coord("latitude").bounds = lat_bounds
            else:
                mesh = self.get_horizontal_grid(cube)
                mesh = mesh.extract_cube(Constraint("cell_area"))
                lon_bounds = mesh.coord("longitude").core_bounds()
                lat_bounds = mesh.coord("latitude").core_bounds()
                lon_bounds = da.moveaxis(da.from_array(lon_bounds), -1, 0).T
                lat_bounds = da.moveaxis(da.from_array(lat_bounds), -1, 0).T
                cube.coord("longitude").bounds = lon_bounds
                cube.coord("latitude").bounds = lat_bounds
                self._horizontal_grids["bounds"] = [lon_bounds, lat_bounds]
            return cube

        # Data is made unstructured (flattened)
        coords_add = []
        for coord in cube.coords():
            if isinstance(coord, iris.coords.DimCoord):
                dim = cube.coord_dims(coord)
                coords_add.append((coord, dim))
        data = da.moveaxis(cube.core_data(), -1, -2).flatten()
        dim_shape = tuple(cube.data.shape[:-2])
        data_shape = data.shape / np.prod(dim_shape)
        data_shape = tuple(map(int, data_shape))
        data = np.reshape(data, dim_shape + data_shape)
        return iris.cube.Cube(data, dim_coords_and_dims=coords_add)

    def _add_coord_from_grid_file(self, cube, coord_name):
        """Add coordinate from grid file to cube.

        Note.
        ----
        Assumes that the input cube has a single unnamed dimension, which will
        be used as dimension for the new coordinate.

        Parameters
        ----------
        cube: iris.cube.Cube
            ORAS5 data to which the coordinate from the grid file is added.
        coord_name: str
            Name of the coordinate to add from the grid file. Must be one of
            ``'latitude'``, ``'longitude'``.

        Raises
        ------
        ValueError
            Invalid ``coord_name`` is given; input cube does not contain a
            single unnamed dimension that can be used to add the new
            coordinate.

        """
        # Use 'cell_area' as dummy cube to extract desired coordinates
        # Reuse coordinates, if already processed
        if "unstructured_" + coord_name in self._horizontal_grids:
            logger.debug("Reusing %s coordinate.", coord_name)
            coord = self._horizontal_grids["unstructured_" + coord_name]
        else:
            horizontal_grid = self.get_horizontal_grid(cube)
            grid_cube = horizontal_grid.extract_cube(Constraint("cell_area"))
            coord = grid_cube.coord(coord_name)
            points = coord.core_points().flatten()
            bounds = da.from_array(coord.core_bounds()).flatten()
            bounds = da.reshape(bounds, (int(len(bounds) / 4), 4))
            coord = iris.coords.AuxCoord(
                points=(points),
                bounds=(bounds),
                standard_name=coord_name,
                units="degrees",
            )
            self._horizontal_grids["unstructured_" + coord_name] = coord

        # Find index of mesh dimension (= single unnamed dimension)
        n_unnamed_dimensions = cube.ndim - len(cube.dim_coords)
        if n_unnamed_dimensions != 1:
            raise ValueError(
                f"Cannot determine coordinate dimension for coordinate "
                f"'{coord_name}', cube does not contain a single unnamed "
                f"dimension:\n{cube}"
            )
        coord_dims = ()
        for idx in range(cube.ndim):
            if not cube.coords(dimensions=idx, dim_coords=True):
                coord_dims = (idx,)
                break

        # Adapt coordinate names so that the coordinate can be referenced with
        # 'cube.coord(coord_name)'; the exact name will be set at a later stage
        coord.standard_name = None
        coord.long_name = coord_name
        cube.add_aux_coord(coord, coord_dims)

    def _fix_lat(self, cube):
        """Fix latitude coordinate of cube."""
        lat_name = self.extra_facets.get("latitude", "latitude")

        # Add latitude coordinate if not already present
        if not cube.coords(lat_name):
            try:
                self._add_coord_from_grid_file(cube, "latitude")
            except Exception as exc:
                msg = "Failed to add missing latitude coordinate to cube"
                raise ValueError(msg) from exc

        # Fix metadata
        lat = self.fix_lat_metadata(cube, lat_name)

        return cube.coord_dims(lat)

    def _fix_lon(self, cube):
        """Fix longitude coordinate of cube."""
        lon_name = self.extra_facets.get("longitude", "longitude")

        # Add longitude coordinate if not already present
        if not cube.coords(lon_name):
            try:
                self._add_coord_from_grid_file(cube, "longitude")
            except Exception as exc:
                msg = "Failed to add missing longitude coordinate to cube"
                raise ValueError(msg) from exc

        # Fix metadata and convert to [0, 360]
        lon = self.fix_lon_metadata(cube, lon_name)
        self._set_range_in_0_360(lon)

        return cube.coord_dims(lon)

    def _fix_time(self, cube, cubes):
        """Fix time coordinate of cube."""
        # Add time coordinate if not already present
        if not cube.coords("time"):
            cube = self._add_time(cube, cubes)

        # Fix metadata
        time_coord = self.fix_time_metadata(cube)

        # If necessary, convert invalid time units of the form "day as
        # %Y%m%d.%f" to CF format (e.g., "days since 1850-01-01")
        if "invalid_units" in time_coord.attributes:
            self._fix_invalid_time_units(time_coord)

        # If not already present, try to add bounds here. Usually bounds are
        # set in _shift_time_coord.
        self.guess_coord_bounds(cube, time_coord)

        return cube

    def _fix_mesh(self, cube, mesh_idx):
        """Fix mesh."""
        # Remove any already-present dimensional coordinate describing the mesh
        # dimension
        if cube.coords(dimensions=mesh_idx, dim_coords=True):
            cube.remove_coord(cube.coord(dimensions=mesh_idx, dim_coords=True))

        # Add dimensional coordinate that describes the mesh dimension
        index_coord = DimCoord(
            np.arange(cube.shape[mesh_idx[0]]),
            var_name="i",
            long_name=(
                "first spatial index for variables stored on an "
                "unstructured grid"
            ),
            units="1",
        )
        cube.add_dim_coord(index_coord, mesh_idx)

        # If desired, get mesh and replace the original latitude and longitude
        # coordinates with their new mesh versions
        if self.extra_facets.get("ugrid", False):
            mesh = self.get_mesh(cube)
            cube.remove_coord("latitude")
            cube.remove_coord("longitude")
            for mesh_coord in mesh.to_MeshCoords("face"):
                cube.add_aux_coord(mesh_coord, mesh_idx)

    def _fix_depth(self, cube):
        """Fix depth coordinate."""
        for i in range(len(cube.coords())):
            if "levels" in cube.coords()[i].name():
                cube.coords()[i].attributes = {"positive": "down"}

        if cube.coords(axis="Z"):
            fix_ocean_depth_coord(cube)
