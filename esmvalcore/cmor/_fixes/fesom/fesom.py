"""On-the-fly CMORizer for FESOM."""

import logging

import iris
import iris.util
import numpy as np
from iris import Constraint
from iris.coords import DimCoord
from iris.cube import CubeList

from esmvalcore.cmor._fixes.icon._base_fixes import NegateData
from esmvalcore.cmor._fixes.icon.icon import AllVars as AllVars_ICON
from esmvalcore.cmor._fixes.shared import (
    add_scalar_areatype_coord,
    fix_ocean_depth_coord,
)

from ._base_fixes import FesomFix

logger = logging.getLogger(__name__)


class AllVars(FesomFix, AllVars_ICON):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cubes = self.add_additional_cubes(cubes)
        cube = self.get_cube(cubes)

        # Fix time
        if self.vardef.has_coord_with_standard_name("time"):
            cube = self._fix_time(cube, cubes)

        if cube.coords(axis="Z"):
            fix_ocean_depth_coord(cube)
        elif cube.coords("level"):
            self._fix_depth(cube)
            fix_ocean_depth_coord(cube)

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

    def _fix_depth(self, cube):
        """Fix depth coordinate."""
        horizontal_grid = self.get_horizontal_grid(cube)
        depth_cube = horizontal_grid.extract_cube("depth_bnds")
        depth_dim = cube.coord_dims("level")
        cube.remove_coord("level")
        depth_coord = DimCoord(depth_cube.data, var_name="depth")
        cube.add_dim_coord(depth_coord, data_dim=depth_dim)
        cube.coord("depth").attributes = {"positive": "down"}

    def _add_coord_from_grid_file(self, cube, coord_name):
        """Add coordinate from grid file to cube.

        Note
        ----
        Assumes that the input cube has a single unnamed dimension, which will
        be used as dimension for the new coordinate.

        Parameters
        ----------
        cube: iris.cube.Cube
            ICON data to which the coordinate from the grid file is added.
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
        # Note: it might be necessary to expand this when more coord_names are
        # supported
        horizontal_grid = self.get_horizontal_grid(cube)
        if isinstance(horizontal_grid, iris.cube.CubeList):
            grid_cube = horizontal_grid.extract_cube(
                Constraint("area of grid cell"),
            )
            coord = grid_cube.coord(coord_name)
        elif coord_name == "longitude":
            coord = iris.coords.AuxCoord(
                points=(horizontal_grid.grid_center_lon.values),
                bounds=(horizontal_grid.grid_corner_lon.values),
                standard_name="longitude",
                units="degrees",
            )
        elif coord_name == "latitude":
            coord = iris.coords.AuxCoord(
                points=(horizontal_grid.grid_center_lat.values),
                bounds=(horizontal_grid.grid_corner_lat.values),
                standard_name="latitude",
                units="degrees",
            )

        # Find index of mesh dimension (= single unnamed dimension)
        n_unnamed_dimensions = cube.ndim - len(cube.dim_coords)
        if n_unnamed_dimensions != 1:
            msg = (
                f"Cannot determine coordinate dimension for coordinate "
                f"'{coord_name}', cube does not contain a single unnamed "
                f"dimension:\n{cube}"
            )
            raise ValueError(
                msg,
            )
        coord_dims = ()
        for idx in range(cube.ndim):
            if not cube.coords(dimensions=idx, dim_coords=True):
                coord_dims = (idx,)
                break

        # Adapt coordinate names so that the coordinate can be referenced with
        # 'cube.coord(coord_name)'; the exact name will be set at a later stage
        coord.standard_name = None
        coord.long_name = None  # coord_name
        coord.var_name = coord_name
        cube.add_aux_coord(coord, coord_dims)
        logger.debug("COORD ADDED FROM GRID FILE:\n%s", coord)
        logger.debug("CUBE AFTER ADDING COORD FROM GRID FILE:\n%s", cube)

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
        if self.extra_facets.get("ugrid", True):
            mesh = self.get_mesh(cube)
            cube.remove_coord("latitude")
            cube.remove_coord("longitude")
            # for i, mesh_coord in enumerate(mesh.to_MeshCoords('face')):
            for mesh_coord in mesh.to_MeshCoords("face"):
                cube.add_aux_coord(mesh_coord, mesh_idx)


Mlotst = NegateData


class Siconc(FesomFix):
    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes)
        add_scalar_areatype_coord(
            cube,
            long_name="Sea Ice area type",
            value="sea_ice",
        )
        return CubeList([cube])
