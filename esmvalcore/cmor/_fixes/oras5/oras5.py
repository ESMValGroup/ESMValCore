"""On-the-fly CMORizer for ORAS5."""

import logging

import iris
import iris.util
import numpy as np
import dask.array as da
from iris import Constraint
from iris.coords import DimCoord
from iris.cube import CubeList

from ..shared import fix_ocean_depth_coord

from ._base_fixes import Oras5Fix
from ..icon.icon import AllVars as AllVars_ICON

logger = logging.getLogger(__name__)


class AllVars(Oras5Fix, AllVars_ICON):
    """Fixes for all variables."""
    
    def fix_metadata(self, cubes):

        """Fix metadata."""
        cubes = self.add_additional_cubes(cubes)
        cube = self.get_cube(cubes)

        # This is just a quick solution for other than horizontal coordinates,
        # needs to be adapted to also deal with depth.
        time = cube.coord('time')

        # Adding the option to make the irregular (2d) grid unstructured (1d)
        # to take advantage of UGRID
        if self.extra_facets.get('make_unstructured', True):
            # ORAS5 has 1 redundant row and 2 redundant columns that need to be
            # removed.
            data = cube.core_data()[...,:-1,1:-1].T.flatten()
            data = da.reshape(data, (len(time.points), len(data)))
            lat_points = cube.coord('latitude').core_points()
            lat_points = lat_points[:-1,1:-1].T.flatten()
            lon_points = cube.coord('longitude').core_points()
            lon_points = lon_points[:-1,1:-1].T.flatten()

            lat_coord = iris.coords.AuxCoord(lat_points, 
                                             standard_name='latitude',
                                             units=cube.coord('latitude').units)
            lon_coord = iris.coords.AuxCoord(lon_points, 
                                             standard_name='longitude',
                                             units=cube.coord('longitude').units)
            
            # See above concerning additional coordinates and dimensions
            new_cube = iris.cube.Cube(data, dim_coords_and_dims=[(time,0)])
            new_cube.add_aux_coord(lat_coord, 1)
            new_cube.add_aux_coord(lon_coord, 1)

            new_cube.long_name = cube.long_name
            cube = new_cube

        else:
            # ORAS5 has 1 redundant row and 2 redundant columns that need to be
            # removed.
            cube = cube[...,:-1,1:-1]
            lon_shape = cube.coord('longitude').points.shape
            mesh = self.get_horizontal_grid(cube)
            mesh = mesh.extract_cube(Constraint('cell_area'))
            lon_bnds = mesh.coord('longitude').bounds
            lat_bnds = mesh.coord('latitude').bounds
            lon_bnds = np.reshape(lon_bnds, (lon_shape[0], lon_shape[1],
                                             min(lon_bnds.shape)))
            lat_bnds = np.reshape(lat_bnds, (lon_shape[0], lon_shape[1],
                                             min(lat_bnds.shape)))
            cube.coord('longitude').bounds = lon_bnds
            cube.coord('latitude').bounds = lat_bnds

        # Fix time
        if self.vardef.has_coord_with_standard_name('time'):
            cube = self._fix_time(cube, cubes)

        if cube.coords(axis='Z'):
            fix_ocean_depth_coord(cube)

        # Fix latitude
        if self.vardef.has_coord_with_standard_name('latitude'):
            lat_idx = self._fix_lat(cube)
        else:
            lat_idx = None

        # Fix longitude
        if self.vardef.has_coord_with_standard_name('longitude'):
            lon_idx = self._fix_lon(cube)
        else:
            lon_idx = None

        # Fix unstructured mesh of unstructured grid if present
        if self._is_unstructured_grid(lat_idx, lon_idx):
            self._fix_mesh(cube, lat_idx)

        # Fix metadata of variable
        self.fix_var_metadata(cube)

        return CubeList([cube])

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
        if type(horizontal_grid) == iris.cube.CubeList:
            grid_cube = horizontal_grid.extract_cube(
                Constraint('cell_area'))
            coord = grid_cube.coord(coord_name)
        else:
            if coord_name == 'longitude':
                coord = iris.coords.AuxCoord(
                                    points = (horizontal_grid.grid_center_lon
                                              .values), 
                                    bounds = (horizontal_grid.grid_corner_lon
                                              .values),
                                    standard_name = 'longitude',
                                    units = 'degrees')
            elif coord_name == 'latitude':
                coord = iris.coords.AuxCoord(
                                    points = (horizontal_grid.grid_center_lat
                                              .values), 
                                    bounds = (horizontal_grid.grid_corner_lat
                                              .values),
                                    standard_name = 'latitude',
                                    units = 'degrees')

        # Find index of mesh dimension (= single unnamed dimension)
        n_unnamed_dimensions = cube.ndim - len(cube.dim_coords)
        if n_unnamed_dimensions != 1:
            raise ValueError(
                f"Cannot determine coordinate dimension for coordinate "
                f"'{coord_name}', cube does not contain a single unnamed "
                f"dimension:\n{cube}")
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

    def _fix_mesh(self, cube, mesh_idx):
        """Fix mesh."""
        # Remove any already-present dimensional coordinate describing the mesh
        # dimension
        if cube.coords(dimensions=mesh_idx, dim_coords=True):
            cube.remove_coord(cube.coord(dimensions=mesh_idx, dim_coords=True))

        # Add dimensional coordinate that describes the mesh dimension
        index_coord = DimCoord(
            np.arange(cube.shape[mesh_idx[0]]),
            var_name='i',
            long_name=('first spatial index for variables stored on an '
                       'unstructured grid'),
            units='1',
        )
        cube.add_dim_coord(index_coord, mesh_idx)

        # If desired, get mesh and replace the original latitude and longitude
        # coordinates with their new mesh versions
        if self.extra_facets.get('ugrid', True):
            mesh = self.get_mesh(cube)
            cube.remove_coord('latitude')
            cube.remove_coord('longitude')
            for mesh_coord in mesh.to_MeshCoords('face'):
                cube.add_aux_coord(mesh_coord, mesh_idx)
