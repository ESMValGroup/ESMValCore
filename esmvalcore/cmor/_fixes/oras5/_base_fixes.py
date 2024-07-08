"""Fix base classes for ORAS5 on-the-fly CMORizer."""

import logging
from pathlib import Path

import iris
import numpy as np
import dask.array as da
import xarray as xr
from iris import Constraint
from iris.experimental.ugrid import Connectivity, Mesh

from ..icon.icon import IconFix

logger = logging.getLogger(__name__)


class Oras5Fix(IconFix):
    """Base class for fixes."""

    CACHE_DIR = Path.home() / '.esmvaltool' / 'cache'
    CACHE_VALIDITY = 7 * 24 * 60 * 60  # [s]; = 1 week
    TIMEOUT = 5 * 60  # [s]; = 5 min
    GRID_FILE_ATTR = 'grid_file_uri'

    def __init__(self, *args, **kwargs):
        """Initialize fix."""
        super().__init__(*args, **kwargs)
        self._horizontal_grids = {}
        self._meshes = {}

    
    def _create_mesh(self, cube):
        """Create mesh from horizontal grid file.

        Note
        ----
        This functions creates a new :class:`iris.experimental.ugrid.Mesh` from
        the ``clat`` (already present in the cube), ``clon`` (already present
        in the cube), ``vertex_index``, ``vertex_of_cell``, ``vlat``, and
        ``vlon`` variables of the horizontal grid file.

        We do not use :func:`iris.experimental.ugrid.Mesh.from_coords` with the
        existing latitude and longitude coordinates here because this would
        produce lots of duplicated entries for the node coordinates. The reason
        for this is that the node coordinates are constructed from the bounds;
        since each node is contained 6 times in the bounds array (each node is
        shared by 6 neighboring cells) the number of nodes is 6 times higher
        with :func:`iris.experimental.ugrid.Mesh.from_coords` compared to using
        the information already present in the horizontal grid file.

        """

        horizontal_grid = self.get_horizontal_grid(cube)
        mesh = horizontal_grid.extract_cube(Constraint('cell_area'))
        face_lon = mesh.coord('longitude').core_points().flatten()
        face_lat = mesh.coord('latitude').core_points().flatten()

        node_lon = mesh.coord('longitude').core_bounds().flatten()
        node_lat = mesh.coord('latitude').core_bounds().flatten()

        # Make the node locations a 2D array
        nodes_flat = np.stack([node_lon, node_lat], axis=1)

        # Find the unique nodes to be able to associate them with the faces
        # Unfortunately, dask does not support the axis parameter...
        nodes_unique, indices = np.unique(nodes_flat, return_inverse=True, 
                                          axis=0)

        node_lon = da.from_array(nodes_unique[:,0])
        node_lat = da.from_array(nodes_unique[:,1])

        n_faces = len(face_lat)
        n_vertices = int(len(indices) / n_faces)

        # Reshaping to N_faces x M_nodes array
        indices = da.reshape(da.from_array(indices), (n_faces, n_vertices))

        # Add the mask, which should not have a True entry for ORAS5
        mask = da.full(da.shape(indices), False)

        ### Define the connectivity
        connectivity = Connectivity(
                    indices=da.ma.masked_array(indices,mask=mask),
                    cf_role='face_node_connectivity',
                    start_index=0,
                    location_axis=0,
        )

        face_lon = (face_lon + 360) % 360
        node_lon = (node_lon + 360) % 360

        # Put everything together to get a U-Grid style mesh
        node_lat = iris.coords.AuxCoord(node_lat, standard_name='latitude',
                                        var_name='lat', long_name='latitude',
                                        units='degrees_north')
        node_lon = iris.coords.AuxCoord(node_lon, standard_name='longitude', 
                                        var_name='lon', long_name='longitude',
                                        units='degrees_east')
        face_lat = iris.coords.AuxCoord(face_lat,  standard_name='latitude',
                                        var_name='lat', long_name='latitude',
                                        units='degrees_north')
        face_lon = iris.coords.AuxCoord(face_lon, standard_name='longitude', 
                                        var_name='lon', long_name='longitude',
                                        units='degrees_east')

        mesh = Mesh(
            topology_dimension=2,
            node_coords_and_axes=[(node_lat, 'y'), (node_lon, 'x')],
            connectivities=[connectivity],
            face_coords_and_axes=[(face_lat, 'y'), (face_lon, 'x')],
        )
        
        return mesh

    def _get_grid_from_facet(self):
        """Get horizontal grid from user-defined facet `horizontal_grid`."""
        grid_path = self._get_path_from_facet(
            'horizontal_grid', 'Horizontal grid file'
        )
        grid_name = grid_path.name

        # If already loaded, return the horizontal grid
        if grid_name in self._horizontal_grids:
            return self._horizontal_grids[grid_name]

        # Load file
        self._horizontal_grids[grid_name] = iris.load_raw(grid_path)
        # self._horizontal_grids[grid_name] = xr.open_dataset(grid_path)     
        logger.debug("Loaded ORAS5 grid file from %s", grid_path)
        return self._horizontal_grids[grid_name]