"""Fix base classes for ORAS5 on-the-fly CMORizer."""

import logging
from pathlib import Path

import dask.array as da
import iris
import numpy as np

from iris.mesh import Connectivity, MeshXY

from ..icon.icon import IconFix

logger = logging.getLogger(__name__)


class Oras5Fix(IconFix):
    """Base class for all ORAS5 fixes."""

    CACHE_DIR = Path.home() / ".esmvaltool" / "cache"
    CACHE_VALIDITY = 7 * 24 * 60 * 60  # [s]; = 1 week
    TIMEOUT = 5 * 60  # [s]; = 5 min
    GRID_FILE_ATTR = "grid_file_uri"

    def __init__(self, *args, **kwargs):
        """Initialize ORAS5 fix."""
        super().__init__(*args, **kwargs)
        self._horizontal_grids = {}
        self._meshes = {}

    def _create_mesh(self, cube):
        """Create mesh from horizontal grid file."""
        # Get coordinates
        face_lon = cube.coord("longitude")
        face_lat = cube.coord("latitude")
        node_lon = cube.coord("longitude").bounds.T.flatten()
        node_lat = cube.coord("latitude").bounds.T.flatten()

        # Make the node locations a 2D array
        nodes_flat = np.stack([node_lon, node_lat], axis=1)

        # Find the unique nodes to be able to associate them with the faces
        # Unfortunately, dask does not support the axis parameter...
        nodes_unique, indices = np.unique(
            nodes_flat, return_inverse=True, axis=0
        )

        # Get the unique nodes as dask arrays
        node_lon = da.from_array(nodes_unique[:, 0])
        node_lat = da.from_array(nodes_unique[:, 1])

        # Get dimensions (N_faces x M_nodes)
        n_faces = len(face_lat.core_points())
        n_nodes = int(len(indices) / n_faces)

        # Reshape indices to N_faces x M_nodes dask array
        indices = da.reshape(da.from_array(indices), (n_nodes, n_faces)).T

        # Create the necessary mask
        mask = da.full(da.shape(indices), False)

        # Define the connectivity
        connectivity = Connectivity(
            indices=da.ma.masked_array(indices, mask=mask),
            cf_role="face_node_connectivity",
            start_index=0,
            location_axis=0,
        )

        # Put everything together to get a U-Grid style mesh
        node_lat = iris.coords.AuxCoord(
            node_lat,
            standard_name="latitude",
            var_name="nlat",
            long_name="node latitude",
            units="degrees",
        )
        node_lon = iris.coords.AuxCoord(
            node_lon,
            standard_name="longitude",
            var_name="nlon",
            long_name="node longitude",
            units="degrees",
        )

        mesh = MeshXY(
            topology_dimension=2,
            node_coords_and_axes=[(node_lat, "y"), (node_lon, "x")],
            connectivities=[connectivity],
            face_coords_and_axes=[(face_lat, "y"), (face_lon, "x")],
        )

        return mesh

    def get_horizontal_grid(self, cube):
        """Get copy of ORAS5 horizontal grid.

        If given, retrieve grid from `horizontal_grid` facet specified by the
        user.

        Parameters
        ----------
        cube: iris.cube.Cube
            Cube for which the ORS5 horizontal grid is retrieved. If the facet
            `horizontal_grid` is not specified by the user, it raises a
            NotImplementedError.

        Returns
        -------
        iris.cube.CubeList
            Copy of ORAS5 horizontal grid.

        Raises
        ------
        FileNotFoundError
            Path specified by `horizontal_grid` facet (absolute or relative to
            `auxiliary_data_dir`) does not exist.
        NotImplementedError
            No `horizontal_grid` facet is defined.

        """
        if self.extra_facets.get("horizontal_grid") is not None:
            grid = self._get_grid_from_facet()
        else:
            raise NotImplementedError(
                "Full path to suitable ORAS5 grid must be specified in facet "
                "'horizontal_grid'"
            )

        return grid.copy()

    def _get_grid_from_facet(self):
        """Get horizontal grid from user-defined facet `horizontal_grid`."""
        grid_path = self._get_path_from_facet(
            "horizontal_grid", "Horizontal grid file"
        )
        grid_name = grid_path.name

        # If already loaded, return the horizontal grid
        if grid_name in self._horizontal_grids:
            return self._horizontal_grids[grid_name]

        # Load file
        self._horizontal_grids[grid_name] = self._load_cubes(grid_path)
        logger.debug("Loaded ORAS5 grid file from %s", grid_path)
        return self._horizontal_grids[grid_name]

    def get_mesh(self, cube):
        """Get mesh.

        Note
        ----
        If possible, this function uses a cached version of the mesh to save
        time.

        Parameters
        ----------
        cube: iris.cube.Cube
            Cube for which the mesh is retrieved.

        Returns
        -------
        iris.mesh.MeshXY
            Mesh of the cube.

        Raises
        ------
        FileNotFoundError
            Path specified by `horizontal_grid` facet (absolute or relative to
            `auxiliary_data_dir`) does not exist.
        NotImplementedError
            No `horizontal_grid` facet is defined.

        """
        # Use `horizontal_grid` facet to determine grid name
        grid_path = self._get_path_from_facet(
            "horizontal_grid", "Horizontal grid file"
        )
        grid_name = grid_path.name

        # Reuse mesh if possible
        if grid_name in self._meshes:
            logger.debug("Reusing ORAS5 mesh for grid %s", grid_name)
        else:
            logger.debug("Creating ORAS5 mesh for grid %s", grid_name)
            self._meshes[grid_name] = self._create_mesh(cube)

        return self._meshes[grid_name]
