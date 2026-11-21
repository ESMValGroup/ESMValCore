"""Fix base classes for FESOM on-the-fly CMORizer."""

import logging
from pathlib import Path

import dask.array as da
import iris
import numpy as np
import xarray as xr
from iris import Constraint
from iris.mesh import Connectivity, MeshXY

from esmvalcore.cmor._fixes.icon.icon import IconFix

logger = logging.getLogger(__name__)


class FesomFix(IconFix):
    """Base class for all FESOM fixes."""

    CACHE_DIR = Path.home() / ".esmvaltool" / "cache"
    CACHE_VALIDITY = 7 * 24 * 60 * 60  # [s]; = 1 week
    TIMEOUT = 5 * 60  # [s]; = 5 min
    GRID_FILE_ATTR = "grid_file_uri"

    def __init__(self, *args, **kwargs):
        """Initialize FESOM fix."""
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

        if isinstance(horizontal_grid, iris.cube.CubeList):
            mesh = horizontal_grid.extract_cube(
                Constraint("area of grid cell"),
            )
            face_lon = mesh.coord("longitude").core_points()
            face_lat = mesh.coord("latitude").core_points()

            node_lon = mesh.coord("longitude").bounds
            node_lat = mesh.coord("latitude").bounds
        else:  # discard?
            face_lon = horizontal_grid["grid_center_lon"].values
            face_lat = horizontal_grid["grid_center_lat"].values
            node_lon = horizontal_grid["grid_corner_lon"].values
            node_lat = horizontal_grid["grid_corner_lat"].values

        # Make the node locations a 2D array
        nodes_flat = np.stack([node_lon.flatten(), node_lat.flatten()], axis=1)

        # Find the unique nodes to be able to associate them with the faces
        # Unfortunately, dask does not support the axis parameter...
        nodes_unique, indices = np.unique(
            nodes_flat,
            return_inverse=True,
            axis=0,
        )

        node_lon = nodes_unique[:, 0]
        node_lat = nodes_unique[:, 1]

        n_faces = len(face_lat)
        n_vertices = int(len(indices) / n_faces)

        indices = np.reshape(indices, (n_faces, n_vertices))

        # Masking doubled node entries for the faces
        duplicates = indices[:, :, None] == indices[:, None, :]
        tri_mask = np.tri(n_vertices, k=-1, dtype=bool)
        mask = np.any(duplicates & tri_mask[None, :, :], axis=2)

        # Define the connectivity
        connectivity = Connectivity(
            indices=da.ma.masked_array(indices, mask=mask),
            cf_role="face_node_connectivity",
            start_index=0,
            location_axis=0,
        )

        face_lon = (face_lon + 360) % 360
        node_lon = da.from_array((node_lon + 360) % 360)

        # Put everything together to get a U-Grid style mesh
        node_lat = iris.coords.AuxCoord(
            node_lat,
            standard_name="latitude",
            var_name="lat",
            long_name="latitude",
            units="degrees_north",
        )
        node_lon = iris.coords.AuxCoord(
            node_lon,
            standard_name="longitude",
            var_name="lon",
            long_name="longitude",
            units="degrees_east",
        )
        face_lat = iris.coords.AuxCoord(
            face_lat,
            standard_name="latitude",
            var_name="lat",
            long_name="latitude",
            units="degrees_north",
        )
        face_lon = iris.coords.AuxCoord(
            face_lon,
            standard_name="longitude",
            var_name="lon",
            long_name="longitude",
            units="degrees_east",
        )

        return MeshXY(
            topology_dimension=2,
            node_coords_and_axes=[(node_lat, "y"), (node_lon, "x")],
            connectivities=[connectivity],
            face_coords_and_axes=[(face_lat, "y"), (face_lon, "x")],
        )

    def _get_grid_from_facet(self):
        """Get horizontal grid from user-defined facet `horizontal_grid`."""
        grid_path = self._get_path_from_facet(
            "horizontal_grid",
            "Horizontal grid file",
        )
        grid_name = grid_path.name

        # If already loaded, return the horizontal grid
        if grid_name in self._horizontal_grids:
            return self._horizontal_grids[grid_name]

        # Load file
        try:
            self._horizontal_grids[grid_name] = self._load_cubes(grid_path)
            (
                self._horizontal_grids[grid_name]  # noqa: B018
                .extract_cube(Constraint("area of grid cell"))
                .coord("longitude")
                .points
            )
        except iris.exceptions.CoordinateNotFoundError:
            self._horizontal_grids[grid_name] = xr.open_dataset(grid_path)

        logger.debug("Loaded FESOM grid file from %s", grid_path)
        return self._horizontal_grids[grid_name]
