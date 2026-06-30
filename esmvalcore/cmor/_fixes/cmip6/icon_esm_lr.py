"""Fixes for ICON-ESM-LR model."""

import numpy as np
from iris.coords import AuxCoord
from iris.mesh import Connectivity, MeshXY

from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.iris_helpers import has_unstructured_grid

# deduplicate decimals - round cartesian coords to merge
# identical vertices (especially pole vertices)
CARTESIAN_COORDINATE_DECIMALS = 12


class AllVars(Fix):
    """Adapt the native ICON mesh fix for ICON-ESM-LR outputs.

    Like the native ICON fix (`esmvalcore.cmor._fixes.icon._base_fixes`).
    this avoids ``MeshXY.from_coords`` because shared
    polygon vertices would be duplicated. Since CMIP6 ICON-ESM-LR files don't
    have ``vertex_of_cell``, recreate the connectivity from coordinate
    bounds.
    """

    @staticmethod
    def _can_create_mesh(cube):
        # check if mesh is there
        if cube.mesh is not None:
            return False
        # unstructured?
        if not has_unstructured_grid(cube):
            return False

        lat = cube.coord("latitude")
        lon = cube.coord("longitude")

        # check bounds
        if not lat.has_bounds() or not lon.has_bounds():
            return False
        if lat.bounds.shape != lon.bounds.shape:
            return False
        return lat.bounds.ndim == 2

    @staticmethod
    def _get_node_coords_and_connectivity(lat_bounds, lon_bounds):
        """Build unique mesh nodes and face-node connectivity.

        Cell vertices are converted to cartesian coordinates to
        identify shared nodes. Duplicate vertices are
        removed and a face-node connectivity array is generated.
        """
        lat_rad = np.deg2rad(lat_bounds)
        lon_rad = np.deg2rad(lon_bounds)

        cartesian = np.stack(
            [
                np.cos(lat_rad) * np.cos(lon_rad),
                np.cos(lat_rad) * np.sin(lon_rad),
                np.sin(lat_rad),
            ],
            axis=-1,
        )

        # round coords to avoid floating point diffs
        rounded = np.round(
            cartesian.reshape(-1, 3),
            decimals=CARTESIAN_COORDINATE_DECIMALS,
        )
        # unique mesh nodes
        unique_nodes, inverse = np.unique(
            rounded,
            axis=0,
            return_inverse=True,
        )

        # we create face-node connectivity and back to lon/lat
        connectivity = inverse.reshape(lat_bounds.shape)
        norm = np.linalg.norm(unique_nodes, axis=1)
        unit_nodes = unique_nodes / norm[:, np.newaxis]
        node_lat = np.rad2deg(np.arcsin(unit_nodes[:, 2]))
        node_lon = (
            np.rad2deg(np.arctan2(unit_nodes[:, 1], unit_nodes[:, 0])) % 360.0
        )

        return node_lat, node_lon, connectivity

    def _fix_unstructured_mesh(self, cube):
        """Create and attach the Iris mesh.

        Constructs node coordinates and face-node connectivity
        from latitude longitude bounds and replaces the original
        coordinate representation with Iris mesh coordinates.
        """

        lat = cube.coord("latitude")
        lon = cube.coord("longitude")
        mesh_dim = cube.coord_dims(lat)

        # construct the face_node_connectivity from bounds
        node_lat_points, node_lon_points, face_node_connectivity = (
            self._get_node_coords_and_connectivity(lat.bounds, lon.bounds)
        )

        node_lat = AuxCoord(
            node_lat_points,
            standard_name="latitude",
            long_name="node latitude",
            var_name="nlat",
            units=lat.units,
        )
        node_lon = AuxCoord(
            node_lon_points,
            standard_name="longitude",
            long_name="node longitude",
            var_name="nlon",
            units=lon.units,
        )

        face_lat = lat.copy()
        face_lon = lon.copy()
        # Update face bounds using the deduplicated node coords.
        face_lat.bounds = node_lat.points[face_node_connectivity]
        face_lon.bounds = node_lon.points[face_node_connectivity]

        # Same create mesh logic with native ICON fix (Iris mesh object).
        connectivity = Connectivity(
            indices=face_node_connectivity,
            cf_role="face_node_connectivity",
            start_index=0,
            location_axis=0,
        )
        mesh = MeshXY(
            topology_dimension=2,
            node_coords_and_axes=[(node_lat, "y"), (node_lon, "x")],
            face_coords_and_axes=[(face_lat, "y"), (face_lon, "x")],
            connectivities=[connectivity],
        )

        cube.remove_coord("latitude")
        cube.remove_coord("longitude")
        for mesh_coord in mesh.to_MeshCoords("face"):
            cube.add_aux_coord(mesh_coord, mesh_dim)

    def fix_metadata(self, cubes):
        """Rename ``var_name`` of latitude and longitude.

        Parameters
        ----------
        cubes: iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
            Fixed cubes.

        """
        varnames_to_change = {
            "latitude": "lat",
            "longitude": "lon",
        }

        for cube in cubes:
            for std_name, var_name in varnames_to_change.items():
                if cube.coords(std_name):
                    cube.coord(std_name).var_name = var_name
            if self._can_create_mesh(cube):
                self._fix_unstructured_mesh(cube)

        return cubes
