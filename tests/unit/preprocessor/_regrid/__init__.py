"""Unit tests for the :mod:`esmvalcore.preprocessor.regrid` module."""

from typing import Literal

import iris
import iris.fileformats
import numpy as np
from iris.coords import AuxCoord, CellMethod, DimCoord


def _make_vcoord(data, dtype=None):
    """Create a synthetic test vertical coordinate."""
    if dtype is None:
        dtype = np.int32

    if isinstance(data, int):
        data = np.arange(data, dtype=dtype)
    elif not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=dtype)

    # Create a pressure vertical coordinate.
    kwargs = {
        "standard_name": "air_pressure",
        "long_name": "Pressure",
        "var_name": "plev",
        "units": "hPa",
        "attributes": {"positive": "down"},
        "coord_system": None,
    }

    try:
        zcoord = DimCoord(data, **kwargs)
    except ValueError:
        zcoord = AuxCoord(data, **kwargs)

    return zcoord


def _make_cube(  # noqa: PLR0915,C901
    data: np.ndarray,
    aux_coord: bool = True,
    dim_coord: bool = True,
    dtype=None,
    grid: Literal["regular", "rotated", "mesh"] = "regular",
) -> iris.cube.Cube:
    """Create a 3d synthetic test cube."""
    if dtype is None:
        dtype = np.int32

    if not isinstance(data, np.ndarray):
        data = np.empty(data, dtype=dtype)

    z, y, x = data.shape
    if grid == "mesh":
        # Meshes have a single lat/lon dimension.
        data = data.reshape(z, -1)

    # Create the cube.
    cm = CellMethod(
        method="mean",
        coords="time",
        intervals="20 minutes",
        comments=None,
    )
    kwargs = {
        "standard_name": "air_temperature",
        "long_name": "Air Temperature",
        "var_name": "ta",
        "units": "K",
        "attributes": {"cube": "attribute"},
        "cell_methods": (cm,),
    }
    cube = iris.cube.Cube(data, **kwargs)

    # Create a synthetic test vertical coordinate.
    if dim_coord:
        cube.add_dim_coord(_make_vcoord(z, dtype=dtype), 0)

    if grid == "rotated":
        # Create a synthetic test latitude coordinate.
        data = np.arange(y, dtype=dtype) + 1
        cs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
        kwargs = {
            "standard_name": "grid_latitude",
            "long_name": "latitude in rotated pole grid",
            "var_name": "rlat",
            "units": "degrees",
            "attributes": {"latitude": "attribute"},
            "coord_system": cs,
        }
        ycoord = DimCoord(data, **kwargs)
        if data.size > 1:
            ycoord.guess_bounds()
        cube.add_dim_coord(ycoord, 1)

        # Create a synthetic test longitude coordinate.
        data = np.arange(x, dtype=dtype) + 1
        kwargs = {
            "standard_name": "grid_longitude",
            "long_name": "longitude in rotated pole grid",
            "var_name": "rlon",
            "units": "degrees",
            "attributes": {"longitude": "attribute"},
            "coord_system": cs,
        }
        xcoord = DimCoord(data, **kwargs)
        if data.size > 1:
            xcoord.guess_bounds()
        cube.add_dim_coord(xcoord, 2)
    elif grid == "mesh":
        # This constructs a trivial rectangular mesh with square faces:
        #   0.  1.  2.
        # 0. +---+---+-
        #    | x | x |
        # 1. +---+---+-
        #    | x | x |
        # 2. +---+---+-
        # where
        # + is a node location
        # x is a face location
        # the lines between the nodes are the boundaries of the faces
        # and the number are degrees latitude/longitude.
        #
        node_data_x = np.arange(x + 1) + 0.5
        node_data_y = np.arange(y + 1) + 0.5
        node_x, node_y = [
            AuxCoord(a.ravel(), name)
            for a, name in zip(
                np.meshgrid(node_data_x, node_data_y),
                ["longitude", "latitude"],
                strict=False,
            )
        ]
        face_data_x = np.arange(x) + 1
        face_data_y = np.arange(y) + 1
        face_x, face_y = [
            AuxCoord(a.ravel(), name)
            for a, name in zip(
                np.meshgrid(face_data_x, face_data_y),
                ["longitude", "latitude"],
                strict=False,
            )
        ]
        # Build the face connectivity indices by creating an array of squares
        # and adding an offset of 1 more to each next square and then dropping:
        # * the last column of connectivities - those would connect the last
        #   nodes in a row to the first nodes of the next row
        # * the last row of connectivities - those refer to nodes outside the
        #   grid
        n_nodes_x = len(node_data_x)
        n_nodes_y = len(node_data_y)
        square = np.array([0, n_nodes_x, n_nodes_x + 1, 1])
        connectivities = (
            (
                np.tile(square, (n_nodes_y * n_nodes_x, 1))
                + np.arange(n_nodes_y * n_nodes_x).reshape(-1, 1)
            )
            .reshape(n_nodes_y, n_nodes_x, 4)[:-1, :-1]
            .reshape(-1, 4)
        )
        face_connectivity = iris.mesh.Connectivity(
            indices=connectivities,
            cf_role="face_node_connectivity",
        )
        mesh = iris.mesh.MeshXY(
            topology_dimension=2,
            node_coords_and_axes=[(node_x, "X"), (node_y, "Y")],
            face_coords_and_axes=[(face_x, "X"), (face_y, "Y")],
            connectivities=[face_connectivity],
        )
        lon, lat = mesh.to_MeshCoords("face")
        cube.add_aux_coord(lon, 1)
        cube.add_aux_coord(lat, 1)
    elif grid == "regular":
        # Create a synthetic test latitude coordinate.
        data = np.arange(y, dtype=dtype) + 1
        cs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
        kwargs = {
            "standard_name": "latitude",
            "long_name": "Latitude",
            "var_name": "lat",
            "units": "degrees_north",
            "attributes": {"latitude": "attribute"},
            "coord_system": cs,
        }
        ycoord = DimCoord(data, **kwargs)
        if data.size > 1:
            ycoord.guess_bounds()
        cube.add_dim_coord(ycoord, 1)

        # Create a synthetic test longitude coordinate.
        data = np.arange(x, dtype=dtype) + 1
        kwargs = {
            "standard_name": "longitude",
            "long_name": "Longitude",
            "var_name": "lon",
            "units": "degrees_east",
            "attributes": {"longitude": "attribute"},
            "coord_system": cs,
        }
        xcoord = DimCoord(data, **kwargs)
        if data.size > 1:
            xcoord.guess_bounds()
        cube.add_dim_coord(xcoord, 2)

    # Create a synthetic test 2d auxiliary coordinate
    # that spans the vertical dimension.
    if aux_coord:
        hsize = y * x if grid == "mesh" else y
        data = np.arange(np.prod((z, hsize)), dtype=dtype).reshape(z, hsize)
        kwargs = {
            "long_name": "Pressure Slice",
            "var_name": "aplev",
            "units": "hPa",
            "attributes": {"positive": "down"},
        }
        zycoord = AuxCoord(data, **kwargs)
        cube.add_aux_coord(zycoord, (0, 1))

    return cube
