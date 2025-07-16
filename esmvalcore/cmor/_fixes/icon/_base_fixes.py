"""Fix base classes for ICON on-the-fly CMORizer."""

from __future__ import annotations

import logging
import os
import shutil
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import urlparse

import iris
import numpy as np
import pandas as pd
import requests
from cf_units import Unit
from iris import NameConstraint
from iris.coords import AuxCoord, Coord, DimCoord
from iris.cube import Cube, CubeList
from iris.mesh import Connectivity, MeshXY

from esmvalcore.cmor._fixes.native_datasets import NativeDatasetFix
from esmvalcore.iris_helpers import add_leading_dim_to_cube, date2num
from esmvalcore.local import _get_data_sources

logger = logging.getLogger(__name__)


class IconFix(NativeDatasetFix):
    """Base class for all ICON fixes.

    This includes code necessary to handle ICON's horizontal and vertical grid.

    """

    CACHE_DIR = Path.home() / ".esmvaltool" / "cache"
    CACHE_VALIDITY = 7 * 24 * 60 * 60  # [s]; = 1 week
    TIMEOUT = 5 * 60  # [s]; = 5 min
    GRID_FILE_ATTR = "grid_file_uri"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize ICON fix."""
        super().__init__(*args, **kwargs)
        self._horizontal_grids: dict[str, CubeList] = {}
        self._meshes: dict[str, MeshXY] = {}

    def _create_mesh(self, cube: Cube) -> MeshXY:
        """Create mesh from horizontal grid file.

        Note
        ----
        This functions creates a new :class:`iris.mesh.MeshXY` from the
        ``clat`` (already present in the cube), ``clon`` (already present in
        the cube), ``vertex_index``, ``vertex_of_cell``, ``vlat``, and ``vlon``
        variables of the horizontal grid file.

        We do not use :func:`iris.mesh.MeshXY.from_coords` with the existing
        latitude and longitude coordinates here because this would produce lots
        of duplicated entries for the node coordinates. The reason for this is
        that the node coordinates are constructed from the bounds; since each
        node is contained 6 times in the bounds array (each node is shared by 6
        neighboring cells) the number of nodes is 6 times higher with
        :func:`iris.mesh.MeshXY.from_coords` compared to using the information
        already present in the horizontal grid file.

        """
        horizontal_grid = self.get_horizontal_grid(cube)

        # Extract connectivity (i.e., the mapping cell faces -> cell nodes)
        # from the the horizontal grid file (in ICON jargon called
        # 'vertex_of_cell'; since UGRID expects a different dimension ordering
        # we transpose the cube here)
        vertex_of_cell = horizontal_grid.extract_cube(
            NameConstraint(var_name="vertex_of_cell"),
        ).copy()
        vertex_of_cell.transpose()

        # Extract start index used to name nodes from the the horizontal grid
        # file
        start_index = self._get_start_index(horizontal_grid)

        # Extract face coordinates from cube (in ICON jargon called 'cell
        # latitude' and 'cell longitude')
        face_lat = cube.coord("latitude")
        face_lon = cube.coord("longitude")

        # Extract node coordinates from horizontal grid
        (node_lat, node_lon) = self._get_node_coords(horizontal_grid)

        # The bounds given by the face coordinates slightly differ from the
        # bounds determined by the connectivity. We arbitrarily assume here
        # that the information given by the connectivity is correct.
        conn_node_inds = vertex_of_cell.data - start_index

        # Latitude: there might be slight numerical differences (-> check that
        # the differences are very small before fixing it)
        close_kwargs = {"rtol": 1e-3, "atol": 1e-5}
        if not np.allclose(
            face_lat.bounds,
            node_lat.points[conn_node_inds],
            **close_kwargs,  # type: ignore
        ):
            logger.warning(
                "Latitude bounds of the face coordinate ('clat_vertices' in "
                "the grid file) differ from the corresponding values "
                "calculated from the connectivity ('vertex_of_cell') and the "
                "node coordinate ('vlat'). Using bounds defined by "
                "connectivity.",
            )
        face_lat.bounds = node_lat.points[conn_node_inds]

        # Longitude: there might be differences at the poles, where the
        # longitude information does not matter (-> check that the only large
        # differences are located at the poles). In addition, values might
        # differ by 360°, which is also okay.
        face_lon_bounds_to_check = face_lon.bounds % 360
        node_lon_conn_to_check = node_lon.points[conn_node_inds] % 360
        idx_notclose = ~np.isclose(  # type: ignore
            face_lon_bounds_to_check,
            node_lon_conn_to_check,
            **close_kwargs,
        )
        if not np.allclose(np.abs(face_lat.bounds[idx_notclose]), 90.0):
            logger.warning(
                "Longitude bounds of the face coordinate ('clon_vertices' in "
                "the grid file) differ from the corresponding values "
                "calculated from the connectivity ('vertex_of_cell') and the "
                "node coordinate ('vlon'). Note that these values are allowed "
                "to differ by 360° or at the poles of the grid. Using bounds "
                "defined by connectivity.",
            )
        face_lon.bounds = node_lon.points[conn_node_inds]

        # Create mesh
        connectivity = Connectivity(
            indices=vertex_of_cell.data,
            cf_role="face_node_connectivity",
            start_index=start_index,
            location_axis=0,
        )
        return MeshXY(
            topology_dimension=2,
            node_coords_and_axes=[(node_lat, "y"), (node_lon, "x")],
            connectivities=[connectivity],
            face_coords_and_axes=[(face_lat, "y"), (face_lon, "x")],
        )

    def _get_grid_url(self, cube: Cube) -> tuple[str, str]:
        """Get ICON grid URL from cube."""
        if self.GRID_FILE_ATTR not in cube.attributes:
            msg = (
                f"Cube does not contain the attribute '{self.GRID_FILE_ATTR}' "
                f"necessary to download the ICON horizontal grid file:\n"
                f"{cube}"
            )
            raise ValueError(
                msg,
            )
        grid_url = cube.attributes[self.GRID_FILE_ATTR]
        parsed_url = urlparse(grid_url)
        grid_name = Path(parsed_url.path).name
        return (grid_url, grid_name)

    def _get_node_coords(
        self,
        horizontal_grid: CubeList,
    ) -> tuple[Coord, Coord]:
        """Get node coordinates from horizontal grid.

        Extract node coordinates from dummy variable 'dual_area' in horizontal
        grid file (in ICON jargon called 'vertex latitude' and 'vertex
        longitude'), remove their bounds (not accepted by UGRID), and adapt
        metadata.

        """
        dual_area_cube = horizontal_grid.extract_cube(
            NameConstraint(var_name="dual_area"),
        )
        node_lat = dual_area_cube.coord(var_name="vlat").copy()
        node_lon = dual_area_cube.coord(var_name="vlon").copy()

        # Fix metadata
        node_lat.bounds = None
        node_lon.bounds = None
        node_lat.var_name = "nlat"
        node_lon.var_name = "nlon"
        node_lat.standard_name = "latitude"
        node_lon.standard_name = "longitude"
        node_lat.long_name = "node latitude"
        node_lon.long_name = "node longitude"
        node_lat.convert_units("degrees_north")
        node_lon.convert_units("degrees_east")

        # Convert longitude to [0, 360]
        self._set_range_in_0_360(node_lon)

        return (node_lat, node_lon)

    def _get_path_from_facet(
        self,
        facet: str,
        description: str | None = None,
    ) -> Path:
        """Try to get path from facet."""
        if description is None:
            description = "File"
        path = Path(os.path.expandvars(self.extra_facets[facet])).expanduser()
        if not path.is_file() and self.session is not None:
            new_path = self.session["auxiliary_data_dir"] / path
            if not new_path.is_file():
                msg = (
                    f"{description} '{path}' given by facet '{facet}' does "
                    f"not exist (specify a valid absolute path or a path "
                    f"relative to the auxiliary_data_dir "
                    f"'{self.session['auxiliary_data_dir']}')"
                )
                raise FileNotFoundError(
                    msg,
                )
            path = new_path
        return path

    def add_additional_cubes(self, cubes: CubeList) -> CubeList:
        """Add additional user-defined cubes to list of cubes (in-place).

        An example use case is adding a vertical coordinate (e.g., `zg`) to the
        dataset if the vertical coordinate data is stored in a separate ICON
        output file.

        Currently, the following cubes can be added:

        - `zg` (`geometric_height_at_full_level_center`) from facet `zg_file`.
          This can be used as vertical coordinate.
        - `zghalf` (`geometric_height_at_half_level_center`) from facet
          `zghalf_file`. This can be used as bounds for the vertical
          coordinate.

        Note
        ----
        Files can be specified as absolute or relative (to the configuration
        option ``auxiliary_data_dir``) paths.

        Parameters
        ----------
        cubes:
            Input cubes which will be modified in place.

        Returns
        -------
        iris.cube.CubeList
            Modified cubes. The cubes are modified in place; they are just
            returned out of convenience for easy access.

        Raises
        ------
        InputFilesNotFound
            A specified file does not exist.

        """
        facets_to_consider = [
            "zg_file",
            "zghalf_file",
        ]
        for facet in facets_to_consider:
            if self.extra_facets.get(facet) is None:
                continue
            path_to_add = self._get_path_from_facet(facet)
            logger.debug("Adding cubes from %s", path_to_add)
            new_cubes = self._load_cubes(path_to_add)
            cubes.extend(new_cubes)

        return cubes

    def _get_grid_from_facet(self) -> CubeList:
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
        self._horizontal_grids[grid_name] = self._load_cubes(grid_path)
        logger.debug("Loaded ICON grid file from %s", grid_path)
        return self._horizontal_grids[grid_name]

    @staticmethod
    def _tmp_local_file(local_file: Path) -> Path:
        """Return the path to a temporary local file for downloading to."""
        with NamedTemporaryFile(prefix=f"{local_file}.") as file:
            return Path(file.name)

    def _get_grid_from_cube_attr(self, cube: Cube) -> Cube:
        """Get horizontal grid from `grid_file_uri` attribute of cube."""
        (grid_url, grid_name) = self._get_grid_url(cube)

        # If already loaded, return the horizontal grid
        if grid_name in self._horizontal_grids:
            return self._horizontal_grids[grid_name]

        # First, check if the grid file is available in the ICON rootpath
        grid = self._get_grid_from_rootpath(grid_name)

        # Second, if that didn't work, try to download grid (or use cached
        # version of it if possible)
        if grid is None:
            grid = self._get_downloaded_grid(grid_url, grid_name)

        # Cache grid for later use
        self._horizontal_grids[grid_name] = grid

        return grid

    def _get_grid_from_rootpath(self, grid_name: str) -> CubeList | None:
        """Try to get grid from the ICON rootpath."""
        glob_patterns: list[Path] = []
        for data_source in _get_data_sources("ICON"):
            glob_patterns.extend(
                data_source.get_glob_patterns(**self.extra_facets),
            )
        possible_grid_paths = [d.parent / grid_name for d in glob_patterns]
        for grid_path in possible_grid_paths:
            if grid_path.is_file():
                logger.debug("Using ICON grid file '%s'", grid_path)
                return self._load_cubes(grid_path)
        return None

    def _get_downloaded_grid(self, grid_url: str, grid_name: str) -> CubeList:
        """Get downloaded horizontal grid.

        Check if grid file has recently been downloaded. If not, download grid
        file here.

        Note
        ----
        In order to make this function thread-safe, the downloaded grid file is
        first saved to a temporary location, then copied to the actual location
        later.

        """
        grid_path = self.CACHE_DIR / grid_name

        # Check cache
        valid_cache = False
        if grid_path.exists():
            mtime = grid_path.stat().st_mtime
            now = datetime.now().timestamp()
            age = now - mtime
            if age < self.CACHE_VALIDITY:
                logger.debug("Using cached ICON grid file '%s'", grid_path)
                valid_cache = True
            else:
                logger.debug(
                    "Existing cached ICON grid file '%s' is outdated",
                    grid_path,
                )

        # File is not present in cache or too old -> download it
        if not valid_cache:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp_path = self._tmp_local_file(grid_path)
            logger.debug(
                "Attempting to download ICON grid file from '%s' to '%s'",
                grid_url,
                tmp_path,
            )
            with requests.get(
                grid_url,
                stream=True,
                timeout=self.TIMEOUT,
            ) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as file:
                    copyfileobj(response.raw, file)
            shutil.move(tmp_path, grid_path)
            logger.info(
                "Successfully downloaded ICON grid file from '%s' to '%s' "
                "and moved it to '%s'",
                grid_url,
                tmp_path,
                grid_path,
            )

        return self._load_cubes(grid_path)

    def get_horizontal_grid(self, cube: Cube) -> CubeList:
        """Get ICON horizontal grid.

        If given, retrieve grid from `horizontal_grid` facet specified by the
        user. Otherwise, try to download the file from the location given by
        the global attribute `grid_file_uri` of the cube.

        Note
        ----
        If necessary, this functions downloads the grid file to a cache
        directory. The downloaded file is valid for 7 days until it is
        downloaded again.

        Parameters
        ----------
        cube:
            Cube for which the ICON horizontal grid is retrieved. If the facet
            `horizontal_grid` is not specified by the user, this cube needs to
            have a global attribute `grid_file_uri` that specifies the download
            location of the ICON horizontal grid file.

        Returns
        -------
        iris.cube.CubeList
            ICON horizontal grid.

        Raises
        ------
        FileNotFoundError
            Path specified by `horizontal_grid` facet (absolute or relative to
            `auxiliary_data_dir`) does not exist.
        ValueError
            Input cube does not contain the necessary attribute `grid_file_uri`
            that specifies the download location of the ICON horizontal grid
            file.

        """
        if self.extra_facets.get("horizontal_grid") is not None:
            grid = self._get_grid_from_facet()
        else:
            grid = self._get_grid_from_cube_attr(cube)

        return grid

    def get_mesh(self, cube: Cube) -> MeshXY:
        """Get mesh.

        Note
        ----
        If possible, this function uses a cached version of the mesh to save
        time.

        Parameters
        ----------
        cube:
            Cube for which the mesh is retrieved. If the facet
            `horizontal_grid` is not specified by the user, this cube needs to
            have the global attribute `grid_file_uri` that specifies the
            download location of the ICON horizontal grid file.

        Returns
        -------
        iris.mesh.MeshXY
            Mesh of the cube.

        Raises
        ------
        FileNotFoundError
            Path specified by `horizontal_grid` facet (absolute or relative to
            `auxiliary_data_dir`) does not exist.
        ValueError
            Input cube does not contain the necessary attribute `grid_file_uri`
            that specifies the download location of the ICON horizontal grid
            file.

        """
        # If specified by the user, use `horizontal_grid` facet to determine
        # grid name; otherwise, use the `grid_file_uri` attribute of the cube
        if self.extra_facets.get("horizontal_grid") is not None:
            grid_path = self._get_path_from_facet(
                "horizontal_grid",
                "Horizontal grid file",
            )
            grid_name = grid_path.name
        else:
            (_, grid_name) = self._get_grid_url(cube)

        # Reuse mesh if possible
        if grid_name in self._meshes:
            logger.debug("Reusing ICON mesh for grid %s", grid_name)
        else:
            logger.debug("Creating ICON mesh for grid %s", grid_name)
            self._meshes[grid_name] = self._create_mesh(cube)

        return self._meshes[grid_name]

    @staticmethod
    def _get_start_index(horizontal_grid: CubeList) -> np.int32:
        """Get start index used to name nodes from horizontal grid.

        Extract start index used to name nodes from the the horizontal grid
        file (in ICON jargon called 'vertex_index').

        Note
        ----
        UGRID expects this to be a int32.

        """
        vertex_index = horizontal_grid.extract_cube(
            NameConstraint(var_name="vertex_index"),
        )
        return np.int32(np.min(vertex_index.data))

    @staticmethod
    def _load_cubes(path: Path | str) -> CubeList:
        """Load cubes and ignore certain warnings."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Ignoring netCDF variable .* invalid units .*",
                category=UserWarning,
                module="iris",
            )  # iris < 3.8
            warnings.filterwarnings(
                "ignore",
                message="Ignoring invalid units .* on netCDF variable .*",
                category=UserWarning,
                module="iris",
            )  # iris >= 3.8
            warnings.filterwarnings(
                "ignore",
                message="Failed to create 'height' dimension coordinate: The "
                "'height' DimCoord bounds array must be strictly "
                "monotonic.",
                category=UserWarning,
                module="iris",
            )
            return iris.load(path)

    @staticmethod
    def _set_range_in_0_360(lon_coord: Coord) -> None:
        """Convert longitude coordinate to [0, 360]."""
        lon_coord.points = (lon_coord.core_points() + 360.0) % 360.0
        if lon_coord.has_bounds():
            lon_coord.bounds = (lon_coord.core_bounds() + 360.0) % 360.0


class AllVarsBase(IconFix):
    """Fixes necessary for all ICON variables."""

    DEFAULT_PFULL_VAR_NAME = "pfull"

    def fix_metadata(self, cubes: CubeList) -> CubeList:
        """Fix metadata."""
        cubes = self.add_additional_cubes(cubes)
        cube = self.get_cube(cubes)

        # Fix time
        if self.vardef.has_coord_with_standard_name("time"):
            cube = self._fix_time(cube, cubes)

        # Fix height (note: cannot use "if 'height' in self.vardef.dimensions"
        # here since the name of the z-coord varies from variable to variable)
        if cube.coords("height"):
            # In case a scalar height is required, remove it here (it is added
            # at a later stage). The step _fix_height() is designed to fix
            # non-scalar height coordinates.
            if cube.coord("height").shape[0] == 1 and (
                "height2m" in self.vardef.dimensions
                or "height10m" in self.vardef.dimensions
            ):
                # If height is a dimensional coordinate with length 1, squeeze
                # the cube.
                # Note: iris.util.squeeze is not used here since it might
                # accidentally squeeze other dimensions.
                if cube.coords("height", dim_coords=True):
                    slices: list[slice | int] = [slice(None)] * cube.ndim
                    slices[cube.coord_dims("height")[0]] = 0
                    cube = cube[tuple(slices)]
                cube.remove_coord("height")
            else:
                cube = self._fix_height(cube, cubes)

        # Fix depth of ocean data
        if cube.coords(long_name="depth_below_sea"):
            self._fix_depth(cube, cubes)

        # Remove undesired lev coordinate with length 1
        lev_coord = DimCoord(0.0, var_name="lev")
        if cube.coords(lev_coord, dim_coords=True):
            slices = [slice(None)] * cube.ndim
            slices[cube.coord_dims(lev_coord)[0]] = 0
            cube = cube[tuple(slices)]
            cube.remove_coord(lev_coord)

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
            self._fix_mesh(cube, lat_idx)  # type: ignore

        # Fix scalar coordinates
        self.fix_scalar_coords(cube)

        # Fix metadata of variable
        self.fix_var_metadata(cube)

        return CubeList([cube])

    def _add_coord_from_grid_file(self, cube: Cube, coord_name: str) -> None:
        """Add coordinate from ``cell_area`` variable of grid file to cube.

        Note
        ----
        Assumes that the input cube has a single unnamed dimension, which will
        be used as dimension for the new coordinate.

        Parameters
        ----------
        cube:
            ICON data to which the coordinate from the grid file is added.
        coord_name:
            Variable name of the coordinate to add from the grid file.

        Raises
        ------
        ValueError
            Input cube does not contain a single unnamed dimension that can be
            used to add the new coordinate.

        """
        # Use 'cell_area' as dummy cube to extract desired coordinates
        # Note: it might be necessary to expand this in the future; currently
        # this only works for clat and clon
        horizontal_grid = self.get_horizontal_grid(cube)
        grid_cube = horizontal_grid.extract_cube(
            NameConstraint(var_name="cell_area"),
        )
        coord = grid_cube.coord(var_name=coord_name).copy()

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
        coord_dims: tuple[()] | tuple[int] = ()
        for idx in range(cube.ndim):
            if not cube.coords(dimensions=idx, dim_coords=True):
                coord_dims = (idx,)
                break

        # Adapt coordinate names so that the coordinate can be referenced with
        # 'cube.coord(coord_name)'; the exact name will be set at a later stage
        coord.standard_name = None
        coord.long_name = None
        cube.add_aux_coord(coord, coord_dims)

    def _add_time(self, cube: Cube, cubes: CubeList) -> Cube:
        """Add time coordinate from other cube in cubes."""
        # Try to find time cube from other cubes and it to target cube
        for other_cube in cubes:
            if not other_cube.coords("time"):
                continue
            time_coord = other_cube.coord("time")
            return add_leading_dim_to_cube(cube, time_coord)
        msg = (
            f"Cannot add required coordinate 'time' to variable "
            f"'{self.vardef.short_name}', cube and other cubes in file do not "
            f"contain it"
        )
        raise ValueError(
            msg,
        )

    def _get_z_coord(
        self,
        cubes: CubeList,
        points_name: str,
        bounds_name: str | None = None,
    ) -> AuxCoord:
        """Get z-coordinate without metadata (reversed)."""
        points_cube = iris.util.reverse(
            cubes.extract_cube(NameConstraint(var_name=points_name)),
            "height",
        )
        points = points_cube.core_data()

        # Get bounds if possible
        if bounds_name is not None:
            bounds_cube = iris.util.reverse(
                cubes.extract_cube(NameConstraint(var_name=bounds_name)),
                "height",
            )
            bounds = bounds_cube.core_data()
            bounds = np.stack(
                (bounds[..., :-1, :], bounds[..., 1:, :]),
                axis=-1,
            )
        else:
            bounds = None

        return AuxCoord(
            points,
            bounds=bounds,
            units=points_cube.units,
        )

    def _fix_height(self, cube: Cube, cubes: CubeList) -> Cube:
        """Fix height coordinate of cube."""
        # Reverse entire cube along height axis so that index 0 is surface
        # level
        cube = iris.util.reverse(cube, "height")

        # If possible, extract reversed air_pressure coordinate from list of
        # cubes and add it to cube
        # Note: pfull/phalf have dimensions (time, height, spatial_dim)
        pfull_var = self.extra_facets.get(
            "pfull_var",
            self.DEFAULT_PFULL_VAR_NAME,
        )
        phalf_var = self.extra_facets.get("phalf_var", "phalf")
        if cubes.extract(NameConstraint(var_name=pfull_var)):
            if cubes.extract(NameConstraint(var_name=phalf_var)):
                phalf = phalf_var
            else:
                phalf = None
            plev_coord = self._get_z_coord(cubes, pfull_var, bounds_name=phalf)
            self.fix_plev_metadata(cube, plev_coord)
            cube.add_aux_coord(plev_coord, np.arange(cube.ndim))
        else:
            logger.debug(
                "Cannot add pressure level information to ICON data; "
                "variables '%s' and/or '%s' are not available",
                pfull_var,
                phalf_var,
            )

        # If possible, extract reversed altitude coordinate from list of cubes
        # and add it to cube
        # Note: zg/zghalf have dimensions (height, spatial_dim)
        zgfull_var = self.extra_facets.get("zgfull_var", "zg")
        zghalf_var = self.extra_facets.get("zghalf_var", "zghalf")
        if cubes.extract(NameConstraint(var_name=zgfull_var)):
            if cubes.extract(NameConstraint(var_name=zghalf_var)):
                zghalf = zghalf_var
            else:
                zghalf = None
            alt_coord = self._get_z_coord(
                cubes,
                zgfull_var,
                bounds_name=zghalf,
            )
            self.fix_alt16_metadata(cube, alt_coord)

            # Altitude coordinate only spans height and spatial dimensions (no
            # time) -> these are always the last two dimensions in the cube
            cube.add_aux_coord(alt_coord, np.arange(cube.ndim)[-2:])
        else:
            logger.debug(
                "Cannot add altitude information to ICON data; variables '%s' "
                "and/or '%s' are not available",
                zgfull_var,
                zghalf_var,
            )

        # Fix metadata
        z_coord = cube.coord("height")
        if z_coord.units.is_convertible("m"):
            self.fix_height_metadata(cube, z_coord)
        else:
            z_coord.var_name = "model_level"
            z_coord.standard_name = None
            z_coord.long_name = "model level number"
            z_coord.units = "no unit"
            z_coord.attributes["positive"] = "up"
            z_coord.points = np.arange(len(z_coord.points))
            z_coord.bounds = None

        return cube

    def _fix_depth(self, cube: Cube, cubes: CubeList) -> None:
        """Fix ocean depth coordinate."""
        depth_coord = self.fix_depth_coord_metadata(cube)
        if depth_coord.has_bounds():
            return

        # Try to get bounds of depth coordinate from depth_2 coordinate that
        # might be present in other variables loaded from the same file
        for other_cube in cubes:
            if not other_cube.coords(var_name="depth_2"):
                continue
            depth_2_coord = other_cube.coord(var_name="depth_2")
            depth_2_coord.convert_units(depth_coord.units)
            bounds = depth_2_coord.core_points()
            depth_coord.bounds = np.stack((bounds[:-1], bounds[1:]), axis=-1)

    def _fix_lat(self, cube: Cube) -> tuple[int, ...]:
        """Fix latitude coordinate of cube."""
        lat_var = self.extra_facets.get("lat_var", "clat")

        # Add latitude coordinate if not already present
        if not cube.coords(var_name=lat_var):
            logger.debug(
                "ICON data does not contain latitude variable '%s', trying to "
                "add it via grid file",
                lat_var,
            )
            try:
                self._add_coord_from_grid_file(cube, lat_var)
            except Exception as exc:
                msg = (
                    f"Failed to add missing latitude coordinate '{lat_var}' "
                    f"to cube"
                )
                raise ValueError(msg) from exc

        # Fix metadata
        lat = cube.coord(var_name=lat_var)
        lat = self.fix_lat_metadata(cube, lat)

        return cube.coord_dims(lat)  # type: ignore

    def _fix_lon(self, cube: Cube) -> tuple[int, ...]:
        """Fix longitude coordinate of cube."""
        lon_var = self.extra_facets.get("lon_var", "clon")

        # Add longitude coordinate if not already present
        if not cube.coords(var_name=lon_var):
            logger.debug(
                "ICON data does not contain longitude variable '%s', trying "
                "to add it via grid file",
                lon_var,
            )
            try:
                self._add_coord_from_grid_file(cube, lon_var)
            except Exception as exc:
                msg = (
                    f"Failed to add missing longitude coordinate '{lon_var}' "
                    f"to cube"
                )
                raise ValueError(msg) from exc

        # Fix metadata and convert to [0, 360]
        lon = cube.coord(var_name=lon_var)
        lon = self.fix_lon_metadata(cube, lon)
        self._set_range_in_0_360(lon)

        return cube.coord_dims(lon)  # type: ignore

    def _fix_time(self, cube: Cube, cubes: CubeList) -> Cube:
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

        # ICON usually reports aggregated values at the end of the time period,
        # e.g., for monthly output, ICON reports the month February as 1 March.
        # Thus, if not disabled, shift all time points back by 1/2 of the given
        # time period.
        if self.extra_facets.get("shift_time", True):
            self._shift_time_coord(cube, time_coord)

        # If not already present, try to add bounds here. Usually bounds are
        # set in _shift_time_coord.
        self.guess_coord_bounds(cube, time_coord)

        return cube

    def _shift_time_coord(self, cube: Cube, time_coord: Coord) -> None:
        """Shift time points back by 1/2 of given time period (in-place)."""
        # Do not modify time coordinate for point measurements
        for cell_method in cube.cell_methods:
            is_point_measurement = (
                "time" in cell_method.coord_names
                and "point" in cell_method.method
            )
            if is_point_measurement:
                logger.debug(
                    "ICON data describes point measurements: time coordinate "
                    "will not be shifted back by 1/2 of output interval (%s)",
                    self.extra_facets["frequency"],
                )
                return

        # Remove bounds; they will be re-added later after shifting
        time_coord.bounds = None

        # For decadal, yearly and monthly data, round datetimes to closest day
        freq = self.extra_facets["frequency"]
        if "dec" in freq or "yr" in freq or "mon" in freq:
            time_units = time_coord.units
            time_coord.convert_units(
                Unit("days since 1850-01-01", calendar=time_units.calendar),
            )
            try:
                time_coord.points = np.around(time_coord.points)
            except ValueError as exc:
                error_msg = (
                    "Cannot shift time coordinate: Rounding to closest day "
                    "failed. Most likely you specified the wrong frequency in "
                    "the recipe (use `frequency: <your_frequency>` to fix "
                    "this). Alternatively, use `shift_time=false` in the "
                    "recipe to disable this feature"
                )
                raise ValueError(error_msg) from exc
            time_coord.convert_units(time_units)
            logger.debug(
                "Rounded ICON time coordinate to closest day for decadal, "
                "yearly and monthly data",
            )

        # Use original time points to calculate bounds (for a given point,
        # start of bounds is previous point, end of bounds is point)
        first_datetime = time_coord.units.num2date(time_coord.points[0])
        previous_time_point = time_coord.units.date2num(
            self._get_previous_timestep(first_datetime),
        )
        extended_time_points = np.concatenate(
            ([previous_time_point], time_coord.points),
        )
        time_coord.points = (
            np.convolve(extended_time_points, np.ones(2), "valid") / 2.0
        )  # running mean with window length 2
        time_coord.bounds = np.stack(
            (extended_time_points[:-1], extended_time_points[1:]),
            axis=-1,
        )
        logger.debug(
            "Shifted ICON time coordinate back by 1/2 of output interval (%s)",
            self.extra_facets["frequency"],
        )

    def _get_previous_timestep(self, datetime_point: datetime) -> datetime:
        """Get previous time step."""
        freq = self.extra_facets["frequency"]
        year = datetime_point.year
        month = datetime_point.month

        # Invalid input
        invalid_freq_error_msg = (
            f"Cannot shift time coordinate: failed to determine previous time "
            f"step for frequency '{freq}'. Use `shift_time=false` in the "
            f"recipe to disable this feature"
        )
        if "fx" in freq or "subhr" in freq:
            raise ValueError(invalid_freq_error_msg)

        # For decadal, yearly and monthly data, the points needs to be the
        # first of the month 00:00:00
        if "dec" in freq or "yr" in freq or "mon" in freq:
            if datetime_point != datetime(year, month, 1):
                msg = (
                    f"Cannot shift time coordinate: expected first of the "
                    f"month at 00:00:00 for decadal, yearly and monthly data, "
                    f"got {datetime_point}. Use `shift_time=false` in the "
                    f"recipe to disable this feature"
                )
                raise ValueError(
                    msg,
                )

        # Decadal data
        if "dec" in freq:
            return datetime_point.replace(year=year - 10)

        # Yearly data
        if "yr" in freq:
            return datetime_point.replace(year=year - 1)

        # Monthly data
        if "mon" in freq:
            new_month = (month - 2) % 12 + 1
            new_year = year + (month - 2) // 12
            return datetime_point.replace(year=new_year, month=new_month)

        # Daily data
        if "day" in freq:
            return datetime_point - timedelta(days=1)

        # Hourly data
        if "hr" in freq:
            (n_hours, _, _) = freq.partition("hr")
            if not n_hours:
                n_hours = 1
            return datetime_point - timedelta(hours=int(n_hours))

        # Unknown input
        raise ValueError(invalid_freq_error_msg)

    def _fix_mesh(self, cube: Cube, mesh_idx: tuple[int, ...]) -> None:
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
            for mesh_coord in mesh.to_MeshCoords("face"):
                cube.add_aux_coord(mesh_coord, mesh_idx)

    @staticmethod
    def _is_unstructured_grid(
        lat_idx: tuple[int, ...] | None,
        lon_idx: tuple[int, ...] | None,
    ) -> bool:
        """Check if data is defined on an unstructured grid."""
        # If either latitude or longitude are not present (i.e., the
        # corresponding index is None), no unstructured grid is present
        if lat_idx is None:
            return False
        if lon_idx is None:
            return False

        # If latitude and longitude do not share their dimensions, no
        # unstructured grid is present
        if lat_idx != lon_idx:
            return False

        # If latitude and longitude are multi-dimensional (e.g., curvilinear
        # grid), no unstructured grid is present
        return len(lat_idx) == 1

    @staticmethod
    def _fix_invalid_time_units(time_coord: Coord) -> None:
        """Fix invalid time units (in-place)."""
        # ICON data usually has no time bounds. To be 100% sure, we remove the
        # bounds here (they will be added at a later stage).
        time_coord.bounds = None
        time_format = "day as %Y%m%d.%f"
        t_unit = time_coord.attributes.pop("invalid_units")
        if t_unit != time_format:
            msg = (
                f"Expected time units '{time_format}' in input file, got "
                f"'{t_unit}'"
            )
            raise ValueError(
                msg,
            )
        new_t_units = Unit(
            "days since 1850-01-01",
            calendar="proleptic_gregorian",
        )

        # New routine to convert time of daily and hourly data. The string %f
        # (fraction of day) is not a valid format string for datetime.strptime,
        # so we have to convert it ourselves.
        time_str = pd.Series(time_coord.points, dtype=str)

        # First, extract date (year, month, day) from string and convert it to
        # datetime object
        year_month_day_str = time_str.str.extract(r"(\d*)\.?\d*", expand=False)
        year_month_day = pd.to_datetime(year_month_day_str, format="%Y%m%d")

        # Second, extract day fraction and convert it to timedelta object
        day_float_str = time_str.str.extract(
            r"\d*(\.\d*)",
            expand=False,
        ).fillna("0.0")
        day_float = pd.to_timedelta(day_float_str.astype(float), unit="D")

        # Finally, add date and day fraction to get final datetime and convert
        # it to correct units. Note: we also round to next second, otherwise
        # this results in times that are off by 1s (e.g., 13:59:59 instead of
        # 14:00:00). We round elements individually since rounding the
        # pd.Series object directly is broken
        # (https://github.com/pandas-dev/pandas/issues/57002).
        datetimes = year_month_day + day_float
        rounded_datetimes = pd.Series(dt.round("s") for dt in datetimes)
        with warnings.catch_warnings():
            # We already fixed the deprecated code as recommended in the
            # warning, but it still shows up -> ignore it
            warnings.filterwarnings(
                "ignore",
                message="The behavior of DatetimeProperties.to_pydatetime .*",
                category=FutureWarning,
            )
            new_datetimes = np.array(rounded_datetimes.dt.to_pydatetime())
        new_dt_points = date2num(np.array(new_datetimes), new_t_units)  # type: ignore

        # Modify time coordinate in place
        time_coord.points = new_dt_points
        time_coord.units = new_t_units


class NegateData(IconFix):
    """Base fix to negate data."""

    def fix_data(self, cube: Cube) -> Cube:
        """Fix data."""
        cube.data = -cube.core_data()
        return cube
