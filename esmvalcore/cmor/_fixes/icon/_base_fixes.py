"""Fix base classes for ICON on-the-fly CMORizer."""
from __future__ import annotations

import logging
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse

import iris
import numpy as np
import requests
from iris import NameConstraint
from iris.cube import Cube, CubeList
from iris.experimental.ugrid import Connectivity, Mesh

from esmvalcore.cmor._fixes.native_datasets import NativeDatasetFix
from esmvalcore.local import _get_rootpath, _replace_tags, _select_drs

logger = logging.getLogger(__name__)


class IconFix(NativeDatasetFix):
    """Base class for all ICON fixes."""

    CACHE_DIR = Path.home() / '.esmvaltool' / 'cache'
    CACHE_VALIDITY = 7 * 24 * 60 * 60  # [s]; = 1 week
    TIMEOUT = 5 * 60  # [s]; = 5 min
    GRID_FILE_ATTR = 'grid_file_uri'

    def __init__(self, *args, **kwargs):
        """Initialize ICON fix."""
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

        # Extract connectivity (i.e., the mapping cell faces -> cell nodes)
        # from the the horizontal grid file (in ICON jargon called
        # 'vertex_of_cell'; since UGRID expects a different dimension ordering
        # we transpose the cube here)
        vertex_of_cell = horizontal_grid.extract_cube(
            NameConstraint(var_name='vertex_of_cell'))
        vertex_of_cell.transpose()

        # Extract start index used to name nodes from the the horizontal grid
        # file
        start_index = self._get_start_index(horizontal_grid)

        # Extract face coordinates from cube (in ICON jargon called 'cell
        # latitude' and 'cell longitude')
        face_lat = cube.coord('latitude')
        face_lon = cube.coord('longitude')

        # Extract node coordinates from horizontal grid
        (node_lat, node_lon) = self._get_node_coords(horizontal_grid)

        # The bounds given by the face coordinates slightly differ from the
        # bounds determined by the connectivity. We arbitrarily assume here
        # that the information given by the connectivity is correct.
        conn_node_inds = vertex_of_cell.data - start_index

        # Latitude: there might be slight numerical differences (-> check that
        # the differences are very small before fixing it)
        close_kwargs = {'rtol': 1e-3, 'atol': 1e-5}
        if not np.allclose(
                face_lat.bounds,
                node_lat.points[conn_node_inds],
                **close_kwargs,
        ):
            logger.warning(
                "Latitude bounds of the face coordinate ('clat_vertices' in "
                "the grid file) differ from the corresponding values "
                "calculated from the connectivity ('vertex_of_cell') and the "
                "node coordinate ('vlat'). Using bounds defined by "
                "connectivity."
            )
        face_lat.bounds = node_lat.points[conn_node_inds]

        # Longitude: there might be differences at the poles, where the
        # longitude information does not matter (-> check that the only large
        # differences are located at the poles). In addition, values might
        # differ by 360°, which is also okay.
        face_lon_bounds_to_check = face_lon.bounds % 360
        node_lon_conn_to_check = node_lon.points[conn_node_inds] % 360
        idx_notclose = ~np.isclose(
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
                "defined by connectivity."
            )
        face_lon.bounds = node_lon.points[conn_node_inds]

        # Create mesh
        connectivity = Connectivity(
            indices=vertex_of_cell.data,
            cf_role='face_node_connectivity',
            start_index=start_index,
            location_axis=0,
        )
        mesh = Mesh(
            topology_dimension=2,
            node_coords_and_axes=[(node_lat, 'y'), (node_lon, 'x')],
            connectivities=[connectivity],
            face_coords_and_axes=[(face_lat, 'y'), (face_lon, 'x')],
        )

        return mesh

    def _get_grid_url(self, cube):
        """Get ICON grid URL from cube."""
        if self.GRID_FILE_ATTR not in cube.attributes:
            raise ValueError(
                f"Cube does not contain the attribute '{self.GRID_FILE_ATTR}' "
                f"necessary to download the ICON horizontal grid file:\n"
                f"{cube}")
        grid_url = cube.attributes[self.GRID_FILE_ATTR]
        parsed_url = urlparse(grid_url)
        grid_name = Path(parsed_url.path).name
        return (grid_url, grid_name)

    def _get_node_coords(self, horizontal_grid):
        """Get node coordinates from horizontal grid.

        Extract node coordinates from dummy variable 'dual_area' in horizontal
        grid file (in ICON jargon called 'vertex latitude' and 'vertex
        longitude'), remove their bounds (not accepted by UGRID), and adapt
        metadata.

        """
        dual_area_cube = horizontal_grid.extract_cube(
            NameConstraint(var_name='dual_area'))
        node_lat = dual_area_cube.coord(var_name='vlat')
        node_lon = dual_area_cube.coord(var_name='vlon')

        # Fix metadata
        node_lat.bounds = None
        node_lon.bounds = None
        node_lat.var_name = 'nlat'
        node_lon.var_name = 'nlon'
        node_lat.standard_name = 'latitude'
        node_lon.standard_name = 'longitude'
        node_lat.long_name = 'node latitude'
        node_lon.long_name = 'node longitude'
        node_lat.convert_units('degrees_north')
        node_lon.convert_units('degrees_east')

        # Convert longitude to [0, 360]
        self._set_range_in_0_360(node_lon)

        return (node_lat, node_lon)

    def _get_path_from_facet(self, facet, description=None):
        """Try to get path from facet."""
        if description is None:
            description = 'File'
        path = Path(os.path.expandvars(self.extra_facets[facet])).expanduser()
        if not path.is_file():
            new_path = self.session['auxiliary_data_dir'] / path
            if not new_path.is_file():
                raise FileNotFoundError(
                    f"{description} '{path}' given by facet '{facet}' does "
                    f"not exist (specify a valid absolute path or a path "
                    f"relative to the auxiliary_data_dir "
                    f"'{self.session['auxiliary_data_dir']}')"
                )
            path = new_path
        return path

    def add_additional_cubes(self, cubes):
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
        Files can be specified as absolute or relative (to
        ``auxiliary_data_dir`` as defined in the :ref:`user configuration
        file`) paths.

        Parameters
        ----------
        cubes: iris.cube.CubeList
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
            'zg_file',
            'zghalf_file',
        ]
        for facet in facets_to_consider:
            if self.extra_facets.get(facet) is None:
                continue
            path_to_add = self._get_path_from_facet(facet)
            logger.debug("Adding cubes from %s", path_to_add)
            new_cubes = self._load_cubes(path_to_add)
            cubes.extend(new_cubes)

        return cubes

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
        rootpaths = _get_rootpath('ICON')
        dirname_template = _select_drs('input_dir', 'ICON')
        dirname_globs = _replace_tags(dirname_template, self.extra_facets)
        possible_grid_paths = [
            r / d / grid_name for r in rootpaths for d in dirname_globs
        ]
        for grid_path in possible_grid_paths:
            if grid_path.is_file():
                logger.debug("Using ICON grid file '%s'", grid_path)
                cubes = self._load_cubes(grid_path)
                return cubes
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
                logger.debug("Existing cached ICON grid file '%s' is outdated",
                             grid_path)

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
                with tmp_path.open('wb') as file:
                    copyfileobj(response.raw, file)
            shutil.move(tmp_path, grid_path)
            logger.info(
                "Successfully downloaded ICON grid file from '%s' to '%s' "
                "and moved it to '%s'",
                grid_url,
                tmp_path,
                grid_path,
            )

        cubes = self._load_cubes(grid_path)
        return cubes

    def get_horizontal_grid(self, cube):
        """Get copy of ICON horizontal grid.

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
        cube: iris.cube.Cube
            Cube for which the ICON horizontal grid is retrieved. If the facet
            `horizontal_grid` is not specified by the user, this cube needs to
            have a global attribute `grid_file_uri` that specifies the download
            location of the ICON horizontal grid file.

        Returns
        -------
        iris.cube.CubeList
            Copy of ICON horizontal grid.

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
        if self.extra_facets.get('horizontal_grid') is not None:
            grid = self._get_grid_from_facet()
        else:
            grid = self._get_grid_from_cube_attr(cube)

        return grid.copy()

    def get_mesh(self, cube):
        """Get mesh.

        Note
        ----
        If possible, this function uses a cached version of the mesh to save
        time.

        Parameters
        ----------
        cube: iris.cube.Cube
            Cube for which the mesh is retrieved. If the facet
            `horizontal_grid` is not specified by the user, this cube needs to
            have the global attribute `grid_file_uri` that specifies the
            download location of the ICON horizontal grid file.

        Returns
        -------
        iris.experimental.ugrid.Mesh
            Mesh.

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
        if self.extra_facets.get('horizontal_grid') is not None:
            grid_path = self._get_path_from_facet(
                'horizontal_grid', 'Horizontal grid file'
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
    def _get_start_index(horizontal_grid):
        """Get start index used to name nodes from horizontal grid.

        Extract start index used to name nodes from the the horizontal grid
        file (in ICON jargon called 'vertex_index').

        Note
        ----
        UGRID expects this to be a int32.

        """
        vertex_index = horizontal_grid.extract_cube(
            NameConstraint(var_name='vertex_index'))
        return np.int32(np.min(vertex_index.data))

    @staticmethod
    def _load_cubes(path: Path | str) -> CubeList:
        """Load cubes and ignore certain warnings."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message="Ignoring netCDF variable .* invalid units .*",
                category=UserWarning,
                module='iris',
            )  # iris < 3.8
            warnings.filterwarnings(
                'ignore',
                message="Ignoring invalid units .* on netCDF variable .*",
                category=UserWarning,
                module='iris',
            )  # iris >= 3.8
            warnings.filterwarnings(
                'ignore',
                message="Failed to create 'height' dimension coordinate: The "
                        "'height' DimCoord bounds array must be strictly "
                        "monotonic.",
                category=UserWarning,
                module='iris',
            )
            cubes = iris.load(path)
        return cubes

    @staticmethod
    def _set_range_in_0_360(lon_coord):
        """Convert longitude coordinate to [0, 360]."""
        lon_coord.points = (lon_coord.core_points() + 360.0) % 360.0
        if lon_coord.has_bounds():
            lon_coord.bounds = (lon_coord.core_bounds() + 360.0) % 360.0


class NegateData(IconFix):
    """Base fix to negate data."""

    def fix_data(self, cube):
        """Fix data."""
        cube.data = -cube.core_data()
        return cube
