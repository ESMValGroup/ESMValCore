"""On-the-fly CMORizer for ICON."""

import logging
from datetime import datetime

import cf_units
import dask.array as da
import iris
import iris.util
import numpy as np
from iris import NameConstraint
from iris.coords import AuxCoord, DimCoord
from iris.cube import CubeList
from iris.experimental.ugrid import Connectivity, Mesh

from esmvalcore.iris_helpers import add_leading_dim_to_cube, date2num

from ._base_fixes import IconFix, SetUnitsTo1

logger = logging.getLogger(__name__)


class AllVars(IconFix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes)

        # Fix time
        if 'time' in self.vardef.dimensions:
            cube = self._fix_time(cube, cubes)

        # Fix height (note: cannot use "if 'height' in self.vardef.dimensions"
        # here since the name of the z-coord varies from variable to variable)
        if cube.coords('height'):
            # In case a scalar height is required, remove it here (it is added
            # at a later stage). The step _fix_height() is designed to fix
            # non-scalar height coordinates.
            if (cube.coord('height').shape[0] == 1 and (
                    'height2m' in self.vardef.dimensions or
                    'height10m' in self.vardef.dimensions)):
                # If height is a dimensional coordinate with length 1, squeeze
                # the cube.
                # Note: iris.util.squeeze is not used here since it might
                # accidentally squeeze other dimensions.
                if cube.coords('height', dim_coords=True):
                    slices = [slice(None)] * cube.ndim
                    slices[cube.coord_dims('height')[0]] = 0
                    cube = cube[tuple(slices)]
                cube.remove_coord('height')
            else:
                cube = self._fix_height(cube, cubes)

        # Fix latitude
        if 'latitude' in self.vardef.dimensions:
            lat_idx = self._fix_lat(cube)
        else:
            lat_idx = None

        # Fix longitude
        if 'longitude' in self.vardef.dimensions:
            lon_idx = self._fix_lon(cube)
        else:
            lon_idx = None

        # Fix unstructured mesh of unstructured grid if present
        if self._is_unstructured_grid(lat_idx, lon_idx):
            self._fix_mesh(cube, lat_idx)

        # Fix scalar coordinates
        self.fix_scalar_coords(cube)

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
        # The following dict maps from desired coordinate name in output file
        # (dict keys) to coordinate name in grid file (dict values)
        coord_names_mapping = {
            'latitude': 'grid_latitude',
            'longitude': 'grid_longitude',
        }
        if coord_name not in coord_names_mapping:
            raise ValueError(
                f"coord_name must be one of {list(coord_names_mapping)}, got "
                f"'{coord_name}'")
        coord_name_in_grid = coord_names_mapping[coord_name]

        # Use 'cell_area' as dummy cube to extract desired coordinates
        # Note: it might be necessary to expand this when more coord_names are
        # supported
        horizontal_grid = self.get_horizontal_grid(cube)
        grid_cube = horizontal_grid.extract_cube(
            NameConstraint(var_name='cell_area'))
        coord = grid_cube.coord(coord_name_in_grid)

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

    def _add_time(self, cube, cubes):
        """Add time coordinate from other cube in cubes."""
        # Try to find time cube from other cubes and it to target cube
        for other_cube in cubes:
            if not other_cube.coords('time'):
                continue
            time_coord = other_cube.coord('time')
            cube = add_leading_dim_to_cube(cube, time_coord)
            return cube
        raise ValueError(
            f"Cannot add required coordinate 'time' to variable "
            f"'{self.vardef.short_name}', cube and other cubes in file do not "
            f"contain it")

    def _fix_height(self, cube, cubes):
        """Fix height coordinate of cube."""
        # Reverse entire cube along height axis so that index 0 is surface
        # level
        cube = iris.util.reverse(cube, 'height')

        # Add air_pressure coordinate if possible
        # (make sure to also reverse pressure cubes)
        if cubes.extract(NameConstraint(var_name='pfull')):
            plev_points_cube = iris.util.reverse(
                cubes.extract_cube(NameConstraint(var_name='pfull')),
                'height',
            )
            air_pressure_points = plev_points_cube.core_data()

            # Get bounds from half levels and reshape array
            if cubes.extract(NameConstraint(var_name='phalf')):
                plev_bounds_cube = iris.util.reverse(
                    cubes.extract_cube(NameConstraint(var_name='phalf')),
                    'height',
                )
                air_pressure_bounds = plev_bounds_cube.core_data()
                air_pressure_bounds = da.stack(
                    (air_pressure_bounds[:, :-1], air_pressure_bounds[:, 1:]),
                    axis=-1)
            else:
                air_pressure_bounds = None

            # Setup air pressure coordinate with correct metadata and add to
            # cube
            air_pressure_coord = AuxCoord(
                air_pressure_points,
                bounds=air_pressure_bounds,
                var_name='plev',
                standard_name='air_pressure',
                long_name='pressure',
                units=plev_points_cube.units,
                attributes={'positive': 'down'},
            )
            cube.add_aux_coord(air_pressure_coord, np.arange(cube.ndim))

        # Fix metadata
        z_coord = cube.coord('height')
        if z_coord.units.is_convertible('m'):
            self.fix_height_metadata(cube, z_coord)
        else:
            z_coord.var_name = 'model_level'
            z_coord.standard_name = None
            z_coord.long_name = 'model level number'
            z_coord.units = 'no unit'
            z_coord.attributes['positive'] = 'up'
            z_coord.points = np.arange(len(z_coord.points))
            z_coord.bounds = None

        return cube

    def _fix_lat(self, cube):
        """Fix latitude coordinate of cube."""
        lat_name = self.extra_facets.get('latitude', 'latitude')

        # Add latitude coordinate if not already present
        if not cube.coords(lat_name):
            try:
                self._add_coord_from_grid_file(cube, 'latitude')
            except Exception as exc:
                msg = "Failed to add missing latitude coordinate to cube"
                raise ValueError(msg) from exc

        # Fix metadata
        lat = self.fix_lat_metadata(cube, lat_name)

        return cube.coord_dims(lat)

    def _fix_lon(self, cube):
        """Fix longitude coordinate of cube."""
        lon_name = self.extra_facets.get('longitude', 'longitude')

        # Add longitude coordinate if not already present
        if not cube.coords(lon_name):
            try:
                self._add_coord_from_grid_file(cube, 'longitude')
            except Exception as exc:
                msg = "Failed to add missing longitude coordinate to cube"
                raise ValueError(msg) from exc

        # Fix metadata
        lon = self.fix_lon_metadata(cube, lon_name)

        return cube.coord_dims(lon)

    def _fix_time(self, cube, cubes):
        """Fix time coordinate of cube."""
        # Add time coordinate if not already present
        if not cube.coords('time'):
            cube = self._add_time(cube, cubes)

        # Fix metadata and add bounds
        time_coord = self.fix_time_metadata(cube)
        self.guess_coord_bounds(cube, time_coord)
        if 'invalid_units' not in time_coord.attributes:
            return cube

        # If necessary, convert invalid time units of the form "day as
        # %Y%m%d.%f" to CF format (e.g., "days since 1850-01-01")
        # Notes:
        # - It might be necessary to expand this to other time formats in the
        #   raw file.
        # - This has not been tested with sub-daily data
        time_format = 'day as %Y%m%d.%f'
        t_unit = time_coord.attributes.pop('invalid_units')
        if t_unit != time_format:
            raise ValueError(
                f"Expected time units '{time_format}' in input file, got "
                f"'{t_unit}'")
        new_t_unit = cf_units.Unit('days since 1850-01-01',
                                   calendar='proleptic_gregorian')

        new_datetimes = [datetime.strptime(str(dt), '%Y%m%d.%f') for dt in
                         time_coord.points]
        new_dt_points = date2num(np.array(new_datetimes), new_t_unit)

        time_coord.points = new_dt_points
        time_coord.units = new_t_unit

        return cube

    def _get_mesh(self, cube):
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
        if not np.allclose(face_lat.bounds, node_lat.points[conn_node_inds]):
            raise ValueError(
                "Cannot create mesh from horizontal grid file: latitude "
                "bounds of the face coordinate ('clat_vertices' in the grid "
                "file) differ from the corresponding values calculated from "
                "the connectivity ('vertex_of_cell') and the node coordinate "
                "('vlat')")
        face_lat.bounds = node_lat.points[conn_node_inds]

        # Longitude: there might be differences at the poles, where the
        # longitude information does not matter (-> check that the only large
        # differences are located at the poles). In addition, values might
        # differ by 360°, which is also okay.
        face_lon_bounds_to_check = face_lon.bounds % 360
        node_lon_conn_to_check = node_lon.points[conn_node_inds] % 360
        idx_notclose = ~np.isclose(face_lon_bounds_to_check,
                                   node_lon_conn_to_check)
        if not np.allclose(np.abs(face_lat.bounds[idx_notclose]), 90.0):
            raise ValueError(
                "Cannot create mesh from horizontal grid file: longitude "
                "bounds of the face coordinate ('clon_vertices' in the grid "
                "file) differ from the corresponding values calculated from "
                "the connectivity ('vertex_of_cell') and the node coordinate "
                "('vlon'). Note that these values are allowed to differ by "
                "360° or at the poles of the grid.")
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

        # Create mesh and replace the original latitude and longitude
        # coordinates with their new mesh versions
        mesh = self._get_mesh(cube)
        cube.remove_coord('latitude')
        cube.remove_coord('longitude')
        for mesh_coord in mesh.to_MeshCoords('face'):
            cube.add_aux_coord(mesh_coord, mesh_idx)

    def _fix_var_metadata(self, cube):
        """Fix metadata of variable."""
        if self.vardef.standard_name == '':
            cube.standard_name = None
        else:
            cube.standard_name = self.vardef.standard_name
        cube.var_name = self.vardef.short_name
        cube.long_name = self.vardef.long_name
        if cube.units != self.vardef.units:
            cube.convert_units(self.vardef.units)
        if self.vardef.positive != '':
            cube.attributes['positive'] = self.vardef.positive

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
    def _get_node_coords(horizontal_grid):
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

        return (node_lat, node_lon)

    @staticmethod
    def _is_unstructured_grid(lat_idx, lon_idx):
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
        if len(lat_idx) != 1:
            return False

        return True


Hur = SetUnitsTo1


Siconc = SetUnitsTo1


Siconca = SetUnitsTo1
