# -*- coding: utf-8 -*-
"""Provides regridding for irregular grids."""

try:
    import esmpy
except ImportError as exc:
    # Prior to v8.4.0, `esmpy`` could be imported as `ESMF`.
    try:
        import ESMF as esmpy  # noqa: N811
    except ImportError:
        raise exc
import iris
import numpy as np
from iris.cube import Cube

from ._mapping import get_empty_data, map_slices, ref_to_dims_index

ESMF_MANAGER = esmpy.Manager(debug=False)

ESMF_LON, ESMF_LAT = 0, 1

ESMF_REGRID_METHODS = {
    'linear': esmpy.RegridMethod.BILINEAR,
    'area_weighted': esmpy.RegridMethod.CONSERVE,
    'nearest': esmpy.RegridMethod.NEAREST_STOD,
}

MASK_REGRIDDING_MASK_VALUE = {
    esmpy.RegridMethod.BILINEAR: np.array([1]),
    esmpy.RegridMethod.CONSERVE: np.array([1]),
    esmpy.RegridMethod.NEAREST_STOD: np.array([]),
}

# ESMF_REGRID_METHODS = {
#     'bilinear': esmpy.RegridMethod.BILINEAR,
#     'patch': esmpy.RegridMethod.PATCH,
#     'conserve': esmpy.RegridMethod.CONSERVE,
#     'nearest_stod': esmpy.RegridMethod.NEAREST_STOD,
#     'nearest_dtos': esmpy.RegridMethod.NEAREST_DTOS,
# }


class ESMPyRegridder:
    """General ESMPy regridder.

    Does not support lazy regridding nor weights caching.

    Parameters
    ----------
    src_cube:
        Cube defining the source grid.
    tgt_cube:
        Cube defining the target grid.
    method:
        Regridding algorithm. Must be one of `linear`, `area_weighted`,
        `nearest`.
    mask_threshold:
        Threshold used to regrid mask of input cube.

    """

    def __init__(
        self,
        src_cube: Cube,
        tgt_cube: Cube,
        method: str = 'linear',
        mask_threshold: float = 0.99,
    ):
        """Initialize class instance."""
        # These regridders are not lazy, so load source and target data once.
        src_cube.data  # pylint: disable=pointless-statement
        tgt_cube.data  # pylint: disable=pointless-statement
        self.src_cube = src_cube
        self.tgt_cube = tgt_cube
        self.method = method
        self.mask_threshold = mask_threshold

    def __call__(self, cube: Cube) -> Cube:
        """Perform regridding.

        Parameters
        ----------
        cube:
            Cube to be regridded.

        Returns
        -------
        Cube
            Regridded cube.

        """
        # These regridders are not lazy, so load source data once.
        cube.data  # pylint: disable=pointless-statement
        src_rep, dst_rep = get_grid_representants(cube, self.tgt_cube)
        regridder = build_regridder(
            src_rep, dst_rep, self.method, mask_threshold=self.mask_threshold
        )
        result = map_slices(cube, regridder, src_rep, dst_rep)
        return result


class _ESMPyScheme:
    """General irregular regridding scheme.

    This class can be used in :meth:`iris.cube.Cube.regrid`.

    Note
    ----
    See `ESMPy <http://www.earthsystemmodeling.org/
    esmf_releases/non_public/ESMF_7_0_0/esmpy_doc/html/
    RegridMethod.html#ESMF.api.constants.RegridMethod>`__ for more details on
    this.

    Parameters
    ----------
    mask_threshold:
        Threshold used to regrid mask of source cube.

    """

    _METHOD = ''

    def __init__(self, mask_threshold: float = 0.99):
        """Initialize class instance."""
        self.mask_threshold = mask_threshold

    def __repr__(self) -> str:
        """Return string representation of class."""
        return (
            f'{self.__class__.__name__}(mask_threshold={self.mask_threshold})'
        )

    def regridder(self, src_cube: Cube, tgt_cube: Cube) -> ESMPyRegridder:
        """Get regridder.

        Parameters
        ----------
        src_cube:
            Cube defining the source grid.
        tgt_cube:
            Cube defining the target grid.

        Returns
        -------
        ESMPyRegridder
            Regridder instance.

        """
        return ESMPyRegridder(
            src_cube,
            tgt_cube,
            method=self._METHOD,
            mask_threshold=self.mask_threshold,
        )


class ESMPyAreaWeighted(_ESMPyScheme):
    """ESMPy area-weighted regridding scheme.

    This class can be used in :meth:`iris.cube.Cube.regrid`.

    Does not support lazy regridding.

    """

    _METHOD = 'area_weighted'


class ESMPyLinear(_ESMPyScheme):
    """ESMPy bilinear regridding scheme.

    This class can be used in :meth:`iris.cube.Cube.regrid`.

    Does not support lazy regridding.

    """

    _METHOD = 'linear'


class ESMPyNearest(_ESMPyScheme):
    """ESMPy nearest-neighbor regridding scheme.

    This class can be used in :meth:`iris.cube.Cube.regrid`.

    Does not support lazy regridding.

    """

    _METHOD = 'nearest'


def cf_2d_bounds_to_esmpy_corners(bounds, circular):
    """Convert cf style 2d bounds to normal (esmpy style) corners."""
    no_lat_points, no_lon_points = bounds.shape[:2]
    no_lat_bounds = no_lat_points + 1
    if circular:
        no_lon_bounds = no_lon_points
    else:
        no_lon_bounds = no_lon_points + 1
    esmpy_corners = np.empty((no_lon_bounds, no_lat_bounds))
    esmpy_corners[:no_lon_points, :no_lat_points] = bounds[:, :, 0].T
    esmpy_corners[:no_lon_points, no_lat_points:] = bounds[-1:, :, 3].T
    esmpy_corners[no_lon_points:, :no_lat_points] = bounds[:, -1:, 1].T
    esmpy_corners[no_lon_points:, no_lat_points:] = bounds[-1:, -1:, 2].T
    return esmpy_corners


def coords_iris_to_esmpy(lat, lon, circular):
    """Build ESMF compatible coordinate information from iris coords."""
    dim = lat.ndim
    if lon.ndim != dim:
        msg = 'Different dimensions in latitude({}) and longitude({}) coords.'
        raise ValueError(msg.format(lat.ndim, lon.ndim))
    if dim == 1:
        for coord in [lat, lon]:
            if not coord.has_bounds():
                coord.guess_bounds()
        esmpy_lat, esmpy_lon = np.meshgrid(lat.points, lon.points)
        lat_corners = np.concatenate([lat.bounds[:, 0], lat.bounds[-1:, 1]])
        if circular:
            lon_corners = lon.bounds[:, 0]
        else:
            lon_corners = np.concatenate([lon.bounds[:, 0],
                                          lon.bounds[-1:, 1]])
        esmpy_lat_corners, esmpy_lon_corners = np.meshgrid(lat_corners,
                                                           lon_corners)
    elif dim == 2:
        esmpy_lat, esmpy_lon = lat.points.T.copy(), lon.points.T.copy()
        esmpy_lat_corners = cf_2d_bounds_to_esmpy_corners(lat.bounds, circular)
        esmpy_lon_corners = cf_2d_bounds_to_esmpy_corners(lon.bounds, circular)
    else:
        raise NotImplementedError(
            f'Coord dimension is {dim}. Expected 1 or 2.'
        )
    return esmpy_lat, esmpy_lon, esmpy_lat_corners, esmpy_lon_corners


def get_grid(esmpy_lat, esmpy_lon,
             esmpy_lat_corners, esmpy_lon_corners, circular):
    """Build EMSF grid from given coordinate information."""
    if circular:
        num_peri_dims = 1
    else:
        num_peri_dims = 0
    grid = esmpy.Grid(np.array(esmpy_lat.shape),
                      num_peri_dims=num_peri_dims,
                      staggerloc=[esmpy.StaggerLoc.CENTER])
    grid.get_coords(ESMF_LON)[...] = esmpy_lon
    grid.get_coords(ESMF_LAT)[...] = esmpy_lat
    grid.add_coords([esmpy.StaggerLoc.CORNER])
    grid_lon_corners = grid.get_coords(ESMF_LON,
                                       staggerloc=esmpy.StaggerLoc.CORNER)
    grid_lat_corners = grid.get_coords(ESMF_LAT,
                                       staggerloc=esmpy.StaggerLoc.CORNER)
    grid_lon_corners[...] = esmpy_lon_corners
    grid_lat_corners[...] = esmpy_lat_corners
    grid.add_item(esmpy.GridItem.MASK, esmpy.StaggerLoc.CENTER)
    return grid


def is_lon_circular(lon):
    """Determine if longitudes are circular."""
    if isinstance(lon, iris.coords.DimCoord):
        circular = lon.circular
    elif isinstance(lon, iris.coords.AuxCoord):
        if lon.ndim == 1:
            seam = lon.bounds[-1, 1] - lon.bounds[0, 0]
        elif lon.ndim == 2:
            seam = (lon.bounds[1:-1, -1, (1, 2)]
                    - lon.bounds[1:-1, 0, (0, 3)])
        else:
            raise NotImplementedError('AuxCoord longitude is higher '
                                      'dimensional than 2d. Giving up.')
        circular = np.all(abs(seam) % 360. < 1.e-3)
    else:
        raise ValueError('longitude is neither DimCoord nor AuxCoord. '
                         'Giving up.')
    return circular


def cube_to_empty_field(cube):
    """Build an empty ESMF field from a cube."""
    lat = cube.coord('latitude')
    lon = cube.coord('longitude')
    circular = is_lon_circular(lon)
    esmpy_coords = coords_iris_to_esmpy(lat, lon, circular)
    grid = get_grid(*esmpy_coords, circular=circular)
    field = esmpy.Field(grid,
                        name=cube.long_name,
                        staggerloc=esmpy.StaggerLoc.CENTER)
    return field


def get_representant(cube, ref_to_slice):
    """Get a representative slice from a cube."""
    slice_dims = ref_to_dims_index(cube, ref_to_slice)
    rep_ind = [0] * cube.ndim
    for dim in slice_dims:
        rep_ind[dim] = slice(None, None)
    rep_ind = tuple(rep_ind)
    return cube[rep_ind]


def regrid_mask_2d(src_data, regridding_arguments, mask_threshold):
    """Regrid the mask from the source field to the destination grid."""
    src_field = regridding_arguments['srcfield']
    dst_field = regridding_arguments['dstfield']
    regrid_method = regridding_arguments['regrid_method']
    original_src_mask = np.ma.getmaskarray(src_data)
    src_field.data[...] = ~original_src_mask.T
    src_mask = src_field.grid.get_item(esmpy.GridItem.MASK,
                                       esmpy.StaggerLoc.CENTER)
    src_mask[...] = original_src_mask.T
    center_mask = dst_field.grid.get_item(esmpy.GridItem.MASK,
                                          esmpy.StaggerLoc.CENTER)
    center_mask[...] = 0
    mask_regridder = esmpy.Regrid(
        src_mask_values=MASK_REGRIDDING_MASK_VALUE[regrid_method],
        dst_mask_values=np.array([]),
        **regridding_arguments)
    regr_field = mask_regridder(src_field, dst_field)
    dst_mask = regr_field.data[...].T < mask_threshold
    center_mask[...] = dst_mask.T
    if not dst_mask.any():
        dst_mask = np.ma.nomask
    return dst_mask


def build_regridder_2d(src_rep, dst_rep, regrid_method, mask_threshold):
    """Build regridder for 2d regridding."""
    dst_field = cube_to_empty_field(dst_rep)
    src_field = cube_to_empty_field(src_rep)
    regridding_arguments = {
        'srcfield': src_field,
        'dstfield': dst_field,
        'regrid_method': regrid_method,
        'unmapped_action': esmpy.UnmappedAction.IGNORE,
        'ignore_degenerate': True,
    }
    dst_mask = regrid_mask_2d(src_rep.data,
                              regridding_arguments, mask_threshold)
    field_regridder = esmpy.Regrid(src_mask_values=np.array([1]),
                                   dst_mask_values=np.array([1]),
                                   **regridding_arguments)

    def regridder(src):
        """Regrid 2d for irregular grids."""
        res = get_empty_data(dst_rep.shape, src.dtype)
        data = src.data
        if np.ma.is_masked(data):
            data = data.data
        src_field.data[...] = data.T
        regr_field = field_regridder(src_field, dst_field)
        res.data[...] = regr_field.data[...].T
        res.mask[...] = dst_mask
        return res

    return regridder


def build_regridder_3d(src_rep, dst_rep, regrid_method, mask_threshold):
    # The necessary refactoring will be done for the full 3d regridding.
    """Build regridder for 2.5d regridding."""
    esmf_regridders = []
    no_levels = src_rep.shape[0]
    for level in range(no_levels):
        esmf_regridders.append(
            build_regridder_2d(src_rep[level], dst_rep[level],
                               regrid_method, mask_threshold)
        )

    def regridder(src):
        """Regrid 2.5d for irregular grids."""
        res = get_empty_data(dst_rep.shape, src.dtype)
        for i, esmf_regridder in enumerate(esmf_regridders):
            res[i, ...] = esmf_regridder(src[i])
        return res

    return regridder


def build_regridder(src_rep, dst_rep, method, mask_threshold=.99):
    """Build regridders from representants."""
    regrid_method = ESMF_REGRID_METHODS[method]
    if src_rep.ndim == 2:
        regridder = build_regridder_2d(src_rep, dst_rep,
                                       regrid_method, mask_threshold)
    elif src_rep.ndim == 3:
        regridder = build_regridder_3d(src_rep, dst_rep,
                                       regrid_method, mask_threshold)
    return regridder


def get_grid_representant(cube, horizontal_only=False):
    """Extract the spatial grid from a cube."""
    horizontal_slice = ['latitude', 'longitude']
    ref_to_slice = horizontal_slice
    if not horizontal_only:
        try:
            cube_z_coord = cube.coord(axis='Z')
            n_zdims = len(cube.coord_dims(cube_z_coord))
            if n_zdims == 0:
                # scalar z coordinate, go on with 2d regridding
                pass
            elif n_zdims == 1:
                ref_to_slice = [cube_z_coord] + horizontal_slice
            else:
                raise ValueError("Cube has multidimensional Z coordinate.")
        except iris.exceptions.CoordinateNotFoundError:
            # no z coordinate, go on with 2d regridding
            pass
    return get_representant(cube, ref_to_slice)


def get_grid_representants(src, dst):
    """
    Construct cubes representing the source and destination grid.

    This method constructs two new cubes that representant the grids,
    i.e. the spatial dimensions of the given cubes.

    Parameters
    ----------
    src: :class:`iris.cube.Cube`
        Cube to be regridded. Typically a time series of 2d or 3d slices.
    dst: :class:`iris.cube.Cube`
        Cube defining the destination grid. Usually just a 2d or 3d cube.

    Returns
    -------
    tuple of :class:`iris.cube.Cube`:
        A tuple containing two cubes, representing the source grid and the
        destination grid, respectively.
    """
    src_rep = get_grid_representant(src)
    dst_horiz_rep = get_grid_representant(dst, horizontal_only=True)
    if src_rep.ndim == 3:
        dst_shape = (src_rep.shape[0],)
        dim_coords = [src_rep.coord(dimensions=[0], dim_coords=True)]
    else:
        dst_shape = tuple()
        dim_coords = []
    dst_shape += dst_horiz_rep.shape
    dim_coords += dst_horiz_rep.coords(dim_coords=True)
    dim_coords_and_dims = [(c, i) for i, c in enumerate(dim_coords)]
    aux_coords_and_dims = []
    for coord in dst_horiz_rep.aux_coords:
        dims = dst_horiz_rep.coord_dims(coord)
        if not dims:
            continue
        if src_rep.ndim == 3:
            dims = [dim + 1 for dim in dims]
        aux_coords_and_dims.append((coord, dims))

    # Add scalar dimensions of source cube to target
    for scalar_coord in src.coords(dimensions=()):
        aux_coords_and_dims.append((scalar_coord, ()))

    dst_rep = iris.cube.Cube(
        data=get_empty_data(dst_shape, src.dtype),
        standard_name=src.standard_name,
        long_name=src.long_name,
        var_name=src.var_name,
        units=src.units,
        attributes=src.attributes,
        cell_methods=src.cell_methods,
        dim_coords_and_dims=dim_coords_and_dims,
        aux_coords_and_dims=aux_coords_and_dims,
    )
    return src_rep, dst_rep
