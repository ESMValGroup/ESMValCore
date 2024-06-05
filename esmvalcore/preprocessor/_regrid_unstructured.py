"""Unstructured grid regridding."""
from __future__ import annotations

import logging

import dask
import dask.array as da
import numpy as np
from iris.analysis import UnstructuredNearest as IrisUnstructuredNearest
from iris.analysis.trajectory import UnstructuredNearestNeigbourRegridder
from iris.coords import Coord
from iris.cube import Cube
from numpy.typing import DTypeLike
from scipy.spatial import ConvexHull, Delaunay

from esmvalcore.iris_helpers import (
    has_regular_grid,
    has_unstructured_grid,
    rechunk_cube,
)
from esmvalcore.preprocessor._shared import preserve_float_dtype

logger = logging.getLogger(__name__)


class UnstructuredNearest(IrisUnstructuredNearest):
    """Unstructured nearest-neighbor regridding scheme.

    This class is a wrapper around :class:`iris.analysis.UnstructuredNearest`
    that removes any additional X or Y coordinates prior to regridding if
    necessary. It can be used in :meth:`iris.cube.Cube.regrid`.

    """

    def regridder(
        self,
        src_cube: Cube,
        tgt_cube: Cube,
    ) -> UnstructuredNearestNeigbourRegridder:
        """Get regridder.

        Parameters
        ----------
        src_cube:
            Cube defining the source grid.
        tgt_cube:
            Cube defining the target grid.

        Returns
        -------
        UnstructuredNearestNeigbourRegridder
            Regridder instance.

        """
        # Unstructured nearest-neighbor regridding requires exactly one X and
        # one Y coordinate (latitude and longitude). Remove any X or Y
        # dimensional coordinates if necessary.
        for axis in ['x', 'y']:
            if src_cube.coords(axis=axis, dim_coords=True):
                coord = src_cube.coord(axis=axis, dim_coords=True)
                src_cube.remove_coord(coord)
        return super().regridder(src_cube, tgt_cube)


class UnstructuredLinearRegridder:
    """Unstructured bilinear regridder.

    Supports lazy regridding and weights caching.

    Warning
    -------
    This will drop all cell measures, ancillary variables and aux factories,
    and any auxiliary coordinate that spans the dimension of the unstrucutred
    grid.

    Parameters
    ----------
    src_cube:
        Cube defining the source grid.
    tgt_cube:
        Cube defining the target grid.

    """

    def __init__(self, src_cube: Cube, tgt_cube: Cube) -> None:
        """Initialize class instance."""
        if not has_unstructured_grid(src_cube):
            raise ValueError(
                f"Source cube {src_cube.summary(shorten=True)} does not have "
                f"unstructured grid"
            )
        if not has_regular_grid(tgt_cube):
            raise ValueError(
                f"Target cube {tgt_cube.summary(shorten=True)} does not have "
                f"regular grid"
            )
        src_lat = src_cube.coord('latitude').copy()
        src_lon = src_cube.coord('longitude').copy()
        tgt_lat = tgt_cube.coord('latitude').copy()
        tgt_lon = tgt_cube.coord('longitude').copy()
        self.src_coords = [src_lat, src_lon]
        self.tgt_coords = [tgt_lat, tgt_lon]
        self.tgt_n_lat = tgt_lat.core_points().size
        self.tgt_n_lon = tgt_lon.core_points().size

        # Calculate regridding weights and indices
        (self._weights, self._indices,
         self._convex_hull_idx) = self._get_weights_and_idx(
            src_lat, src_lon, tgt_lat, tgt_lon)

    def _get_weights_and_idx(
        self,
        src_lat: Coord,
        src_lon: Coord,
        tgt_lat: Coord,
        tgt_lon: Coord,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get regridding weights and indices.

        Note
        ----
        To consider the periodic boundary conditions of a sphere, the source
        points are expanded first with all points on their convex hull wrapped
        around by -360° and +360° before the weights calculation. This needs to
        be considered in the interpolation (see self._interpolate)!

        The output arrays will be numpy arrays here (instead of dask) since
        resulting arrays are only 2D and will be computed anyway later during
        regridding.

        """
        # Make sure that source and target grid have identical units
        src_lat = src_lat.copy()
        src_lon = src_lon.copy()
        tgt_lat = tgt_lat.copy()
        tgt_lon = tgt_lon.copy()
        src_lat.convert_units('degrees')
        src_lon.convert_units('degrees')
        tgt_lat.convert_units('degrees')
        tgt_lon.convert_units('degrees')

        # Bring points into correct format
        # src_points: (N, 2) where N is the number of source grid points
        # tgt_points: (M, 2) where M is the number of target grid points
        (src_points, tgt_points) = dask.compute(
            np.stack((src_lat.core_points(), src_lon.core_points()), axis=-1),
            np.stack(
                tuple(
                    tgt_coord.ravel() for tgt_coord in
                    np.meshgrid(
                        tgt_lat.core_points(),
                        tgt_lon.core_points(),
                        indexing='ij',
                    )
                ),
                axis=-1,
            ),
        )
        src_points = np.array(src_points)  # cannot be masked array
        tgt_points = np.array(tgt_points)  # cannot be masked array

        # Calculate convex hull of source points to consider periodic boundary
        # conditions
        hull = ConvexHull(src_points)
        n_hull = len(hull.vertices)

        # Wrap around points on convex hull by -360° and +360° and add them to
        # list of source points
        src_points_with_convex_hull = self._add_convex_hull_twice(
            src_points, hull.vertices
        )
        src_points_with_convex_hull[-2 * n_hull:-n_hull, 1] -= 360
        src_points_with_convex_hull[-n_hull:, 1] += 360

        # Actual weights calculation
        (weights, indices) = self._calculate_weights(
            src_points_with_convex_hull, tgt_points
        )

        return (weights, indices, hull.vertices)

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
        if not has_unstructured_grid(cube):
            raise ValueError(
                f"Cube {cube.summary(shorten=True)} does not have "
                f"unstructured grid"
            )
        coords = [cube.coord('latitude'), cube.coord('longitude')]
        if coords != self.src_coords:
            raise ValueError(
                f"The given cube {cube.summary(shorten=True)} is not defined "
                f"on the same source grid as this regridder"
            )

        # Get coordinates of regridded cube

        # (1) New dimensional coordinates are the ones from the source cube
        # (excluding the unstructured grid dimension) plus the (x, y) target
        # grid dimensions. All dimensions to the right of the unstructured grid
        # dimension need to be shifted to the right by 1.
        udim = cube.coord_dims('latitude')[0]
        dim_coords_and_dims = [
            (c, cube.coord_dims(c)[0]) for c in cube.coords(dim_coords=True) if
            udim not in cube.coord_dims(c)
        ]
        dim_coords_and_dims = [
            (c, d) if d < udim else (c, d + 1) for (c, d) in
            dim_coords_and_dims
        ]
        dim_coords_and_dims.append((self.tgt_coords[0], udim))
        dim_coords_and_dims.append((self.tgt_coords[1], udim + 1))

        # (2) Include all auxiliary coordinates that do not span unstructured
        # grid dimension (also make sure to shift all dimensions to the right
        # of the unstructured grid to the right by 1)
        old_aux_coords_and_dims = [
            (c, cube.coord_dims(c)) for c in cube.coords(dim_coords=False) if
            udim not in cube.coord_dims(c)
        ]
        aux_coords_and_dims = []
        for (aux_coord, dims) in old_aux_coords_and_dims:
            dims = tuple(d if d < udim else d + 1 for d in dims)
            aux_coords_and_dims.append((aux_coord, dims))

        # Create new cube with regridded data
        regridded_data = preserve_float_dtype(self._get_regridded_data)(cube)
        regridded_cube = Cube(
            regridded_data,
            dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims,
        )
        regridded_cube.metadata = cube.metadata

        return regridded_cube

    def _get_regridded_data(self, cube: Cube) -> np.ndarray | da.Array:
        """Get regridded data."""
        udim = cube.coord_dims('latitude')[0]

        # Cube must not be chunked along latitude and longitude dimension
        rechunk_cube(cube, ['latitude', 'longitude'])

        # Make sure that masked arrays are filled with nan's so they are
        # handled properly
        npx = da if cube.has_lazy_data() else np
        src_data = npx.ma.filled(cube.core_data(), np.nan)

        # Perform actual regridding and assign correct mask
        regridded_data: np.ndarray | da.Array
        if cube.has_lazy_data():
            regridded_data = self._regrid_lazy(
                src_data, udim, self._weights.dtype
            )
        else:
            regridded_data = self._regrid_eager(src_data, udim)
        regridded_data = npx.ma.masked_invalid(regridded_data)

        return regridded_data

    def _regrid_eager(self, data: np.ndarray, axis: int) -> np.ndarray:
        """Eager regridding."""
        v_interpolate = np.vectorize(
            self._interpolate, signature='(i)->(lat,lon)'
        )

        # Make sure that interpolation dimension is rightmost dimension and
        # change it back after regridding
        data = np.moveaxis(data, axis, -1)
        regridded_arr = v_interpolate(data)
        regridded_arr = np.moveaxis(regridded_arr, -2, axis)
        regridded_arr = np.moveaxis(regridded_arr, -1, axis + 1)

        return regridded_arr

    def _regrid_lazy(
        self,
        data: da.Array,
        axis: int,
        dtype: DTypeLike,
    ) -> da.Array:
        """Lazy regridding."""
        regridded_arr = da.apply_gufunc(
            self._interpolate,
            '(i)->(lat,lon)',
            data,
            axes=[(axis,), (axis, axis + 1)],
            vectorize=True,
            output_dtypes=dtype,
            output_sizes={'lat': self.tgt_n_lat, 'lon': self.tgt_n_lon},
        )
        return regridded_arr

    def _interpolate(self, data: np.ndarray) -> np.ndarray:
        """Interpolate data.

        Data to interpolate must be an (N,) array, where N is the number of
        source grid points. Indices used to index the data and interpolation
        weights must be (M, 3) arrays, where M is the number of target grid
        points.

        The returned array is of shape (lat, lon), where lat is the number of
        latitudes in the target grid, and lon the number of longitudes in the
        target grid such that lat x lon = M.

        Note
        ----
        Before the interpolation, the input is extended by the data on the
        convex hull to consider the periodic boundary conditions of a sphere
        (this has also been done for the weights calculation).

        """
        data = self._add_convex_hull_twice(data, self._convex_hull_idx)
        interp_data = np.einsum(
            'nj,nj->n', np.take(data, self._indices), self._weights
        )
        interp_data = interp_data.reshape(self.tgt_n_lat, self.tgt_n_lon)
        return interp_data

    @staticmethod
    def _add_convex_hull_twice(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Expand array with convex hull values (given by indices) twice."""
        convex_hull = arr[idx]
        return np.concatenate((arr, convex_hull, convex_hull))

    @staticmethod
    def _calculate_weights(
        src_points: np.ndarray,
        tgt_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate regridding weights using Delaunay triagulation.

        Partly taken from https://stackoverflow.com/a/20930910.

        Output shapes (M: number of target grid points)
        - weights: (M, 3)
        - indices: (M, 3)

        """
        tri = Delaunay(src_points)
        simplex = tri.find_simplex(tgt_points)
        indices = np.take(tri.simplices, simplex, axis=0)
        transform = np.take(tri.transform, simplex, axis=0)
        delta = tgt_points - transform[:, 2]
        bary = np.einsum('njk,nk->nj', transform[:, :2, :], delta)
        weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
        extra_idx = simplex == -1
        weights[extra_idx, :] = np.nan  # missing values
        return (weights, indices)


class UnstructuredLinear:
    """Unstructured bilinear regridding scheme.

    This class can be used in :meth:`iris.cube.Cube.regrid`.

    Supports lazy regridding.

    Warning
    -------
    This will drop all cell measures, ancillary variables and aux factories,
    and any auxiliary coordinate that spans the dimension of the unstrucutred
    grid.

    """

    def __repr__(self) -> str:
        """Return string representation of class."""
        return 'UnstructuredLinear()'

    def regridder(
        self,
        src_cube: Cube,
        tgt_cube: Cube,
    ) -> UnstructuredLinearRegridder:
        """Get regridder.

        Parameters
        ----------
        src_cube:
            Cube defining the source grid.
        tgt_cube:
            Cube defining the target grid.

        Returns
        -------
        UnstructuredLinearRegridder
            Regridder instance.

        """
        return UnstructuredLinearRegridder(src_cube, tgt_cube)
