"""Unstructured grid regridding."""
from __future__ import annotations

import logging

import dask
import dask.array as da
import numpy as np
from iris.analysis import UnstructuredNearest as IrisUnstructuredNearest
from iris.analysis.trajectory import UnstructuredNearestNeigbourRegridder
from iris.cube import Cube
from numpy.typing import DTypeLike
from scipy.spatial import Delaunay

from esmvalcore.iris_helpers import (
    has_regular_grid,
    has_unstructured_grid,
    rechunk_cube,
)

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

        # Calculate regridding weights
        # Note: we force numpy arrays here (instead of dask) since resulting
        # arrays are only 2D and will be computed anyway later during
        # regridding

        # (1) Bring points into correct format
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

        # (2) Actual indices and weights calculation using Delaunay
        # triagulation (partly taken from https://stackoverflow.com/a/20930910)
        # Array shapes:
        # indices: (M, 3)
        # weights: (M, 3)
        tri = Delaunay(np.array(src_points))
        simplex = tri.find_simplex(np.array(tgt_points))
        indices = np.take(tri.simplices, simplex, axis=0)
        transform = np.take(tri.transform, simplex, axis=0)
        delta = tgt_points - transform[:, 2]
        bary = np.einsum('njk,nk->nj', transform[:, :2, :], delta)
        weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
        extra_idx = simplex == -1
        weights[extra_idx, :] = np.nan  # missing values

        self.indices = indices
        self.weights = weights

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

        # Get regridded data
        regridded_data = self._get_regridded_data(cube)

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
            dims = tuple([d if d < udim else d + 1 for d in dims])
            aux_coords_and_dims.append((aux_coord, dims))

        # Create new cube
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

        shape = [s for (d, s) in enumerate(cube.shape) if d != udim]
        shape.insert(udim, self.tgt_coords[0].core_points().size)
        shape.insert(udim + 1, self.tgt_coords[1].core_points().size)
        new_shape = tuple(shape)

        # Perform actual regridding and assign correct mask
        regridded_data: np.ndarray | da.Array
        if cube.has_lazy_data():
            regridded_data = self._regrid_lazy(
                src_data, udim, self.weights.dtype, self.weights.shape[0]
            )
            # TODO: add limit='128 MiB' to reshape once dask bug is solved
            # see https://github.com/dask/dask/issues/10603
            regridded_data = regridded_data.reshape(new_shape)
        else:
            regridded_data = self._regrid_eager(src_data, udim)
            regridded_data = regridded_data.reshape(new_shape)
        regridded_data = npx.ma.masked_invalid(regridded_data)

        # Ensure correct dtype
        if regridded_data.dtype != cube.dtype:
            regridded_data = regridded_data.astype(cube.dtype)

        return regridded_data

    def _interpolate(self, arr: np.ndarray) -> np.ndarray:
        """Interpolate data.

        Takes input array of shape (N,) (N: number of source points) and
        returns array of shape (M,) (M: number of target points).

        """
        return np.einsum('nj,nj->n', np.take(arr, self.indices), self.weights)

    def _regrid_eager(self, arr: np.ndarray, axis: int) -> np.ndarray:
        """Eager regridding."""
        v_interpolate = np.vectorize(self._interpolate, signature='(i)->(j)')

        # Make sure that interpolation dimension is rightmost dimension and
        # change it back after regridding
        arr = np.moveaxis(arr, axis, -1)
        regridded_arr = v_interpolate(arr)
        regridded_arr = np.moveaxis(regridded_arr, -1, axis)

        return regridded_arr

    def _regrid_lazy(
        self,
        arr: da.Array,
        axis: int,
        dtype: DTypeLike,
        target_grid_size: int,
    ) -> da.Array:
        """Lazy regridding."""
        regridded_arr = da.apply_gufunc(
            self._interpolate,
            '(i)->(j)',
            arr,
            axes=[(axis,), (axis,)],
            vectorize=True,
            output_dtypes=dtype,
            output_sizes={'j': target_grid_size},
        )
        return regridded_arr


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
