"""Unstructured grid regridding."""
from __future__ import annotations

import logging
from typing import Dict

import dask.array as da
import numpy as np
from iris.cube import Cube
from scipy.spatial import Delaunay

from esmvalcore.iris_helpers import has_unstructured_grid
from iris.analysis import UnstructuredNearest as IrisUnstructuredNearest
from iris.analysis.trajectory import UnstructuredNearestNeigbourRegridder


logger = logging.getLogger(__name__)


class UnstructuredNearest(IrisUnstructuredNearest):
    """Unstructured nearest-neighbor regridding scheme."""

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


# class UnstructuredLinear:
#     """Unstructured bilinear regridding scheme."""

#     def __init__(self):
#         """Unstructured bilinear regridding scheme.

#         This class can be used in :meth:`iris.cube.Cube.regrid`.

#         """

#     def __repr__(self) -> str:
#         """String representation of class."""
#         return 'UnstructuredLinear()'

#     def regridder(self, src_cube: Cube, tgt_cube: Cube) -> Cube:
#         """Perform regridding.

#         Parameters
#         ----------
#         src_cube:
#             Source cube to be regridded.
#         tgt_cube:
#             Cube defining the target grid.

#         Returns
#         -------
#         Cube
#             Regridded cube.

#         """
#         return src_cube


# def _bilinear_unstructured_regrid(
#     src_cube: Cube,
#     tgt_cube: Cube,
# ) -> Cube:
#     """Bilinear regridding for unstructured grids.

#     The spatial dimension of the data (i.e., the one describing the
#     unstructured grid) needs to be the rightmost dimension.

#     Note
#     ----
#     This private function has been introduced to regrid native ERA5 in GRIB
#     format similarly to how it is done if you download an interpolated versions
#     of ERA5 (see
#     https://confluence.ecmwf.int/display/CKB/ERA5%3A+What+is+the+spatial+reference).

#     Currently, we do not support bilinear regridding for unstructured grids in
#     our `regrid` preprocessor (only nearest-neighbor). Since iris is currently
#     doing a massive overhaul of their in-built regridding
#     (https://github.com/SciTools/iris/issues/4754), it does not make sense to
#     include the following piece of code in their package just now.

#     Thus, we provide this function here. Please be aware that it can be removed
#     at any point in time without prior warning (just like any other private
#     function).

#     Warning
#     -------
#     This function will drop all cell measures, ancillary variables and aux
#     factories, and any auxiliary coordinate that spans the spatial dimension.

#     """
#     # This function should only be called on unstructured grid cubes
#     if not has_unstructured_grid(src_cube):
#         raise ValueError(
#             f"Cube {src_cube.summary(shorten=True)} does not have unstructured "
#             f"grid"
#         )

#     # The unstructured grid dimension needs to be the rightmost dimension
#     udim = src_cube.coord_dims('latitude')[0]
#     if udim != src_cube.ndim - 1:
#         raise ValueError(
#             f"The spatial dimension of cube {src_cube.summary(shorten=True)} "
#             f"(i.e, the one describing the unstructured grid) needs to be the "
#             f"rightmost dimension"
#         )

#     # Make sure the cube has lazy data and rechunk it properly (cube cannot be
#     # chunked along latitude and longitude dimension)
#     if not src_cube.has_lazy_data():
#         src_cube.data = da.from_array(src_cube.data)
#     in_chunks = ['auto'] * src_cube.ndim
#     in_chunks[udim] = -1  # type: ignore
#     src_cube.data = src_cube.lazy_data().rechunk(in_chunks)

#     # Calculate indices and interpolation weights
#     (indices, weights) = _get_linear_interpolation_weights(src_cube, tgt_cube)

#     # Perform actual regridding
#     regridded_data = da.apply_gufunc(
#         _interpolate,
#         '(i),(j,3),(j,3)->(j)',
#         src_cube.lazy_data(),
#         indices,
#         weights,
#         vectorize=True,
#         output_dtypes=src_cube.dtype,
#     )
#     regridded_data = regridded_data.rechunk('auto')

#     # Create new cube with correct metadata
#     dim_coords_and_dims = [
#         (c, src_cube.coord_dims(c)) for c in src_cube.coords(dim_coords=True) if
#         udim not in src_cube.coord_dims(c)
#     ]
#     dim_coords_and_dims.extend([
#         (tgt_cube.coord('latitude'), src_cube.ndim - 1),
#         (tgt_cube.coord('longitude'), src_cube.ndim),
#     ])
#     aux_coords_and_dims = [
#         (c, src_cube.coord_dims(c)) for c in src_cube.coords(dim_coords=False) if
#         udim not in src_cube.coord_dims(c)
#     ]
#     new_shape = src_cube.shape[:-1] + tgt_cube.shape
#     regridded_cube = Cube(
#         regridded_data.reshape(new_shape, limit='128MiB'),
#         standard_name=src_cube.standard_name,
#         long_name=src_cube.long_name,
#         var_name=src_cube.var_name,
#         units=src_cube.units,
#         attributes=src_cube.attributes,
#         cell_methods=src_cube.cell_methods,
#         dim_coords_and_dims=dim_coords_and_dims,
#         aux_coords_and_dims=aux_coords_and_dims,
#     )

#     return regridded_cube


# _CACHE_WEIGHTS: Dict[str, tuple[np.ndarray, np.ndarray]] = {}


# def _get_linear_interpolation_weights(
#     src_cube: Cube,
#     tgt_cube: Cube,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Get vertices and weights for 2D linear regridding of unstructured grids.

#     Partly taken from
#     https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids.
#     This is more than 80x faster than :func:`scipy.interpolate.griddata` and
#     gives identical results.

#     """
#     # Cache result to avoid re-calculating this over and over
#     src_lat = src_cube.coord('latitude')
#     src_lon = src_cube.coord('longitude')
#     tgt_lat = tgt_cube.coord('latitude')
#     tgt_lon = tgt_cube.coord('longitude')
#     cache_key = (
#         f"{src_lat.shape}_"
#         f"{src_lat.points[0]}-{src_lat.points[-1]}-{src_lat.units}_"
#         f"{src_lon.points[0]}-{src_lon.points[-1]}-{src_lon.units}_"
#         f"{tgt_lat.shape}_"
#         f"{tgt_lat.points[0]}-{tgt_lat.points[-1]}-{tgt_lat.units}_"
#         f"{tgt_lon.points[0]}-{tgt_lon.points[-1]}-{tgt_lon.units}_"
#     )
#     if cache_key in _CACHE_WEIGHTS:
#         return _CACHE_WEIGHTS[cache_key]

#     # Bring points into correct format
#     # src_points: (N, 2) where N is the number of source grid points
#     # tgt_points: (M, 2) where M is the number of target grid points
#     src_points = np.stack((src_lat.points, src_lon.points), axis=-1)
#     (tgt_lat, tgt_lon) = np.meshgrid(
#         tgt_cube.coord('latitude').points,
#         tgt_cube.coord('longitude').points,
#         indexing='ij',
#     )
#     tgt_points = np.stack((tgt_lat.ravel(), tgt_lon.ravel()), axis=-1)

#     # Actual indices and weights calculation using Delaunay triagulation
#     # Return shapes:
#     # indices: (M, 3)
#     # weights: (M, 3)
#     n_dims = 2
#     tri = Delaunay(src_points)
#     simplex = tri.find_simplex(tgt_points)
#     extra_idx = (simplex == -1)
#     indices = np.take(tri.simplices, simplex, axis=0)
#     temp = np.take(tri.transform, simplex, axis=0)
#     delta = tgt_points - temp[:, n_dims]
#     bary = np.einsum('njk,nk->nj', temp[:, :n_dims, :], delta)
#     weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
#     weights[extra_idx, :] = np.nan  # missing values

#     # Cache result
#     _CACHE_WEIGHTS[cache_key] = (indices, weights)

#     return (indices, weights)


# def _interpolate(
#     data: np.ndarray,
#     indices: np.ndarray,
#     weights: np.ndarray,
# ) -> np.ndarray:
#     """Interpolate data.

#     Parameters
#     ----------
#     data: np.ndarray
#         Data to interpolate. Must be an (N,) array, where N is the number of
#         source grid points.
#     indices: np.ndarray
#         Indices used to index the data. Must be an (M, 3) array, where M is the
#         number of target grid points.
#     weights: np.ndarray
#         Interpolation weights. Must be an (M, 3) array, where M is the number
#         of target grid points.

#     Returns
#     -------
#     np.ndarray
#         Interpolated data of shape (M,).

#     """
#     return np.einsum('nj,nj->n', np.take(data, indices), weights)
