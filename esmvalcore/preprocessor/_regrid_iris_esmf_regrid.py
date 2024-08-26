"""Iris-esmf-regrid based regridding scheme."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import dask
import dask.array as da
import iris.cube
import iris.exceptions
import numpy as np
from esmf_regrid import (
    ESMFAreaWeightedRegridder,
    ESMFBilinearRegridder,
    ESMFNearestRegridder,
)

METHODS = {
    'conservative': ESMFAreaWeightedRegridder,
    'bilinear': ESMFBilinearRegridder,
    'nearest': ESMFNearestRegridder,
}


def _get_dims_along_axes(
    cube: iris.cube.Cube,
    axes: Iterable[Literal["T", "Z", "Y", "X"]],
) -> tuple[int, ...]:
    """Get a tuple with the dimensions along one or more axis."""

    def _get_dims_along_axis(cube, axis):
        try:
            coord = cube.coord(axis=axis, dim_coords=True)
        except iris.exceptions.CoordinateNotFoundError:
            try:
                coord = cube.coord(axis=axis)
            except iris.exceptions.CoordinateNotFoundError:            
                return tuple()
        return cube.coord_dims(coord)

    dims = {d for axis in axes for d in _get_dims_along_axis(cube, axis)}
    return tuple(sorted(dims))


class IrisESMFRegrid:
    """:doc:`esmf_regrid:index` based regridding scheme.

    Supports lazy regridding.

    Parameters
    ----------
    method:
        Either "conservative", "bilinear" or "nearest". Corresponds to the
        :mod:`esmpy` methods
        :attr:`~esmpy.api.constants.RegridMethod.CONSERVE`,
        :attr:`~esmpy.api.constants.RegridMethod.BILINEAR` or
        :attr:`~esmpy.api.constants.RegridMethod.NEAREST_STOD` used to
        calculate regridding weights.
    mdtol:
        Tolerance of missing data. The value returned in each element of
        the returned array will be masked if the fraction of masked data
        exceeds ``mdtol``. ``mdtol=0`` means no missing data is tolerated while
        ``mdtol=1`` will mean the resulting element will be masked if and only
        if all the contributing elements of data are masked. If no value is
        given, this will default to 1 for conservative regridding and 0
        otherwise. Only available for methods 'bilinear' and 'conservative'.
    use_src_mask:
        If True, derive a mask from the source cube data,
        which will tell :mod:`esmpy` which points to ignore. If an array is
        provided, that will be used.
        If set to :obj:`None`, it will be set to :obj:`True` for methods
        'bilinear' and `conservative' and to :obj:`False` for method 'nearest'.
    use_tgt_mask:
        If True, derive a mask from (first time step) of the target cube,
        which will tell :mod:`esmpy` which points to ignore. If an array is
        provided, that will be used.
        If set to :obj:`None`, it will be set to :obj:`True` for methods
        `bilinear' and 'conservative' and to :obj:`False` for method 'nearest'.
    collapse_src_mask_along:
        When deriving the mask from the source cube data, collapse the mask
        along the dimensions idenfied by these axes. Only points that are
        masked at all time (``'T'``), vertical (``'Z'``), or both time and
        vertical points will be considered masked.
    collapse_tgt_mask_along:
        When deriving the mask from the target cube data, collapse the mask
        along the dimensions idenfied by these axes. Only points that are
        masked at all time (``'T'``), vertical (``'Z'``), or both time and
        vertical points will be considered masked.
    src_resolution:
        If present, represents the amount of latitude slices per source cell
        given to ESMF for calculation. If resolution is set, the source cube
        must have strictly increasing bounds (bounds may be transposed
        plus or minus 360 degrees to make the bounds strictly increasing).
        Only available for method 'conservative'.
    tgt_resolution:
        If present, represents the amount of latitude slices per target cell
        given to ESMF for calculation. If resolution is set, the target cube
        must have strictly increasing bounds (bounds may be transposed
        plus or minus 360 degrees to make the bounds strictly increasing).
        Only available for method 'conservative'.
    tgt_location:
        Either "face" or "node". Describes the location for data on the mesh
        if the target is not a :class:`~iris.cube.Cube`.

    Attributes
    ----------
    kwargs:
        Keyword arguments that will be provided to the regridder.
    """

    def __init__(
        self,
        method: Literal["bilinear", "conservative", "nearest"],
        mdtol: float | None = None,
        use_src_mask: None | bool | np.ndarray = None,
        use_tgt_mask: None | bool | np.ndarray = None,
        collapse_src_mask_along: Iterable[Literal['T', 'Z']] = ('Z', ),
        collapse_tgt_mask_along: Iterable[Literal['T', 'Z']] = ('Z', ),
        src_resolution: int | None = None,
        tgt_resolution: int | None = None,
        tgt_location: Literal['face', 'node'] | None = None,
    ) -> None:
        if method not in METHODS:
            raise ValueError(
                "`method` should be one of 'bilinear', 'conservative', or "
                "'nearest'")

        if use_src_mask is None:
            use_src_mask = False if method == "nearest" else True
        if use_tgt_mask is None:
            use_tgt_mask = False if method == "nearest" else True

        self.kwargs: dict[str, Any] = {
            'method': method,
            'use_src_mask': use_src_mask,
            'use_tgt_mask': use_tgt_mask,
            'collapse_src_mask_along': collapse_src_mask_along,
            'collapse_tgt_mask_along': collapse_tgt_mask_along,
            'tgt_location': tgt_location,
        }
        if method == 'nearest':
            if mdtol is not None:
                raise TypeError(
                    "`mdol` can only be specified when `method='bilinear'` "
                    "or `method='conservative'`")
        else:
            self.kwargs['mdtol'] = mdtol
        if method == 'conservative':
            self.kwargs['src_resolution'] = src_resolution
            self.kwargs['tgt_resolution'] = tgt_resolution
        elif src_resolution is not None:
            raise TypeError("`src_resolution` can only be specified when "
                            "`method='conservative'`")
        elif tgt_resolution is not None:
            raise TypeError("`tgt_resolution` can only be specified when "
                            "`method='conservative'`")

    def __repr__(self) -> str:
        """Return string representation of class."""
        kwargs_str = ", ".join(f"{k}={repr(v)}"
                               for k, v in self.kwargs.items())
        return f'{self.__class__.__name__}({kwargs_str})'

    @staticmethod
    def _get_mask(
        cube: iris.cube.Cube,
        collapse_mask_along: Iterable[Literal['T', 'Z']] = ('Z', ),
    ) -> np.ndarray:
        """Read the mask from the cube data.

        This function assumes that the mask is constant in dimensions
        that are not horizontal or specified in `collapse_mask_along`.
        """
        horizontal_dims = _get_dims_along_axes(cube, ["X", "Y"])
        other_dims = _get_dims_along_axes(cube, collapse_mask_along)

        slices = tuple(
            slice(None) if i in horizontal_dims + other_dims else 0
            for i in range(cube.ndim)
        )
        subcube = cube[slices]
        subcube_other_dims = _get_dims_along_axes(subcube, collapse_mask_along)

        mask = da.ma.getmaskarray(subcube.core_data())
        return mask.all(axis=subcube_other_dims)

    def regridder(
        self,
        src_cube: iris.cube.Cube,
        tgt_cube: iris.cube.Cube,
    ) -> (ESMFAreaWeightedRegridder
          | ESMFBilinearRegridder
          | ESMFNearestRegridder):
        """Create an :doc:`esmf_regrid:index` based regridding function.

        Parameters
        ----------
        src_cube:
            Cube defining the source grid.
        tgt_cube:
            Cube defining the target grid.

        Returns
        -------
        :obj:`esmf_regrid.ESMFAreaWeightedRegridder` or
        :obj:`esmf_regrid.ESMFBilinearRegridder` or
        :obj:`esmf_regrid.ESMFNearestRegridder`:
            iris-esmf-regrid regridding function.
        """
        kwargs = self.kwargs.copy()
        regridder_cls = METHODS[kwargs.pop('method')]
        src_mask = kwargs.pop('use_src_mask')
        collapse_mask_along = kwargs.pop('collapse_src_mask_along')
        if src_mask is True:
            src_mask = self._get_mask(src_cube, collapse_mask_along)
        tgt_mask = kwargs.pop('use_tgt_mask')
        collapse_mask_along = kwargs.pop('collapse_tgt_mask_along')
        if tgt_mask is True:
            tgt_mask = self._get_mask(tgt_cube, collapse_mask_along)
        src_mask, tgt_mask = dask.compute(src_mask, tgt_mask)
        return regridder_cls(
            src_cube,
            tgt_cube,
            use_src_mask=src_mask,
            use_tgt_mask=tgt_mask,
            **kwargs,
        )
