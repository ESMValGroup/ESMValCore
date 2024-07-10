"""Iris-esmf-regrid based regridding scheme."""
from __future__ import annotations

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


class IrisESMFRegrid:
    """Iris-esmf-regrid based regridding scheme.

    Supports lazy regridding.

    Parameters
    ----------
    method:
        Either "conservative", "bilinear" or "nearest". Corresponds to the
        :mod:`esmpy` methods
        :attr:`~esmpy.api.constants.RegridMethod.CONSERVE`,
        :attr:`~esmpy.api.constants.RegridMethod.BILINEAR` or
        :attr:`~esmpy.api.constants.RegridMethod.NEAREST_STOD` used to
        calculate weights.
    mdtol:
        Tolerance of missing data. The value returned in each element of
        the returned array will be masked if the fraction of masked data
        exceeds ``mdtol``. ``mdtol=0`` means no missing data is tolerated while
        ``mdtol=1`` will mean the resulting element will be masked if and only
        if all the contributing elements of data are masked. If no value is
        given, this will default to 1 for conservative regridding and 0
        otherwise. Only available for methods 'bilinear' and 'conservative'.
    use_src_mask:
        If True, derive a mask from (first time step) of the source cube,
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
        method: Literal['bilinear', 'conservative', 'nearest'],
        mdtol: float | None = None,
        use_src_mask: bool | np.ndarray = True,
        use_tgt_mask: bool | np.ndarray = True,
        src_resolution: int | None = None,
        tgt_resolution: int | None = None,
        tgt_location: Literal['face', 'node'] | None = None,
    ) -> None:
        if method not in METHODS:
            raise ValueError(
                "`method` should be one of 'bilinear', 'conservative', or "
                "'nearest'")

        self.kwargs: dict[str, Any] = {
            'method': method,
            'use_src_mask': use_src_mask,
            'use_tgt_mask': use_tgt_mask,
            'tgt_location': tgt_location,
        }
        if method == 'nearest':
            if mdtol is not None:
                raise ValueError(
                    "`mdol` can only be specified when `method='bilinear'` "
                    "or `method='bilinear'`")
        else:
            self.kwargs['mdtol'] = mdtol
        if method == 'conservative':
            self.kwargs['src_resolution'] = src_resolution
            self.kwargs['tgt_resolution'] = tgt_resolution
        elif src_resolution is not None:
            raise ValueError("`src_resolution` can only be specified when "
                             "`method='conservative'`")
        elif tgt_resolution is not None:
            raise ValueError("`tgt_resolution` can only be specified when "
                             "`method='conservative'`")

    def __repr__(self) -> str:
        """Return string representation of class."""
        kwargs_str = ", ".join(f"{k}={repr(v)}"
                               for k, v in self.kwargs.items())
        return f'{self.__class__.__name__}({kwargs_str})'

    @staticmethod
    def _get_mask(cube):
        """Read the mask from the cube data.

        If the cube has a vertical dimension, the mask will consist of
        those points which are masked in all vertical levels.

        This function assumes that the mask is constant in dimensions
        that are not horizontal or vertical.
        """

        def _get_coord(cube, axis):
            try:
                coord = cube.coord(axis=axis, dim_coords=True)
            except iris.exceptions.CoordinateNotFoundError:
                coord = cube.coord(axis=axis)
            return coord

        src_x, src_y = (_get_coord(cube, "x"), _get_coord(cube, "y"))
        horizontal_dims = tuple(
            set(cube.coord_dims(src_x) + cube.coord_dims(src_y)))
        try:
            vertical_coord = cube.coord(axis="z", dim_coords=True)
        except iris.exceptions.CoordinateNotFoundError:
            vertical_coord = None

        data = cube.core_data()
        if vertical_coord is None:
            slices = tuple(
                slice(None) if i in horizontal_dims else 0
                for i in range(cube.ndim))
            mask = da.ma.getmaskarray(data[slices])
        else:
            vertical_dim = cube.coord_dims(vertical_coord)[0]
            slices = tuple(
                slice(None) if i in horizontal_dims + (vertical_dim, ) else 0
                for i in range(cube.ndim))
            mask = da.ma.getmaskarray(data[slices])
            mask_vertical_dim = sum(i < vertical_dim for i in horizontal_dims)
            mask = mask.all(axis=mask_vertical_dim)
        return mask

    def regridder(
        self,
        src_cube: iris.cube.Cube,
        tgt_cube: iris.cube.Cube,
    ) -> (ESMFAreaWeightedRegridder
          | ESMFBilinearRegridder
          | ESMFNearestRegridder):
        """Create an iris-esmf-regrid based regridding function.

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
        if src_mask is True:
            src_mask = self._get_mask(src_cube)
        tgt_mask = kwargs.pop('use_tgt_mask')
        if tgt_mask is True:
            tgt_mask = self._get_mask(tgt_cube)
        src_mask, tgt_mask = dask.compute(src_mask, tgt_mask)
        return regridder_cls(
            src_cube,
            tgt_cube,
            use_src_mask=src_mask,
            use_tgt_mask=tgt_mask,
            **kwargs,
        )
