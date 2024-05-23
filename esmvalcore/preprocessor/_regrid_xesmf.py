"""xESMF regridding.

To use this, install xesmf and ncdata, e.g. ``mamba install xesmf ncdata``.
"""

import inspect

import dask.array as da
import iris.cube
import numpy as np


class xESMFRegridder:  # noqa
    """xESMF regridding function.

    This is a wrapper around :class:`xesmf.Regrid` so it can be used in
    :meth:`iris.cube.Cube.regrid`.

    Supports lazy regridding.

    Parameters
    ----------
    src_cube:
        Cube describing the source grid.
    tgt_cube:
        Cube describing the target grid.
    **kwargs:
        Any keyword argument to :class:`xesmf.Regrid` or
        :meth:`xesmf.Regrid.__call__` can be provided.

    Attributes
    ----------
    kwargs:
        Keyword arguments to :class:`xesmf.Regrid`.
    default_call_kwargs:
        Default keyword arguments to :meth:`xesmf.Regrid.__call__`.
    """

    def __init__(
        self,
        src_cube: iris.cube.Cube,
        tgt_cube: iris.cube.Cube,
        **kwargs,
    ) -> None:
        import ncdata.iris_xarray
        import xesmf

        call_arg_names = list(
            inspect.signature(xesmf.Regridder.__call__).parameters
        )[2:]
        self.kwargs = {
            k: v
            for k, v in kwargs.items() if k not in call_arg_names
        }
        self.default_call_kwargs = {
            k: v
            for k, v in kwargs.items() if k in call_arg_names
        }

        src_ds = ncdata.iris_xarray.cubes_to_xarray([src_cube])
        tgt_ds = ncdata.iris_xarray.cubes_to_xarray([tgt_cube])
        for var in src_ds.values():
            var.data = da.ma.filled(var.data, np.nan)
        for var in tgt_ds.values():
            var.data = da.ma.filled(var.data, np.nan)

        self._regridder = xesmf.Regridder(src_ds, tgt_ds, **self.kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the class."""
        kwargs = self.kwargs | self.default_call_kwargs
        return f"{self.__class__.__name__}(**{kwargs})"

    def __call__(self, src_cube: iris.cube.Cube, **kwargs) -> iris.cube.Cube:
        """Run the regridder.

        Parameters
        ----------
        src_cube:
            The cube to regrid.
        **kwargs:
            Keyword arguments to :meth:`xesmf.Regrid.__call__`.

        Returns
        -------
        iris.cube.Cube
            The regridded cube.
        """
        import ncdata.iris_xarray

        src_ds = ncdata.iris_xarray.cubes_to_xarray([src_cube])
        for var in src_ds.values():
            var.data = da.ma.filled(var.data, np.nan)

        call_args = dict(self.default_call_kwargs)
        call_args.update(kwargs)
        tgt_ds = self._regridder(src_ds, **call_args)
        for var in tgt_ds.values():
            var.data = da.ma.masked_where(da.isnan(var.data), var.data)

        cube = ncdata.iris_xarray.cubes_from_xarray(
            tgt_ds,
            iris_load_kwargs={'constraints': src_cube.standard_name},
        )[0]
        return cube


class xESMF:  # noqa
    """xESMF regridding scheme.

    This is a wrapper around :class:`xesmf.Regrid` so it can be used in
    :meth:`iris.cube.Cube.regrid`. Ut uses the :mod:`ncdata` package to
    convert the :class:`iris.cube.Cube` to an :class:`xarray.Dataset` before
    regridding and back after regridding.

    Supports lazy regridding.

    Masks are converted to :obj:`np.nan` before regridding and converted
    back to masks after regridding.

    Parameters
    ----------
    **kwargs:
        Any keyword argument to :class:`xesmf.Regrid` or
        :meth:`xesmf.Regrid.__call__` can be provided. By default,
        the arguments ``ignore_degenerate=True``, ``keep_attrs=True``,
        ``skipna=True``, an ``unmapped_to_nan=True`` will be used.

    Attributes
    ----------
    kwargs:
        Keyword arguments that will be provided to the regridder.
    """

    def __init__(self, **kwargs) -> None:
        args = {
            'ignore_degenerate': True,
            'skipna': True,
            'keep_attrs': True,
            'unmapped_to_nan': True,
        }
        args.update(kwargs)
        self.kwargs = args

    def __repr__(self) -> str:
        """Return string representation of class."""
        return f'{self.__class__.__name__}(**{self.kwargs})'

    def regridder(
        self,
        src_cube: iris.cube.Cube,
        tgt_cube: iris.cube.Cube,
    ) -> xESMFRegridder:
        """Create xESMF regridding function.

        Parameters
        ----------
        src_cube:
            Cube defining the source grid.
        tgt_cube:
            Cube defining the target grid.

        Returns
        -------
        xESMFRegridder
            xESMF regridding function.
        """
        return xESMFRegridder(src_cube, tgt_cube, **self.kwargs)
