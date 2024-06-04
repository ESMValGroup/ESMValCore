"""Regridding schemes."""
from __future__ import annotations

import logging
from collections.abc import Callable

from iris.cube import Cube

from esmvalcore.preprocessor._regrid_esmpy import (
    ESMPyAreaWeighted,
    ESMPyLinear,
    ESMPyNearest,
    ESMPyRegridder,
)
from esmvalcore.preprocessor._regrid_unstructured import (
    UnstructuredLinear,
    UnstructuredLinearRegridder,
    UnstructuredNearest,
)

logger = logging.getLogger(__name__)


__all__ = [
    'ESMPyAreaWeighted',
    'ESMPyLinear',
    'ESMPyNearest',
    'ESMPyRegridder',
    'GenericFuncScheme',
    'GenericRegridder',
    'UnstructuredLinear',
    'UnstructuredLinearRegridder',
    'UnstructuredNearest',
]


class GenericRegridder:
    r"""Generic function regridder.

    Does support lazy regridding if `func` does. Does not support weights
    caching.

    Parameters
    ----------
    src_cube:
        Cube defining the source grid.
    tgt_cube:
        Cube defining the target grid.
    func:
        Generic regridding function with signature f(src_cube: Cube, grid_cube:
        Cube, \*\*kwargs) -> Cube.
    **kwargs:
        Keyword arguments for the generic regridding function.

    """

    def __init__(
        self,
        src_cube: Cube,
        tgt_cube: Cube,
        func: Callable,
        **kwargs,
    ):
        """Initialize class instance."""
        self.src_cube = src_cube
        self.tgt_cube = tgt_cube
        self.func = func
        self.kwargs = kwargs

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
        return self.func(cube, self.tgt_cube, **self.kwargs)


class GenericFuncScheme:
    r"""Regridding with a generic function.

    This class can be used in :meth:`iris.cube.Cube.regrid`.

    Does support lazy regridding if `func` does.

    Parameters
    ----------
    func:
        Generic regridding function with signature f(src_cube: Cube, grid_cube:
        Cube, \*\*kwargs) -> Cube.
    **kwargs:
        Keyword arguments for the generic regridding function.

    """

    def __init__(self, func: Callable, **kwargs):
        """Initialize class instance."""
        self.func = func
        self.kwargs = kwargs

    def __repr__(self) -> str:
        """Return string representation of class."""
        kwargs = ', '.join(f"{k}={v}" for (k, v) in self.kwargs.items())
        return f'GenericFuncScheme({self.func.__name__}, {kwargs})'

    def regridder(self, src_cube: Cube, tgt_cube: Cube) -> GenericRegridder:
        """Get regridder.

        Parameters
        ----------
        src_cube:
            Cube defining the source grid.
        tgt_cube:
            Cube defining the target grid.

        Returns
        -------
        GenericRegridder
            Regridder instance.

        """
        return GenericRegridder(src_cube, tgt_cube, self.func, **self.kwargs)
