"""Regridding schemes."""
from __future__ import annotations

import logging
from collections.abc import Callable

from iris.cube import Cube

from esmvalcore.preprocessor._regrid_esmpy import (
    ESMPyAreaWeighted,
    ESMPyLinear,
    ESMPyNearest,
)
from esmvalcore.preprocessor._regrid_unstructured import UnstructuredNearest

logger = logging.getLogger(__name__)


__all__ = [
    'ESMPyAreaWeighted',
    'ESMPyLinear',
    'ESMPyNearest',
    'GenericFuncScheme',
    'UnstructuredNearest',
]


class _GenericRegridder:
    """Generic function regridder."""

    def __init__(
        self,
        src_cube: Cube,
        tgt_cube: Cube,
        func: Callable,
        **kwargs,
    ):
        """Generic function regridder.

        This class can be used in :meth:`iris.cube.Cube.regrid`.

        Parameters
        ----------
        func:
            Generic regridding function with signature f(src_cube: Cube,
            grid_cube: Cube, **kwargs) -> Cube.
        **kwargs:
            Keyword arguments for the generic regridding function.

        """
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
    """Regridding with a generic function."""

    def __init__(self, func: Callable, **kwargs):
        """Regridding with a generic function.

        This class can be used in :meth:`iris.cube.Cube.regrid`.

        Parameters
        ----------
        func:
            Generic regridding function with signature f(src_cube: Cube,
            grid_cube: Cube, **kwargs) -> Cube.
        **kwargs:
            Keyword arguments for the generic regridding function.

        """
        self.func = func
        self.kwargs = kwargs

    def __repr__(self) -> str:
        """String representation of class."""
        kwargs = ', '.join(f"{k}={v}" for (k, v) in self.kwargs.items())
        return f'GenericFuncScheme({self.func.__name__}, {kwargs})'

    def regridder(self, src_cube: Cube, tgt_cube: Cube) -> _GenericRegridder:
        """Get regridder.

        Parameters
        ----------
        src_cube:
            Cube defining the source grid.
        tgt_cube:
            Cube defining the target grid.

        Returns
        -------
        _GenericRegridder
            Regridder instance.

        """
        return _GenericRegridder(src_cube, tgt_cube, self.func, **self.kwargs)
