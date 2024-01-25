"""Unstructured grid regridding."""
from __future__ import annotations

import logging

from iris.analysis import UnstructuredNearest as IrisUnstructuredNearest
from iris.analysis.trajectory import UnstructuredNearestNeigbourRegridder
from iris.cube import Cube

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
