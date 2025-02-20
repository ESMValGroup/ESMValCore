"""Fixes for MRI-ESM1 model."""

from typing import Sequence

import iris.coords
import iris.cube
import iris.exceptions
import numpy as np
from dask import array as da

from esmvalcore.cmor._fixes.fix import Fix


class AllVars(Fix):
    def fix_metadata(
        self, cubes: Sequence[iris.cube.Cube]
    ) -> Sequence[iris.cube.Cube]:
        """Replace rlat and rlon by index coordinates."""
        # In file fgco2_Omon_MRI-ESM1_historical_r1i1p1_185101-200512.nc
        # (v20130307) the usual horizontal index coordinates appear to have
        # been replaced by rotated pole coordinates.
        #
        # This leads to iris-esmf-regrid selecting the wrong coordinate when
        # regridding.
        for cube in cubes:
            for coord_name, index_var_name, index_long_name in [
                ("latitude", "i", "cell index along first dimension"),
                ("longitude", "j", "cell index along second dimension"),
            ]:
                try:
                    rotated_coord = cube.coord(f"grid_{coord_name}")
                    horizontal_coord = cube.coord(coord_name)
                except iris.exceptions.CoordinateNotFoundError:
                    pass
                else:
                    if len(horizontal_coord.shape) == 2:
                        (dim,) = cube.coord_dims(rotated_coord)
                        (size,) = rotated_coord.shape
                        cube.remove_coord(rotated_coord)
                        index_coord = iris.coords.DimCoord(
                            points=np.arange(1, size + 1),
                            var_name=index_var_name,
                            long_name=index_long_name,
                            units="1",
                        )
                        cube.add_dim_coord(index_coord, dim)
        return cubes


class Msftmyz(Fix):
    """Fixes for msftmyz."""

    def fix_data(self, cube):
        """
        Fix msftmyz data.

        Fixes mask

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        cube.data = da.ma.masked_equal(cube.core_data(), 0.0)
        return cube
