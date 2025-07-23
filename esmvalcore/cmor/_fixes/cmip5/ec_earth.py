"""Fixes for EC-Earth model."""

from collections.abc import Iterable

import iris
import numpy as np
from dask import array as da

from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor._fixes.shared import (
    add_scalar_height_coord,
    cube_to_aux_coord,
)


class Sic(Fix):
    """Fixes for sic."""

    def fix_data(self, cube):
        """Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube
            Cube to fix.

        Returns
        -------
        iris.cube.Cube
        """
        metadata = cube.metadata
        cube *= 100
        cube.metadata = metadata
        return cube


class Sftlf(Fix):
    """Fixes for sftlf."""

    def fix_data(self, cube):
        """Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube
            Cube to fix.

        Returns
        -------
        iris.cube.Cube
        """
        metadata = cube.metadata
        cube *= 100
        cube.metadata = metadata
        return cube


class Tos(Fix):
    """Fixes for tos."""

    def fix_data(self, cube):
        """Fix tos data.

        Fixes mask

        Parameters
        ----------
        cube: iris.cube.Cube
            Cube to fix.

        Returns
        -------
        iris.cube.Cube
        """
        cube.data = da.ma.masked_equal(cube.core_data(), 273.15)
        return cube


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """Fix potentially missing scalar dimension.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList
        """
        for cube in cubes:
            if not cube.coords(var_name="height"):
                add_scalar_height_coord(cube)

            if cube.coord("time").long_name is None:
                cube.coord("time").long_name = "time"

        return cubes


class Areacello(Fix):
    """Fixes for areacello."""

    def fix_metadata(self, cubes):
        """Fix potentially missing scalar dimension.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList
        """
        areacello = cubes.extract("Areas of grid cell")[0]
        lat = cubes.extract("latitude")[0]
        lon = cubes.extract("longitude")[0]

        areacello.add_aux_coord(cube_to_aux_coord(lat), (0, 1))
        areacello.add_aux_coord(cube_to_aux_coord(lon), (0, 1))

        return iris.cube.CubeList(
            [
                areacello,
            ],
        )


class Pr(Fix):
    """Fixes for pr."""

    def fix_metadata(
        self,
        cubes: Iterable[iris.cube.Cube],
    ) -> iris.cube.CubeList:
        """Fix time coordinate.

        Last file (2000-2009) has erroneously duplicated points
        in time coordinate (e.g. [t1, t2, t3, t4, t2, t3, t4, t5])
        which should be removed except the last one which is correct.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Cubes to fix.

        Returns
        -------
        iris.cube.CubeList
        """
        new_list = iris.cube.CubeList()
        for cube in cubes:
            try:
                time_coord = cube.coord("time")
            except iris.exceptions.CoordinateNotFoundError:
                new_list.append(cube)
            else:
                if time_coord.is_monotonic():
                    new_list.append(cube)
                else:
                    # erase erroneously copy-pasted points
                    select = np.unique(time_coord.points, return_index=True)[1]
                    new_cube = cube[select]
                    iris.util.promote_aux_coord_to_dim_coord(new_cube, "time")
                    new_list.append(new_cube)

        return new_list
