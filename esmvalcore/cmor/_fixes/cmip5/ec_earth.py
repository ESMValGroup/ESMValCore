"""Fixes for EC-Earth model."""
import iris
import numpy as np
from dask import array as da

from ..fix import Fix
from ..shared import add_scalar_height_coord, cube_to_aux_coord


class Sic(Fix):
    """Fixes for sic."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube

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
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube

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
        """
        Fix tos data.

        Fixes mask

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        cube.data = da.ma.masked_equal(cube.core_data(), 273.15)
        return cube


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Fix potentially missing scalar dimension.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """

        for cube in cubes:
            if not cube.coords(var_name='height'):
                add_scalar_height_coord(cube)

            if cube.coord('time').long_name is None:
                cube.coord('time').long_name = 'time'

        return cubes


class Areacello(Fix):
    """Fixes for areacello."""

    def fix_metadata(self, cubes):
        """
        Fix potentially missing scalar dimension.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        areacello = cubes.extract('Areas of grid cell')[0]
        lat = cubes.extract('latitude')[0]
        lon = cubes.extract('longitude')[0]

        areacello.add_aux_coord(cube_to_aux_coord(lat), (0, 1))
        areacello.add_aux_coord(cube_to_aux_coord(lon), (0, 1))

        return iris.cube.CubeList([areacello, ])


class Pr(Fix):
    """Fixes for pr."""

    def fix_metadata(self, cubes):
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
                old_time = cube.coord('time')
            except iris.exceptions.CoordinateNotFoundError:
                new_list.append(cube)
            else:
                if old_time.is_monotonic():
                    new_list.append(cube)
                else:
                    time_units = old_time.units
                    time_data = old_time.points

                    # erase erroneously copy-pasted points
                    time_diff = np.diff(time_data)
                    idx_neg = np.where(time_diff <= 0.)[0]
                    while len(idx_neg) > 0:
                        time_data = np.delete(time_data, idx_neg[0] + 1)
                        time_diff = np.diff(time_data)
                        idx_neg = np.where(time_diff <= 0.)[0]

                    # create the new time coord
                    new_time = iris.coords.DimCoord(time_data,
                                                    standard_name='time',
                                                    var_name='time',
                                                    units=time_units)

                    # create a new cube with the right shape
                    dims = (time_data.shape[0],
                            cube.coord('latitude').shape[0],
                            cube.coord('longitude').shape[0])
                    data = cube.data
                    new_data = np.ma.append(data[:dims[0] - 1, :, :],
                                            data[-1, :, :])
                    new_data = new_data.reshape(dims)

                    tmp_cube = iris.cube.Cube(
                        new_data,
                        standard_name=cube.standard_name,
                        long_name=cube.long_name,
                        var_name=cube.var_name,
                        units=cube.units,
                        attributes=cube.attributes,
                        cell_methods=cube.cell_methods,
                        dim_coords_and_dims=[(new_time, 0),
                                             (cube.coord('latitude'), 1),
                                             (cube.coord('longitude'), 2)])

                    new_list.append(tmp_cube)

        return new_list
