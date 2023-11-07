"""Fixes for MIROC5 model."""
from dask import array as da

from ..common import ClFixHybridPressureCoord
from ..fix import Fix
from ..shared import round_coordinates


Cl = ClFixHybridPressureCoord


class Sftof(Fix):
    """Fixes for sftof."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 100
        cube.metadata = metadata
        return cube


class Snw(Fix):
    """Fixes for snw."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 100
        cube.metadata = metadata
        return cube


class Snc(Snw):
    """Fixes for snc."""

    # dayspermonth = (/31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31/)

    # if ((name.eq."snc".or.name.eq."snw").and.FIELD.eq."T2Ds".and. \
    #     ENSEMBLE.eq."r1i1p1") then
    #     opt = 0
    #     opt@calendar = var&time@calendar
    #     t = 0.0
    #     t@calendar = var&time@calendar
    #     t@units = var&time@units
    #     res = cd_calendar(t, -5)
    #     yy = res(0, 0)
    #     mm = res(0, 1)
    #     dd = res(0, 2)
    #     do ii = 0, dimsizes(var&time) - 1
    #         var&time(ii) = tofloat(cd_inv_calendar(yy, mm, dd, 12, 0, 0, \
    #                                var&time@units, opt))
    #         dd = dd + 1
    #         if (dd.gt.dayspermonth(mm-1)) then
    #             mm = mm + 1
    #             dd = 1
    #         end if
    #         if (mm.gt.12) then
    #             mm = 1
    #             yy = yy + 1
    #         end if
    #     end do
    #     ret = 0
    # end if


class Msftmyz(Fix):
    """Fixes for msftmyz."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes mask

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        cube.data = da.ma.masked_equal(cube.core_data(), 0.)
        return cube


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Some coordinate points vary for different files of this dataset (for
        different time range). This fix removes these inaccuracies by rounding
        the coordinates.

        Parameters
        ----------
        cubes: iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        return round_coordinates(cubes)


class Hur(Tas):
    """Fixes for hur."""


class Tos(Fix):
    """Fixes for tos."""

    def fix_data(self, cube):
        """
        Fix tos data.

        Fixes mask

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        cube.data = da.ma.masked_equal(cube.core_data(), 0.)
        return cube


class Evspsbl(Tas):
    """Fixes for evspsbl."""


class Hfls(Tas):
    """Fixes for hfls."""


class Pr(Tas):
    """Fixes for pr."""
