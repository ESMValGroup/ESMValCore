"""Fixes for CCSM4 model."""

import dask.array as da

from ..common import ClFixHybridPressureCoord
from ..fix import Fix
from ..shared import round_coordinates


Cl = ClFixHybridPressureCoord


class Csoil(Fix):
    """Fixes for Csoil."""

    def fix_data(self, cube):
        """Fix data.

        The data is not properly masked.
        This fixes the mask.

        Parameters
        ----------
        cube : iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube
        """
        cube.data = da.ma.masked_equal(cube.core_data(), 1.e33)
        return cube


Cveg = Csoil


Gpp = Csoil


class Rlut(Fix):
    """Fixes for rlut."""

    def fix_metadata(self, cubes):
        """Fix data.

        Some coordinate points vary for different files of this dataset (for
        different time range). This fix removes these inaccuracies by rounding
        the coordinates.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        return round_coordinates(cubes, 3)


class Rlutcs(Rlut):
    """Fixes for rlutcs."""


class Rsut(Rlut):
    """Fixes for rsut."""


class Rsutcs(Rlut):
    """Fixes for rsutcs."""


class Rlus(Rlut):
    """Fixes for rlus."""


class Rsus(Rlut):
    """Fixes for rsus."""


class Rsuscs(Rlut):
    """Fixes for rsuscs."""


class Rlds(Rlut):
    """Fixes for rlds."""


class Rldscs(Rlut):
    """Fixes for rldscs."""


class Rsds(Rlut):
    """Fixes for rsds."""


class Rsdscs(Rlut):
    """Fixes for rsdscs."""


class Rsdt(Rlut):
    """Fixes for rsdt."""


class So(Fix):
    """Fixes for so."""

    def fix_metadata(self, cubes):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        self.get_cube_from_list(cubes).units = '1e3'
        return cubes
