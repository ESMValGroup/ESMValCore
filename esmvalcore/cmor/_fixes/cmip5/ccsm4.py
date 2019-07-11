"""Fixes for CCSM4 model."""

import numpy as np

from ..fix import Fix


class Rlut(Fix):
    """Fixes for rlut."""

    def fix_metadata(self, cubes):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        lat = cube.coord('latitude')
        lat.points = np.round(lat.points, 3)
        lat.bounds = np.round(lat.bounds, 3)
        return cubes


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
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

        """
        self.get_cube_from_list(cubes).units = '1e3'
        return cubes
