"""Fixes for CIESM model."""
from ..common import ClFixHybridPressureCoord
from ..fix import Fix


class Cl(ClFixHybridPressureCoord):
    """Fixes for cl."""

    def fix_data(self, cube):
        """Fix data.

        Fixes discrepancy between declared units and real units.

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube
        """
        if cube.core_data().max() <= 1.0:
            cube.units = '1'
            cube.convert_units('%')
        return cube


class Clt(Fix):
    """Fixes for clt."""

    def fix_data(self, cube):
        """Fix data.

        Fixes discrepancy between declared units and real units.

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube
        """
        if cube.core_data().max() <= 1.0:
            cube.units = '1'
            cube.convert_units('%')
        return cube


class Pr(Fix):
    """Fixes for pr."""

    def fix_data(self, cube):
        """Fix data.

        The values of v20200417 are off by a factor 1000.
        """
        if float(cube.core_data()[:10].mean()) < 1.e-5:
            cube.data = cube.core_data() * 1000.
        return cube
