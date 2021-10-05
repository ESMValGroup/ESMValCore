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
        metadata = cube.metadata
        cube *= 100
        cube.metadata = metadata
        return cube


class Pr(Fix):
    """Fixes for pr."""

    def fix_metadata(self, cube):
        """Fix metadata.

        The values of v20200417 are off by a factor 1000.
        """
        cube.data = cube.core_data() * 1000.
        return cube
