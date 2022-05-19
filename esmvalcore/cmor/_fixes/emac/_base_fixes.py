"""Fix base classes for EMAC on-the-fly CMORizer."""

import logging

import iris.analysis
from iris import NameConstraint
from iris.cube import CubeList

from ..fix import Fix

logger = logging.getLogger(__name__)


class EmacFix(Fix):
    """Base class for all EMAC fixes."""

    def get_cube(self, cubes, var_name=None):
        """Extract single cube."""
        if var_name is None:
            var_name = self.extra_facets.get('raw_name',
                                             self.vardef.short_name)
        if not cubes.extract(NameConstraint(var_name=var_name)):
            raise ValueError(
                f"Variable '{var_name}' used to extract "
                f"'{self.vardef.short_name}' is not available in input "
                f"file")
        return cubes.extract_cube(NameConstraint(var_name=var_name))


class NegateData(EmacFix):
    """Base fix to negate data."""

    def fix_data(self, cube):
        """Fix data."""
        cube.data = -cube.core_data()
        return cube


class SetUnitsTo1(EmacFix):
    """Base fix to set units to '1'."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes)
        cube.units = '1'
        return cubes


class SetUnitsTo1SumOverZ(EmacFix):
    """Base fix to set units to '1' and sum over Z-coordinate."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes)
        cube.units = '1'
        cube = self.sum_over_z_coord(cube)
        return cubes

    @staticmethod
    def sum_over_z_coord(cube):
        """Perform sum over Z-coordinate."""
        z_coord = cube.coord(axis='Z')
        cube = cube.collapsed(z_coord, iris.analysis.SUM)
        return cube
