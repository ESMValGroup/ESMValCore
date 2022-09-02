"""Fix base classes for EMAC on-the-fly CMORizer."""

import logging

from iris import NameConstraint
from iris.exceptions import ConstraintMismatchError

from ..native_datasets import NativeDatasetFix

logger = logging.getLogger(__name__)


class EmacFix(NativeDatasetFix):
    """Base class for all EMAC fixes."""

    def get_cube(self, cubes, var_name=None):
        """Extract single cube."""
        # If no var_name given, use the CMOR short_name
        if var_name is None:
            var_name = self.extra_facets.get('raw_name',
                                             self.vardef.short_name)

        # Convert to list if only a single var_name is given
        if isinstance(var_name, str):
            var_names = [var_name]
        else:
            var_names = var_name

        # Try to extract the variable (prioritize variables as given by the
        # list)
        for v_name in var_names:
            try:
                return cubes.extract_cube(NameConstraint(var_name=v_name))
            except ConstraintMismatchError:
                pass

        # If no cube could be extracted, raise an error
        raise ValueError(
            f"No variable of {var_names} necessary for the extraction/"
            f"derivation the CMOR variable '{self.vardef.short_name}' is "
            f"available in the input file. Hint: in case you tried to extract "
            f"a 3D variable defined on pressure levels, it might be necessary "
            f"to define the EMAC variable name in the recipe (e.g., "
            f"'raw_name: tm1_p39_cav') if the default number of pressure "
            f"levels is not available in the input file."
        )


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
