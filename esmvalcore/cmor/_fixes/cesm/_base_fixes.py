"""Fix base classes for CESM on-the-fly CMORizer."""

import logging

from iris import NameConstraint
from iris.exceptions import ConstraintMismatchError

from ..fix import Fix

logger = logging.getLogger(__name__)


class CesmFix(Fix):
    """Base class for all CESM fixes."""

    def get_cube(self, cubes, var_name=None):
        """Extract single cube."""
        # If no var_name given, use the CMOR short_name
        if var_name is None:
            var_name = self.extra_facets.get('raw_name',
                                             self.vardef.short_name)

        # Try to extract the variable
        try:
            return cubes.extract_cube(NameConstraint(var_name=var_name))
        except ConstraintMismatchError:
            raise ValueError(
                f"No variable of {var_name} necessary for the extraction/"
                f"derivation the CMOR variable '{self.vardef.short_name}' is "
                f"available in the input file"
            )
