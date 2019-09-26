# pylint: disable=invalid-name, no-self-use, too-few-public-methods
"""Fixes for UKESM1-0-LL."""
# import numpy as np
# import iris

from ..fix import Fix


class allvars(Fix):
    """Fixes for all vars."""

    def fix_metadata(self, cubelist):
        """
        Fix non-standard dimension names.

        Parameters
        ----------
        cubelist: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList
        """
        parent_units = 'parent_time_units'
        bad_value = 'days since 1850-01-01-00-00-00'
        for cube in cubelist:
            try:
                if cube.attributes[parent_units] == bad_value:
                    cube.attributes[parent_units] = 'days since 1850-01-01'
            except AttributeError:
                pass
        return cubelist
