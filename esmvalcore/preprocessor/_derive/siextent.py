"""Create mask for derivation of variable `siextent`."""

import logging

import dask.array as da
import iris
from iris import Constraint

from ._baseclass import DerivedVariableBase

logger = logging.getLogger(__name__)


class DerivedVariable(DerivedVariableBase):
    """Create mask for derivation of variable `siextent`."""

    @staticmethod
    def required(project):  # noqa: ARG004
        """Declare the variable needed for derivation."""
        # 'sic' is sufficient as there is already an entry
        # in the mapping table esmvalcore/cmor/variable_alt_names.yml
        return [{"short_name": "sic"}]

    @staticmethod
    def calculate(cubes):
        """Compute sea ice extent.

        Returns an array of ones in every grid point where
        the sea ice area fraction has values > 15% .

        Use in combination with the preprocessor
        `area_statistics(operator='sum')` to weigh by the area and
        compute global or regional sea ice extent values (in m2).

        Arguments
        ---------
            cubes: cubelist containing sea ice area fraction.

        Returns
        -------
            Cube containing sea ice extent.
        """
        try:
            sic = cubes.extract_cube(Constraint(name="sic"))
        except iris.exceptions.ConstraintMismatchError:
            try:
                sic = cubes.extract_cube(Constraint(name="siconc"))
            except iris.exceptions.ConstraintMismatchError as exc:
                msg = (
                    "Derivation of siextent failed due to missing variables "
                    "sic and siconc."
                )
                raise ValueError(msg) from exc

        ones = da.ones_like(sic)
        siextent_data = da.ma.masked_where(sic.lazy_data() < 15.0, ones)
        siextent = sic.copy(siextent_data)
        # unit is 1 as this is just a mask that has to be used with
        # preprocessor area_statistics(operator='sum') to obtain the
        # sea ice extent (m2)
        siextent.units = "1"

        return siextent
