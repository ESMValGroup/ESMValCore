"""Derivation of variable `soz`."""

import dask.array as da
import iris

from ._baseclass import DerivedVariableBase
from .toz import DerivedVariable as Toz


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `soz`."""

    # O3 mole fraction threshold (in ppb) that is used for the defintion of the
    # stratosphere (stratosphere = region where O3 mole fraction is at least as
    # hight as the threshold value)
    STRATOSPHERIC_O3_THRESHOLD = 125.0

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        if project == 'CMIP6':
            required = [{'short_name': 'o3'}, {'short_name': 'ps'}]
        else:
            required = [{'short_name': 'tro3'}, {'short_name': 'ps'}]
        return required

    def calculate(self, cubes):
        """Compute stratospheric column ozone.

        Note
        ----
        In this context, the stratosphere is defined as the region in which the
        O3 mole fraction is at least as high as the given threshold
        (``STRATOSPHERIC_O3_THRESHOLD``).

        The calculation of ``soz`` consists of two steps:
        (1) Mask out O3 mole fractions smaller than threshold
        (2) Use derivation function of ``toz`` to calculate ``soz`` (using the
            masked data)

        """
        # (1) Mask O3 mole fraction using the given threshold
        o3_cube = cubes.extract_cube(
            iris.Constraint(name='mole_fraction_of_ozone_in_air'))
        o3_cube.convert_units('1e-9')
        mask = (o3_cube.lazy_data() < self.STRATOSPHERIC_O3_THRESHOLD)
        mask |= da.ma.getmaskarray(o3_cube.lazy_data())
        o3_cube.data = da.ma.masked_array(o3_cube.lazy_data(), mask=mask)

        # (2) Use derivation function of toz to calculate soz using the masked
        # cubes
        return Toz.calculate(cubes)
