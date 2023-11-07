"""Derivation of variable `vegFrac`."""

import dask.array as da
import iris
from iris import NameConstraint

from .._regrid import regrid
from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `vegFrac`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {'short_name': 'baresoilFrac'},
            {'short_name': 'residualFrac'},
            {'short_name': 'sftlf', 'mip': 'fx'},
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute vegetation fraction from bare soil fraction."""
        baresoilfrac_cube = cubes.extract_cube(NameConstraint(
            var_name='baresoilFrac'))
        residualfrac_cube = cubes.extract_cube(NameConstraint(
            var_name='residualFrac'))
        sftlf_cube = cubes.extract_cube(NameConstraint(var_name='sftlf'))

        # Add time dimension to sftlf
        target_shape_sftlf = (baresoilfrac_cube.shape[0], *sftlf_cube.shape)
        sftlf_data = iris.util.broadcast_to_shape(sftlf_cube.data,
                                                  target_shape_sftlf, (1, 2))
        sftlf_cube = baresoilfrac_cube.copy(sftlf_data)
        sftlf_cube.data = sftlf_cube.lazy_data()

        # Regrid sftlf if necessary and adapt mask
        if sftlf_cube.shape != baresoilfrac_cube.shape:
            sftlf_cube = regrid(sftlf_cube, baresoilfrac_cube, 'linear')
        sftlf_cube.data = da.ma.masked_array(
            sftlf_cube.core_data(),
            mask=da.ma.getmaskarray(baresoilfrac_cube.core_data()))

        # Calculate vegetation fraction
        baresoilfrac_cube.data = (sftlf_cube.core_data() -
                                  baresoilfrac_cube.core_data() -
                                  residualfrac_cube.core_data())
        return baresoilfrac_cube
