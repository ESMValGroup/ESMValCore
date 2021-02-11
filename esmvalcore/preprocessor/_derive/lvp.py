"""Derivation of variable `lvp`.

authors:
    - weig_ka

"""

from esmvalcore.iris_helpers import var_name_constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `lvp`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'hfls'
            },
            {
                'short_name': 'pr'
            },
            {
                'short_name': 'evspsbl'
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute Latent Heat Release from Precipitation."""
        hfls_cube = cubes.extract_cube(var_name_constraint('hfls'))
        pr_cube = cubes.extract_cube(var_name_constraint('pr'))
        evspsbl_cube = cubes.extract_cube(var_name_constraint('evspsbl'))

        lvp_cube = hfls_cube * (pr_cube / evspsbl_cube)

        return lvp_cube
