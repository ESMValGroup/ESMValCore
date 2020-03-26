"""Derivation of variable ``co2s``."""
import dask.array as da
import iris

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable ``co2s``."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [{'short_name': 'co2'}]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute mole fraction of CO2 at surface."""
        cube = cubes.extract_strict(
            iris.Constraint(name='mole_fraction_of_carbon_dioxide_in_air'))
        mask = da.ma.getmaskarray(cube.core_data())
        if not mask.any():
            cube = cube[:, 0, :, :]
        else:
            numerical_mask = da.where(mask, -1.0, 1.0)
            indices_first_positive = da.argmax(numerical_mask, axis=1)
            indices = da.meshgrid(
                da.arange(cube.shape[0]),
                da.arange(cube.shape[2]),
                da.arange(cube.shape[3]),
                indexing='ij',
            )
            indices.insert(1, indices_first_positive)
            surface_data = cube.data[tuple(indices)]
            cube = cube[:, 0, :, :]
            cube.data = surface_data
        cube.convert_units('1e-6')
        return cube
