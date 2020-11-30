"""Derivation of variable ``dco2mass``."""
import numpy as np
from scipy.interpolate import interp1d

from esmvalcore.iris_helpers import var_name_constraint

from ._baseclass import DerivedVariableBase
from ._shared import get_absolute_time_units


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable ``dco2mass``.

    First, use cubic spline interpolation (and extrapolation at start and end
    time) to calculate co2mass at the time bounds (from the values at the time
    points). Second, calculate co2mass change for given time point by
    subtracting the values at the corresponding time bounds. Finally, divide by
    length of time interval to get rate of change.

    """

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [{'short_name': 'co2mass'}]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute mole fraction of CO2 at surface."""
        cube = cubes.extract_strict(var_name_constraint('co2mass'))

        # Get interpolating function time -> co2mass
        time_coord = cube.coord('time')
        f_co2mass = interp1d(time_coord.points, cube.data, kind='cubic',
                             fill_value='extrapolate')

        # Interpolate to values at time bounds
        bounds = np.unique(time_coord.bounds.reshape(-1))
        co2mass_at_bounds = f_co2mass(bounds)

        # Get change in co2mass at time point
        dco2mass = np.diff(co2mass_at_bounds)

        # Get rate of change and setup new cube
        time_sizes = np.diff(bounds)
        dco2mass = dco2mass / time_sizes
        cube.data = dco2mass

        # Adapt metadata
        time_units = get_absolute_time_units(time_coord.units)
        cube.units /= time_units
        cube.convert_units('kg s-1')
        return cube
