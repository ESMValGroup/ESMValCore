"""Derivation of variable ``co2s``."""
import dask.array as da
import iris
import numpy as np
import stratify

from ._baseclass import DerivedVariableBase


def _get_first_unmasked_data(array, axis):
    """Get first unmasked value of an array along an axis."""
    mask = da.ma.getmaskarray(array)
    numerical_mask = da.where(mask, -1.0, 1.0)
    indices_first_positive = da.argmax(numerical_mask, axis=axis)
    indices = da.meshgrid(
        *[da.arange(array.shape[i]) for i in range(array.ndim) if i != axis],
        indexing='ij')
    indices.insert(axis, indices_first_positive)
    first_unmasked_data = np.array(array)[tuple(indices)]
    return first_unmasked_data


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable ``co2s``.

    Use linear interpolation/extrapolation and surface air pressure to
    calculate CO2 mole fraction at surface.

    Note
    ----
    In some cases, ``co2`` data is masked. In these cases, the masked values
    correspond to values where the pressure level is higher than the surface
    air pressure (e.g. the 1000 hPa level for grid cells with high elevation).
    To obtain an unmasked ``co2s`` field, it is necessary to fill these masked
    values accordingly, i.e. with the lowest unmasked value for each grid cell.

    """

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [{'short_name': 'co2'}, {'short_name': 'ps'}]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute mole fraction of CO2 at surface."""
        co2_cube = cubes.extract_cube(
            iris.Constraint(name='mole_fraction_of_carbon_dioxide_in_air'))
        ps_cube = cubes.extract_cube(
            iris.Constraint(name='surface_air_pressure'))

        # Fill masked data if necessary (interpolation fails with masked data)
        (z_axis,) = co2_cube.coord_dims(co2_cube.coord(axis='Z',
                                                       dim_coords=True))
        mask = da.ma.getmaskarray(co2_cube.core_data())
        if mask.any():
            first_unmasked_data = _get_first_unmasked_data(
                co2_cube.core_data(), axis=z_axis)
            dim_map = [dim for dim in range(co2_cube.ndim) if dim != z_axis]
            first_unmasked_data = iris.util.broadcast_to_shape(
                first_unmasked_data, co2_cube.shape, dim_map)
            co2_cube.data = da.where(mask, first_unmasked_data,
                                     co2_cube.core_data())

        # Interpolation (not supported for dask arrays)
        air_pressure_coord = co2_cube.coord('air_pressure')
        original_levels = iris.util.broadcast_to_shape(
            air_pressure_coord.points, co2_cube.shape,
            co2_cube.coord_dims(air_pressure_coord))
        target_levels = np.expand_dims(ps_cube.data, axis=z_axis)
        co2s_data = stratify.interpolate(
            target_levels,
            original_levels,
            co2_cube.data,
            axis=z_axis,
            interpolation='linear',
            extrapolation='linear',
        )
        co2s_data = np.squeeze(co2s_data, axis=z_axis)

        # Construct co2s cube
        indices = [slice(None)] * co2_cube.ndim
        indices[z_axis] = 0
        co2s_cube = co2_cube[tuple(indices)]
        co2s_cube.data = co2s_data
        if co2s_cube.coords('air_pressure'):
            co2s_cube.remove_coord('air_pressure')
        ps_coord = iris.coords.AuxCoord(ps_cube.data,
                                        var_name='plev',
                                        standard_name='air_pressure',
                                        long_name='pressure',
                                        units=ps_cube.units)
        co2s_cube.add_aux_coord(ps_coord, np.arange(co2s_cube.ndim))
        co2s_cube.convert_units('1e-6')
        return co2s_cube
