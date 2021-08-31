"""Derivation of variable `toz`."""

import warnings

import cf_units
import iris
from scipy import constants

from ._baseclass import DerivedVariableBase
from ._shared import pressure_level_widths

# Constants
AVOGADRO_CONST = constants.value('Avogadro constant')
AVOGADRO_CONST_UNIT = constants.unit('Avogadro constant')
STANDARD_GRAVITY = constants.value('standard acceleration of gravity')
STANDARD_GRAVITY_UNIT = constants.unit('standard acceleration of gravity')
MW_AIR = 29
MW_AIR_UNIT = cf_units.Unit('g mol^-1')
MW_O3 = 48
MW_O3_UNIT = cf_units.Unit('g mol^-1')
DOBSON_UNIT = cf_units.Unit('2.69e20 m^-2')


def ensure_correct_lon(o3_cube, ps_cube=None):
    """Ensure that ``o3`` cube contains ``longitude`` and adapt ``ps`` cube."""
    if o3_cube.coords('longitude'):
        return (o3_cube, ps_cube)

    # Get zonal mean ps if necessary
    if ps_cube is not None:
        ps_cube = ps_cube.collapsed('longitude', iris.analysis.MEAN)
        ps_cube.remove_coord('longitude')

    # Add longitude dimension to o3 (and ps if necessary) with length 1
    cubes = (o3_cube, ps_cube)
    new_cubes = []
    lon_coord = iris.coords.DimCoord([180.0], bounds=[[0.0, 360.0]],
                                     var_name='lon',
                                     standard_name='longitude',
                                     long_name='longitude',
                                     units='degrees_east')
    for cube in cubes:
        if cube is None:
            new_cubes.append(None)
            continue
        new_dim_coords = [(c, cube.coord_dims(c)) for c in cube.dim_coords]
        new_dim_coords.append((lon_coord, cube.ndim))
        new_aux_coords = [(c, cube.coord_dims(c)) for c in cube.aux_coords]
        new_cube = iris.cube.Cube(cube.core_data()[..., None],
                                  dim_coords_and_dims=new_dim_coords,
                                  aux_coords_and_dims=new_aux_coords)
        new_cube.metadata = cube.metadata
        new_cubes.append(new_cube)

    return tuple(new_cubes)


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `toz`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        if project == 'CMIP6':
            required = [{'short_name': 'o3'}, {'short_name': 'ps'}]
        else:
            required = [{'short_name': 'tro3'}, {'short_name': 'ps'}]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute total column ozone.

        Note
        ----
        The surface pressure is used as a lower integration bound. A fixed
        upper integration bound of 0 Pa is used.

        """
        o3_cube = cubes.extract_cube(
            iris.Constraint(name='mole_fraction_of_ozone_in_air'))
        ps_cube = cubes.extract_cube(
            iris.Constraint(name='surface_air_pressure'))

        # To support zonal mean o3 (e.g., from Table AERmonZ), add longitude
        # coordinate if necessary and ensure that ps has correct shape
        (o3_cube, ps_cube) = ensure_correct_lon(o3_cube, ps_cube=ps_cube)

        # Actual derivation of toz using o3 mole fraction and pressure level
        # widths
        p_layer_widths = pressure_level_widths(o3_cube,
                                               ps_cube,
                                               top_limit=0.0)
        toz_cube = (o3_cube * p_layer_widths / STANDARD_GRAVITY * MW_O3 /
                    MW_AIR)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning,
                message='Collapsing a non-contiguous coordinate')
            toz_cube = toz_cube.collapsed('air_pressure', iris.analysis.SUM)
        toz_cube.units = (o3_cube.units * p_layer_widths.units /
                          STANDARD_GRAVITY_UNIT * MW_O3_UNIT / MW_AIR_UNIT)

        # Convert from kg m^-2 to Dobson unit (2.69e20 m^-2 )
        toz_cube = toz_cube / MW_O3 * AVOGADRO_CONST
        toz_cube.units = toz_cube.units / MW_O3_UNIT * AVOGADRO_CONST_UNIT
        toz_cube.convert_units(DOBSON_UNIT)
        toz_cube.units = 'DU'

        return toz_cube
