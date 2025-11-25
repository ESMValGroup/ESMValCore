"""Derivation of variable ``toz``."""

import cf_units
import iris
from scipy import constants

from esmvalcore.cmor.table import CMOR_TABLES
from esmvalcore.iris_helpers import ignore_iris_vague_metadata_warnings
from esmvalcore.preprocessor._regrid import extract_levels, regrid

from ._baseclass import DerivedVariableBase
from ._shared import pressure_level_widths

# Constants
AVOGADRO_CONST = constants.value("Avogadro constant")
AVOGADRO_CONST_UNIT = constants.unit("Avogadro constant")
STANDARD_GRAVITY = constants.value("standard acceleration of gravity")
STANDARD_GRAVITY_UNIT = constants.unit("standard acceleration of gravity")
MW_AIR = 29
MW_AIR_UNIT = cf_units.Unit("g mol^-1")
MW_O3 = 48
MW_O3_UNIT = cf_units.Unit("g mol^-1")
DOBSON_UNIT = cf_units.Unit("2.69e20 m^-2")


def add_longitude_coord(cube):
    """Add dimensional ``longitude`` coordinate of length 1 to cube."""
    lon_coord = iris.coords.DimCoord(
        [180.0],
        bounds=[[0.0, 360.0]],
        var_name="lon",
        standard_name="longitude",
        long_name="longitude",
        units="degrees_east",
    )
    new_dim_coords = [(c, cube.coord_dims(c)) for c in cube.dim_coords]
    new_dim_coords.append((lon_coord, cube.ndim))
    new_aux_coords = [(c, cube.coord_dims(c)) for c in cube.aux_coords]
    new_cube = iris.cube.Cube(
        cube.core_data()[..., None],
        dim_coords_and_dims=new_dim_coords,
        aux_coords_and_dims=new_aux_coords,
    )
    new_cube.metadata = cube.metadata
    return new_cube


def interpolate_hybrid_plevs(cube):
    """Interpolate hybrid pressure levels."""
    # Use CMIP6's plev19 target levels (in Pa)
    target_levels = CMOR_TABLES["CMIP6"].coords["plev19"].requested
    cube.coord("air_pressure").convert_units("Pa")
    return extract_levels(
        cube,
        target_levels,
        "linear",
        coordinate="air_pressure",
    )


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable ``toz``."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        # TODO: make get_required _derive/__init__.py use variables as argument
        # and make this dependent on mip
        if project == "CMIP6":
            required = [
                {"short_name": "o3"},
                {"short_name": "ps", "mip": "Amon"},
            ]
        else:
            required = [
                {"short_name": "tro3"},
                {"short_name": "ps"},
            ]
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
            iris.Constraint(name="mole_fraction_of_ozone_in_air"),
        )
        ps_cube = cubes.extract_cube(
            iris.Constraint(name="surface_air_pressure"),
        )

        # If o3 is given on hybrid pressure levels (e.g., from Table AERmon),
        # interpolate it to regular pressure levels
        if len(o3_cube.coord_dims("air_pressure")) > 1:
            o3_cube = interpolate_hybrid_plevs(o3_cube)

        # To support zonal mean o3 (e.g., from Table AERmonZ), add longitude
        # coordinate and collapsed ps cube if necessary to ensure that they
        # have correct shapes
        if not o3_cube.coords("longitude"):
            o3_cube = add_longitude_coord(o3_cube)
            with ignore_iris_vague_metadata_warnings():
                ps_cube = ps_cube.collapsed("longitude", iris.analysis.MEAN)
            ps_cube.remove_coord("longitude")
            ps_cube = add_longitude_coord(ps_cube)

        # If the horizontal dimensions of ps and o3 differ, regrid ps
        # Note: regrid() checks if the regridding is really necessary before
        # running the actual interpolation
        ps_cube = regrid(ps_cube, o3_cube, "linear")

        # Actual derivation of toz using o3 mole fraction and pressure level
        # widths
        p_layer_widths = pressure_level_widths(o3_cube, ps_cube, top_limit=0.0)
        toz_cube = o3_cube * p_layer_widths / STANDARD_GRAVITY * MW_O3 / MW_AIR
        with ignore_iris_vague_metadata_warnings():
            toz_cube = toz_cube.collapsed("air_pressure", iris.analysis.SUM)
        toz_cube.units = (
            o3_cube.units
            * p_layer_widths.units
            / STANDARD_GRAVITY_UNIT
            * MW_O3_UNIT
            / MW_AIR_UNIT
        )

        # Convert from kg m^-2 to Dobson units DU (2.69e20 m^-2 ) and from
        # DU to m (1 mm = 100 DU)
        toz_cube = toz_cube / MW_O3 * AVOGADRO_CONST
        toz_cube.units = toz_cube.units / MW_O3_UNIT * AVOGADRO_CONST_UNIT
        toz_cube.convert_units(DOBSON_UNIT)
        toz_cube.data = toz_cube.core_data() * 1e-5
        toz_cube.units = "m"

        return toz_cube
