"""Derivation of variable ``soz``."""

import dask.array as da
import iris

from ._baseclass import DerivedVariableBase
from .toz import DerivedVariable as Toz
from .toz import add_longitude_coord, interpolate_hybrid_plevs

# O3 mole fraction threshold (in ppb) that is used for the definition of the
# stratosphere (stratosphere = region where O3 mole fraction is at least as
# high as the threshold value)
STRATOSPHERIC_O3_THRESHOLD = 125.0


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable ``soz``."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        if project == "CMIP6":
            required = [{"short_name": "o3"}]
        else:
            required = [{"short_name": "tro3"}]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute stratospheric column ozone.

        Note
        ----
        Here, the stratosphere is defined as the region in which the O3 mole
        fraction is at least as high as the given threshold
        (``STRATOSPHERIC_O3_THRESHOLD``).

        In the calculation of ``toz``, the surface air pressure (``ps``) is
        used to determine the pressure level width of the lowest layer. For
        ``soz``, this lowest layer can be ignored since it is not located in
        the stratosphere (it will be masked out due to the O3 mole fraction
        threshold). Thus, the surface air pressure (``ps``) is not necessary
        for the derivation of ``soz`` and is simply replaced with the lowest
        pressure level in the data to be able to use the ``toz`` derivation
        function.

        The calculation of ``soz`` consists of three steps:
        (1) Mask out O3 mole fractions smaller than given threshold.
        (2) Cut out the lowest pressure level from the data and use it as
            surface air pressure (``toz``).
        (3) Use derivation function of ``toz`` to calculate ``soz`` (using the
            masked data).

        """
        o3_cube = cubes.extract_cube(
            iris.Constraint(name="mole_fraction_of_ozone_in_air"),
        )

        # If o3 is given on hybrid pressure levels (e.g., from Table AERmon),
        # interpolate it to regular pressure levels
        if len(o3_cube.coord_dims("air_pressure")) > 1:
            o3_cube = interpolate_hybrid_plevs(o3_cube)

        # To support zonal mean o3 (e.g., from Table AERmonZ), add longitude
        # coordinate if necessary
        if not o3_cube.coords("longitude"):
            o3_cube = add_longitude_coord(o3_cube)

        # (1) Mask O3 mole fraction using the given threshold
        o3_cube.convert_units("1e-9")
        mask = o3_cube.lazy_data() < STRATOSPHERIC_O3_THRESHOLD
        mask |= da.ma.getmaskarray(o3_cube.lazy_data())
        o3_cube.data = da.ma.masked_array(o3_cube.lazy_data(), mask=mask)

        # (2) Add surrogate for the surface air pressure (ps) cube using the
        # lowest pressure level available in the data (this is fine since the
        # the lowest pressure level is far away from the stratosphere).

        # Get dummy ps cube with correct dimensions
        ps_dims = (
            o3_cube.coord_dims("time")
            + o3_cube.coord_dims("latitude")
            + o3_cube.coord_dims("longitude")
        )
        idx_to_extract_ps = [0] * o3_cube.ndim
        for ps_dim in ps_dims:
            idx_to_extract_ps[ps_dim] = slice(None)
        ps_cube = o3_cube[tuple(idx_to_extract_ps)].copy()

        # Set ps data using lowest pressure level available and add correct
        # metadata
        lowest_plev = o3_cube.coord("air_pressure").points.max()
        ps_data = da.broadcast_to(lowest_plev, ps_cube.shape)
        ps_cube.data = ps_data
        ps_cube.var_name = "ps"
        ps_cube.standard_name = "surface_air_pressure"
        ps_cube.long_name = "Surface Air Pressure"
        ps_cube.units = o3_cube.coord("air_pressure").units

        # Cut lowest pressure level from o3_cube
        z_dim = o3_cube.coord_dims("air_pressure")[0]
        idx_to_cut_lowest_plev = [slice(None)] * o3_cube.ndim
        idx_to_cut_lowest_plev[z_dim] = slice(1, None)
        o3_cube = o3_cube[tuple(idx_to_cut_lowest_plev)]

        # (3) Use derivation function of toz to calculate soz using the masked
        # o3 cube and the surrogate ps cube
        cubes = iris.cube.CubeList([o3_cube, ps_cube])
        return Toz.calculate(cubes)
