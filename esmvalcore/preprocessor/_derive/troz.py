"""Derivation of variable ``troz``."""

import dask.array as da
import iris

from esmvalcore.iris_helpers import ignore_iris_vague_metadata_warnings

from ._baseclass import DerivedVariableBase
from .soz import STRATOSPHERIC_O3_THRESHOLD
from .toz import DerivedVariable as Toz
from .toz import add_longitude_coord, interpolate_hybrid_plevs


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable ``troz``."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        return Toz.required(project)

    @staticmethod
    def calculate(cubes):
        """Compute tropospheric column ozone.

        Note
        ----
        Here, the troposphere is defined as the region in which the O3 mole
        fraction is smaller than the given threshold
        (``STRATOSPHERIC_O3_THRESHOLD``).

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

        # Mask O3 mole fraction using the given threshold
        o3_cube.convert_units("1e-9")
        mask = o3_cube.lazy_data() >= STRATOSPHERIC_O3_THRESHOLD
        mask |= da.ma.getmaskarray(o3_cube.lazy_data())
        o3_cube.data = da.ma.masked_array(o3_cube.lazy_data(), mask=mask)

        # Use derivation function of toz to calculate troz using the masked o3
        # cube and the ps cube
        cubes = iris.cube.CubeList([o3_cube, ps_cube])
        return Toz.calculate(cubes)
