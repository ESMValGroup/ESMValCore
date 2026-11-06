"""Derivation of variable ``phcint``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cf_units import Unit
from iris import NameConstraint

from esmvalcore.preprocessor._shared import get_coord_weights

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from iris.cube import Cube, CubeList

    from esmvalcore.typing import Facets

RHO_CP = 4.09169e6
RHO_CP_UNIT = Unit("kg m-3 J kg-1 K-1")


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `ohc`."""

    @staticmethod
    def required(project: str) -> list[Facets]:  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [{"short_name": "thetao"}]

    @staticmethod
    def calculate(cubes: CubeList) -> Cube:
        """Compute vertically-integrated heat content.

        Use c_p * rho_0 = 4.09169e+6 J m-3 K-1 (Kuhlbrodt et al., 2015, Clim.
        Dyn.)

        Arguments
        ---------
        cubes:
           Input cubes.

        Returns
        -------
        :
            Output cube.

        """
        cube = cubes.extract_cube(NameConstraint(var_name="thetao"))
        cube.convert_units("K")

        # In the following, we modify the cube's data and units instead of the
        # cube directly to avoid dropping cell measures and ancillary variables
        # (https://scitools-iris.readthedocs.io/en/stable/further_topics/lenient_maths.html#finer-detail)

        # Multiply by c_p * rho_0 -> J m-3
        cube.data = cube.core_data() * RHO_CP
        cube.units *= RHO_CP_UNIT

        # Multiply by layer depth -> J m-2
        z_coord = cube.coord(axis="z")
        layer_depth = get_coord_weights(cube, z_coord, broadcast=True)
        cube.data = cube.core_data() * layer_depth
        cube.units *= z_coord.units

        return cube
