"""Derivation of variable ``phcint_total``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris
from cf_units import Unit
from iris import NameConstraint

from esmvalcore.preprocessor._volume import depth_integration

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from iris.cube import Cube, CubeList

    from esmvalcore.typing import Facets

RHO_CP = iris.coords.AuxCoord(4.09169e6, units=Unit("kg m-3 J kg-1 K-1"))


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `ohc`."""

    @staticmethod
    def required(project: str) -> list[Facets]:  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [{"short_name": "thetao"}]

    @staticmethod
    def calculate(cubes: CubeList) -> Cube:
        """Compute total column vertically-integrated heat content.

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
        cube = cube * RHO_CP
        return depth_integration(cube)
