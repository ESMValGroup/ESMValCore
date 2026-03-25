"""Derivation of variable ``phcint``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cf_units import Unit
from iris import NameConstraint

from esmvalcore.preprocessor._volume import _add_axis_stats_weights_coord

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

        # J m-3 (multiply by c_p * rho_0)
        cube.data = (
            cube.core_data() * RHO_CP
        )  # https://github.com/SciTools/iris/issues/6990
        cube.units *= RHO_CP_UNIT

        # J m-2 (multiply by layer depth)
        _add_axis_stats_weights_coord(cube, cube.coord(axis="z"))
        cube = cube * cube.coord("_axis_statistics_weights_")
        cube.remove_coord("_axis_statistics_weights_")

        return cube
