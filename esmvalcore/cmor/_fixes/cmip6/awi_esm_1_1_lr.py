"""Fixes for AWI-ESM-1-1-LR model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.cmor._fixes.fix import Fix

if TYPE_CHECKING:
    from collections.abc import Sequence

    from iris.cube import Cube


class AllVars(Fix):
    """Fixes for all vars."""

    def fix_metadata(self, cubes):
        """Fix parent time units.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        parent_units = "parent_time_units"
        bad_value = "days since 0000-01-01 00:00:00"
        for cube in cubes:
            try:
                if parent_units in cube.attributes:
                    if cube.attributes[parent_units] == bad_value:
                        cube.attributes[parent_units] = (
                            "days since 0001-01-01 00:00:00"
                        )
            except AttributeError:
                pass
        return cubes


class FesomSeaIceScalar(Fix):
    """Shared fix for FESOM sea-ice hemispheric scalar diagnostics.

    AWI-ESM's sea-ice/ocean grid (FESOM) has a spurious length-1 'nodes' dimension that isn't part of the CMOR spec
    (a scalar hemispheric integral over time only).
    """

    def fix_metadata(self, cubes: Sequence[Cube]) -> list[Cube]:
        """Remove dimension of length 1 if it is either anonymous (without mapped coordinate) or called 'nodes'."""
        fixed_cubes = []
        for cube in cubes:
            mask_remove = [
                cube.shape[dim] == 1 and not cube.coords(dimensions=dim)
                for dim in range(cube.ndim)
            ]
            if cube.coords("nodes"):
                dim = cube.coord_dims("nodes")[
                    0
                ]  # assumes single coordinate 'nodes'
                if cube.shape[dim] == 1:
                    mask_remove[dim] = True

            if any(mask_remove):
                indices = [
                    0 if remove else slice(None) for remove in mask_remove
                ]
                fixed_cubes.append(cube[tuple(indices)])
            else:
                fixed_cubes.append(cube)
        return fixed_cubes


class Siextentn(FesomSeaIceScalar):
    """Fixes for siextentn."""


class Siextents(FesomSeaIceScalar):
    """Fixes for siextents."""


class Siarean(FesomSeaIceScalar):
    """Fixes for siarean."""


class Siareas(FesomSeaIceScalar):
    """Fixes for siareas."""
