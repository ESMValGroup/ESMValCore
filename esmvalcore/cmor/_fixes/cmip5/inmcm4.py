
"""Fixes for inmcm4 model."""
import iris

from ..fix import Fix


class Gpp(Fix):
    """Fixes for gpp."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= -1
        cube.metadata = metadata
        return cube


class Lai(Fix):
    """Fixes for lai."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 0.01
        cube.metadata = metadata
        return cube


class Nbp(Fix):
    """Fixes for nbp."""

    def fix_metadata(self, cubes):
        """
        Fix missing scalar dimension.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        cubes[0].standard_name = (
            'surface_net_downward_mass_flux_of_carbon_dioxide_expressed_as_'
            'carbon_due_to_all_land_processes'
        )
        return cubes


class BaresoilFrac(Fix):
    """Fixes for baresoilFrac."""

    def fix_metadata(self, cubes):
        """
        Fix missing scalar dimension.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        typebare = iris.coords.AuxCoord(
            'bare_ground',
            standard_name='area_type',
            long_name='surface type',
            var_name='type',
            units='1',
            bounds=None)
        for cube in cubes:
            cube.add_aux_coord(typebare)
        return cubes
