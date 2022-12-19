"""Fixes for IPSL-CM6A-LR model."""
import numpy as np
from iris import NameConstraint
from iris.aux_factory import HybridPressureFactory
from iris.coords import AuxCoord, DimCoord
from iris.cube import CubeList

from ..fix import Fix
from ..shared import cube_to_aux_coord, fix_ocean_depth_coord


class AllVars(Fix):
    """Fixes for thetao."""

    def fix_metadata(self, cubes):
        """
        Fix cell_area coordinate.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        if cube.coords('latitude'):
            cube.coord('latitude').var_name = 'lat'
        if cube.coords('longitude'):
            cube.coord('longitude').var_name = 'lon'
        return CubeList([cube])


class Clcalipso(Fix):
    """Fixes for ``clcalipso``."""

    def fix_metadata(self, cubes):
        """Fix ``alt40`` coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        alt_40_coord = cube.coord('height')
        alt_40_coord.long_name = 'altitude'
        alt_40_coord.standard_name = 'altitude'
        alt_40_coord.var_name = 'alt40'
        return CubeList([cube])


class Cl(Fix):
    """Fixes for ``cl``."""

    def fix_metadata(self, cubes):
        """Fix hybrid pressure level coordinate.

        The data is defined on the cell midpoints of the vertical grid (length
        `n`), whereas the coefficients for the hybrid pressure coordinate (`ap`
        and `b`) are defined for the cell interfaces (length `n+1`). This fix
        fixes this discrepancy.

        Note
        ----
        To calculate the cell midpoints from the cell interfaces (= bounds) we
        assume that the midpoint is in the center between to interfaces.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes

        Returns
        -------
        iris.cube.CubeList
            Fixed cubes.

        """
        # Extract coefficients for hybrid pressure coordinate (ap and b)
        # defined on the cell interfaces and convert them to a coordinate with
        # midpoints and bounds assuming that the midpoint lies in the center
        # between two bounds
        ap_cube = cubes.extract_cube(NameConstraint(var_name='ap'))
        ap_points = (ap_cube.data[:-1] + ap_cube.data[1:]) / 2.0
        ap_bounds = np.stack((ap_cube.data[:-1], ap_cube.data[1:]), axis=-1)
        ap_coord = AuxCoord(
            ap_points,
            bounds=ap_bounds,
            var_name='ap',
            long_name='vertical coordinate formula term: ap(k)',
            units='Pa',
        )

        b_cube = cubes.extract_cube(NameConstraint(var_name='b'))
        b_points = (b_cube.data[:-1] + b_cube.data[1:]) / 2.0
        b_bounds = np.stack((b_cube.data[:-1], b_cube.data[1:]), axis=-1)
        b_coord = AuxCoord(
            b_points,
            bounds=b_bounds,
            var_name='b',
            long_name='vertical coordinate formula term: b(k)',
            units='1',
        )

        # Create lev coordinate
        # Note: lev = a + b with a = ap / p0 (p0 = 100000 Pa)
        lev_coord = DimCoord(
            ap_points / 100000.0 + b_points,
            bounds=(ap_bounds / 100000.0 + b_bounds),
            var_name='lev',
            standard_name='atmosphere_hybrid_sigma_pressure_coordinate',
            long_name='hybrid sigma pressure coordinate',
            units='1',
            attributes={'positive': 'down'},
        )

        # Assign correct coordinates to cube
        cube = self.get_cube_from_list(cubes)
        cube.remove_coord(cube.coord(axis='Z'))
        cube.add_dim_coord(lev_coord, 1)
        cube.add_aux_coord(ap_coord, 1)
        cube.add_aux_coord(b_coord, 1)
        ps_coord = cube_to_aux_coord(
            cubes.extract_cube(NameConstraint(var_name='ps')))
        cube.add_aux_coord(ps_coord, (0, 2, 3))

        # Add coordinate factory
        pressure_coord_factory = HybridPressureFactory(
            delta=ap_coord,
            sigma=b_coord,
            surface_air_pressure=ps_coord,
        )
        cube.add_aux_factory(pressure_coord_factory)

        return cubes


Cli = Cl


Clw = Cl


class Omon(Fix):
    """Fixes for ocean variables."""

    def fix_metadata(self, cubes):
        """Fix ocean depth coordinate.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            if cube.coords(axis='Z'):
                z_coord = cube.coord(axis='Z')
                if z_coord.var_name == 'olevel':
                    fix_ocean_depth_coord(cube)
        return cubes
