"""Fixes for EC-Earth3-AerChem model."""

from esmvalcore.cmor._fixes.fix import Fix


class Oh(Fix):
    """Fixes for oh."""

    def fix_metadata(self, cubes):
        """Fix standard name for ps.

        Fix standard_name for Surface Air Pressure (ps).
        See discussion in
        https://github.com/ESMValGroup/ESMValCore/issues/2613
        Cube has two coordinates called air_pressure: an AuxCoord ps
        and a DerivedCoord that is 4D and derived using formula terms,
        we are setting the former's standard_name to "surface_air_pressure".

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        cube = self.get_cube_from_list(cubes)

        for cube in cubes:
            for coord in cube.coords():
                if coord.var_name == "ps":
                    coord.standard_name = "surface_air_pressure"

        return cubes
