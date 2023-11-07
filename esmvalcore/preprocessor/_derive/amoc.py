"""Derivation of variable `amoc`."""
import iris
import numpy as np

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `amoc`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        if project == "CMIP5":
            required = [{'short_name': 'msftmyz', 'mip': 'Omon'}]
        elif project == "CMIP6":
            required = [{'short_name': 'msftmz', 'optional': True},
                        {'short_name': 'msftyz', 'optional': True}]
        else:
            raise ValueError(f"Project {project} can not be used "
                             f"for Amoc derivation.")

        return required

    @staticmethod
    def calculate(cubes):
        """Compute Atlantic meriodinal overturning circulation.

        Arguments
        ---------
        cube: iris.cube.Cube
           input cube.

        Returns
        -------
        iris.cube.Cube
              Output AMOC cube.
        """
        # 0. Load the msft(m)(y)z cube.
        try:
            # msftmyz and msfmz
            cube = cubes.extract_cube(
                iris.Constraint(
                    name='ocean_meridional_overturning_mass_streamfunction'))
            meridional = True
            lats = cube.coord('latitude').points
        except iris.exceptions.ConstraintMismatchError:
            # msftyz
            cube = cubes.extract_cube(
                iris.Constraint(
                    name='ocean_y_overturning_mass_streamfunction'))
            meridional = False
            lats = cube.coord('grid_latitude').points

        cube_orig = cube.copy()

        # 1: find the relevant region
        atl_constraint = iris.Constraint(region='atlantic_arctic_ocean')
        cube = cube.extract(constraint=atl_constraint)

        if cube is None:
            raise ValueError(f"Amoc calculation: {cube_orig} doesn't contain"
                             f" atlantic_arctic_ocean.")

        # 2: Remove the shallowest 500m to avoid wind driven mixed layer.
        depth_constraint = iris.Constraint(depth=lambda d: d >= 500.)
        cube = cube.extract(constraint=depth_constraint)

        # 3: Find the latitude closest to 26.5N (location of RAPID measurement)
        rapid_location = 26.5
        rapid_index = np.argmin(np.abs(lats - rapid_location))

        if meridional:
            rapid_constraint = iris.Constraint(latitude=lats[rapid_index])
        else:
            rapid_constraint = iris.Constraint(grid_latitude=lats[rapid_index])

        cube = cube.extract(constraint=rapid_constraint)

        # 4: find the maximum in the water column along the time axis.
        cube = cube.collapsed(
            ['depth', 'region'],
            iris.analysis.MAX,
        )

        return cube
