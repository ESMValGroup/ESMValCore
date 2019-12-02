"""Derivation of variable `ohcgt`."""
import iris
from iris import Constraint

from dask import array as da
from cf_units import Unit
import  numpy as np
from ._baseclass import DerivedVariableBase

RHO_CP = iris.coords.AuxCoord(4.09169e+6, units=Unit('kg m-3 J kg-1 K-1'))


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `ohcgt`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'thetaoga'
            },
            {
                'short_name': 'volcello',
                'mip': 'fx'
            },
        ]
        if project == 'CMIP6':
            required = [
                {
                    'short_name': 'thetaoga'
                },
                {
                    'short_name': 'volcello',
                    'mip': 'Omon',
                },

            ]
        return required

    @staticmethod
    def calculate(cubes):
        """
        Compute ocean heat content.

        Use c_p*rho_0= 4.09169e+6 J m-3 K-1
        (Kuhlbrodt et al., 2015, Clim. Dyn.)

        Arguments
        ---------
        cube: iris.cube.Cube
           input cube.

        Returns
        -------
        iris.cube.Cube
              Output OHC cube.
        """
        # 1. Load the thetao and volcello cubes
        cube = cubes.extract_strict(
            Constraint(cube_func=lambda c: c.var_name == 'thetaoga'))
        volume = cubes.extract_strict(
            Constraint(cube_func=lambda c: c.var_name == 'volcello'))
        # 2. multiply with each other and with cprho0
        # some juggling with coordinates needed since Iris is very
        # restrictive in this regard
        try:
            t_coord_dims = cube.coord_dims('time')
        except iris.exceptions.CoordinateNotFoundError:
            time_coord_present = False
        else:
            time_coord_present = True
            t_coord_dim = t_coord_dims[0]
            dim_coords = [(coord, cube.coord_dims(coord)[0])
                          for coord in cube.coords(
                              contains_dimension=t_coord_dim, dim_coords=True)]
            aux_coords = [
                (coord, cube.coord_dims(coord))
                for coord in cube.coords(contains_dimension=t_coord_dim,
                                         dim_coords=False)
            ]
            for coord, dims in dim_coords + aux_coords:
                cube.remove_coord(coord)
        volume = volume.data

        if volume.ndim == 3:
            volume = da.sum(volume.data)
            volume = np.tile(volume, [cube.data.shape[0],])

        elif volume.ndim == 4:
            volume = da.sum(volume.data, axis=(1,2,3))

        else:
            print(cube.data.shape , 'does not match', volume.data.shape)
            assert 0

        const = 4.09169e+6
        cube.data = cube.data * volume * const
        if time_coord_present:
            for coord, dim in dim_coords:
                cube.add_dim_coord(coord, dim)
            for coord, dims in aux_coords:
                cube.add_aux_coord(coord, dims)
        return cube
