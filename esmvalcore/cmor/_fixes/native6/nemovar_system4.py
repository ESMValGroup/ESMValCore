"""Fixes for ERA5."""
import logging
import os

import cf_units
import iris
import iris.exceptions
import numpy as np

from ..fix import Fix

logger = logging.getLogger(__name__)


class AllVars(Fix):
    """Fixes for all variables."""

    _points = {}
    _bounds = {}

    _nemo_names = {
        'thetao': 'votemper',
        'tos': 'sosstsst',
        'so': 'vosaline',
        'sos': 'sosaline',
    }

    _nemo_units = {
        'C': 'degC',
    }

    @classmethod
    def get_points(cls, name):
        if name not in cls._points:
            points = np.load(
                os.path.join(os.path.dirname(__file__), f"{name}.npy"))
            cls._points[name] = points
            cls._bounds[name] = cls.create_bounds(points, name)
        return cls._points[name]

    @classmethod
    def get_bounds(cls, name):
        return cls._bounds[name]

    def fix_metadata(self, cubes):
        """Fix metadata."""
        fixed_cubes = iris.cube.CubeList()
        try:
            cube = cubes.extract(self._nemo_names[self.vardef.short_name])[0]
        except IndexError:
            return fixed_cubes
        cube.standard_name = self.vardef.standard_name
        cube.long_name = self.vardef.long_name
        cube.var_name = self.vardef.short_name
        cube.units = self._nemo_units.get(cube.units.origin, cube.units)
        lat = cube.coord('latitude')
        lat_dims = cube.coord_dims(lat)
        i = iris.coords.DimCoord(np.arange(1,
                                           lat.shape[1] + 1).astype(np.int32),
                                 long_name='cell index along first dimension',
                                 units='1',
                                 var_name='i')
        cube.add_dim_coord(i, lat_dims[1])
        j = iris.coords.DimCoord(np.arange(1,
                                           lat.shape[0] + 1).astype(np.int32),
                                 long_name='cell index along second dimension',
                                 units='1',
                                 var_name='j')
        cube.add_dim_coord(j, lat_dims[0])

        date = os.path.basename(cube.attributes['source_file']).split('_')[2]
        year = int(date[0:4])
        month = int(date[4:6])
        cube.remove_coord('time_counter')
        cube.add_dim_coord(
            iris.coords.DimCoord(
                points=[
                    0.,
                ],
                var_name='time',
                standard_name='time',
                units=cf_units.Unit(
                    f'seconds since {year}-{month:02d}-01 00:00:00',
                    calendar='proleptic_gregorian'),
            ), 0)
        if cube.coords('model_level_number'):
            depth = cube.coord('model_level_number')
            depth.var_name = 'lev'
            depth.standard_name = 'depth'
            depth.long_name = 'ocean depth coordinate'

        for name in ['latitude', 'longitude']:
            cube.replace_coord(
                cube.coord(name).copy(
                    self.get_points(name),
                    self.get_bounds(name),
                ))
            cube.coord(name).units = self.vardef.coordinates[name].units
            del cube.coord(name).attributes['valid_min']
            del cube.coord(name).attributes['valid_max']
        fixed_cubes.append(cube)
        return fixed_cubes

    @staticmethod
    def create_bounds(points, name):
        if name == 'latitude':
            return AllVars.create_vertex_lats(points)
        else:
            return AllVars.create_vertex_lons(points)

    @staticmethod
    def create_vertex_lons(a):
        ny = a.shape[0]
        nx = a.shape[1]
        if nx == 1:  # Longitudes were integrated out
            if ny == 1:
                return np.array([a[0, 0]])
            return np.zeros([ny, 2])
        b = np.zeros([ny, nx, 4])
        b[:, 1:nx, 0] = 0.5 * (a[:, 0:nx - 1] + a[:, 1:nx])
        b[:, 0, 0] = 1.5 * a[:, 0] - 0.5 * a[:, 1]
        b[:, 0:nx - 1, 1] = b[:, 1:nx, 0]
        b[:, nx - 1, 1] = 1.5 * a[:, nx - 1] - 0.5 * a[:, nx - 2]
        b[:, :, 2] = b[:, :, 1]
        b[:, :, 3] = b[:, :, 0]
        b[b < 0] = b[b < 0] + 360.
        return b

    @staticmethod
    def create_vertex_lats(a):
        ny = a.shape[0]
        nx = a.shape[1]
        f = np.vectorize(lambda x: (x + 90) % 180 - 90)
        if nx == 1:  # Longitudes were integrated out
            if ny == 1:
                return f(np.array([a[0, 0]]))
            b = np.zeros([ny, 2])
            b[1:ny, 0] = f(0.5 * (a[0:ny - 1, 0] + a[1:ny, 0]))
            b[0, 0] = f(2 * a[0, 0] - b[1, 0])
            b[0:ny - 1, 1] = b[1:ny, 0]
            b[ny - 1, 1] = f(1.5 * a[ny - 1, 0] - 0.5 * a[ny - 2, 0])
            return b
        b = np.zeros([ny, nx, 4])
        b[1:ny, :, 0] = f(0.5 * (a[0:ny - 1, :] + a[1:ny, :]))
        b[0, :, 0] = f(2 * a[0, :] - b[1, :, 0])
        b[:, :, 1] = b[:, :, 0]
        b[0:ny - 1, :, 2] = b[1:ny, :, 0]
        b[ny - 1, :, 2] = f(1.5 * a[ny - 1, :] - 0.5 * a[ny - 2, :])
        b[:, :, 3] = b[:, :, 2]
        return b
