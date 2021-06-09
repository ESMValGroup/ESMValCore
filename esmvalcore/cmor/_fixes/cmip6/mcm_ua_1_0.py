"""Fixes for MCM-UA-1-0 model."""
import iris

from ..fix import Fix
from ..shared import add_scalar_height_coord


def strip_cube_metadata(cube):
    """Remove unnecessary spaces in cube metadata."""
    attributes_to_strip = ('standard_name', 'long_name')
    for attr in attributes_to_strip:
        if getattr(cube, attr) is not None:
            setattr(cube, attr, getattr(cube, attr).strip())
    for coord in cube.coords():
        for attr in attributes_to_strip:
            if getattr(coord, attr) is not None:
                setattr(coord, attr, getattr(coord, attr).strip())


class AllVars(Fix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Remove unnecessary spaces in metadat and rename ``var_name`` of
        latitude and longitude and fix longitude boundary description may be
        wrong (lons=[0, ..., 356.25]; on_bnds=[[-1.875, 1.875], ..., [354.375,
        360]]).

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Cubes to fix.

        Returns
        -------
        iris.cube.Cube

        """
        coords_to_change = {
            'latitude': 'lat',
            'longitude': 'lon',
        }
        for cube in cubes:
            strip_cube_metadata(cube)
            for (std_name, var_name) in coords_to_change.items():
                try:
                    coord = cube.coord(std_name)
                except iris.exceptions.CoordinateNotFoundError:
                    pass
                else:
                    coord.var_name = var_name
            time_units = cube.attributes.get('parent_time_units')
            if time_units is not None:
                cube.attributes['parent_time_units'] = time_units.replace(
                    ' (noleap)', '')

        for cube in cubes:
            coord_names = [cor.standard_name for cor in cube.coords()]
            if 'longitude' in coord_names:
                if cube.coord('longitude').ndim == 1 and \
                        cube.coord('longitude').has_bounds():
                    lon_bnds = cube.coord('longitude').bounds.copy()
                    if cube.coord('longitude').points[0] == 0. and \
                            cube.coord('longitude').points[-1] == 356.25 and \
                            lon_bnds[-1][-1] == 360.:
                        lon_bnds[-1][-1] = 358.125
                    cube.coord('longitude').bounds = lon_bnds

        return cubes


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """Add height (2m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Cubes to fix.

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube, 2.0)
        return [cube]
