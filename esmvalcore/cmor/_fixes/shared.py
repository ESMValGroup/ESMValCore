"""Shared functions for fixes."""
import logging
import os
from functools import lru_cache

import dask.array as da
import iris
import pandas as pd
from cf_units import Unit
from iris import NameConstraint
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def add_aux_coords_from_cubes(cube, cubes, coord_dict):
    """Add auxiliary coordinate to cube from another cube in list of cubes.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube to which the auxiliary coordinates will be added.
    cubes : iris.cube.CubeList
        List of cubes which contains the desired coordinates as single cubes.
    coord_dict : dict
        Dictionary of the form ``coord_name: coord_dims``, where ``coord_name``
        is the ``var_name`` (:obj:`str`) of the desired coordinates and
        ``coord_dims`` a :obj:`tuple` of :obj:`int` describing the coordinate
        dimensions in ``cube``.

    Raises
    ------
    ValueError
        ``cubes`` do not contain a desired coordinate or multiple copies of
        it.
    """
    for (coord_name, coord_dims) in coord_dict.items():
        coord_cube = cubes.extract(NameConstraint(var_name=coord_name))
        if len(coord_cube) != 1:
            raise ValueError(
                f"Expected exactly one coordinate cube '{coord_name}' in "
                f"list of cubes {cubes}, got {len(coord_cube):d}")
        coord_cube = coord_cube[0]
        aux_coord = cube_to_aux_coord(coord_cube)
        cube.add_aux_coord(aux_coord, coord_dims)
        cubes.remove(coord_cube)


def add_plev_from_altitude(cube):
    """Add pressure level coordinate from altitude coordinate.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube.

    Raises
    ------
    ValueError
        ``cube`` does not contain coordinate ``altitude``.
    """
    if cube.coords('altitude'):
        height_coord = cube.coord('altitude')
        if height_coord.units != 'm':
            height_coord.convert_units('m')
        altitude_to_pressure = get_altitude_to_pressure_func()
        pressure_points = altitude_to_pressure(height_coord.core_points())
        if height_coord.core_bounds() is None:
            pressure_bounds = None
        else:
            pressure_bounds = altitude_to_pressure(height_coord.core_bounds())
        pressure_coord = iris.coords.AuxCoord(pressure_points,
                                              bounds=pressure_bounds,
                                              standard_name='air_pressure',
                                              units='Pa')
        cube.add_aux_coord(pressure_coord, cube.coord_dims(height_coord))
        return
    raise ValueError(
        "Cannot add 'air_pressure' coordinate, 'altitude' coordinate not "
        "available")


def add_altitude_from_plev(cube):
    """Add altitude coordinate from pressure level coordinate.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube.

    Raises
    ------
    ValueError
        ``cube`` does not contain coordinate ``air_pressure``.
    """
    if cube.coords('air_pressure'):
        plev_coord = cube.coord('air_pressure')
        if plev_coord.units != 'Pa':
            plev_coord.convert_units('Pa')
        pressure_to_altitude = get_pressure_to_altitude_func()
        altitude_points = pressure_to_altitude(plev_coord.core_points())
        if plev_coord.core_bounds() is None:
            altitude_bounds = None
        else:
            altitude_bounds = pressure_to_altitude(plev_coord.core_bounds())
        altitude_coord = iris.coords.AuxCoord(altitude_points,
                                              bounds=altitude_bounds,
                                              standard_name='altitude',
                                              units='m')
        cube.add_aux_coord(altitude_coord, cube.coord_dims(plev_coord))
        return
    raise ValueError(
        "Cannot add 'altitude' coordinate, 'air_pressure' coordinate not "
        "available")


def add_scalar_depth_coord(cube, depth=0.0):
    """Add scalar coordinate 'depth' with value of `depth`m."""
    logger.debug("Adding depth coordinate (%sm)", depth)
    depth_coord = iris.coords.AuxCoord(depth,
                                       var_name='depth',
                                       standard_name='depth',
                                       long_name='depth',
                                       units=Unit('m'),
                                       attributes={'positive': 'down'})
    try:
        cube.coord('depth')
    except iris.exceptions.CoordinateNotFoundError:
        cube.add_aux_coord(depth_coord, ())
    return cube


def add_scalar_height_coord(cube, height=2.0):
    """Add scalar coordinate 'height' with value of `height`m."""
    logger.debug("Adding height coordinate (%sm)", height)
    height_coord = iris.coords.AuxCoord(height,
                                        var_name='height',
                                        standard_name='height',
                                        long_name='height',
                                        units=Unit('m'),
                                        attributes={'positive': 'up'})
    try:
        cube.coord('height')
    except iris.exceptions.CoordinateNotFoundError:
        cube.add_aux_coord(height_coord, ())
    return cube


def add_scalar_lambda550nm_coord(cube):
    """Add scalar coordinate 'lambda550nm'."""
    logger.debug("Adding lambda550nm coordinate")
    lambda550nm_coord = iris.coords.AuxCoord(
        550.0,
        var_name='wavelength',
        standard_name='radiation_wavelength',
        long_name='Radiation Wavelength 550 nanometers',
        units='nm',
    )
    try:
        cube.coord('radiation_wavelength')
    except iris.exceptions.CoordinateNotFoundError:
        cube.add_aux_coord(lambda550nm_coord, ())
    return cube


def add_scalar_typeland_coord(cube, value='default'):
    """Add scalar coordinate 'typeland' with value of `value`."""
    logger.debug("Adding typeland coordinate (%s)", value)
    typeland_coord = iris.coords.AuxCoord(value,
                                          var_name='type',
                                          standard_name='area_type',
                                          long_name='Land area type',
                                          units=Unit('no unit'))
    try:
        cube.coord('area_type')
    except iris.exceptions.CoordinateNotFoundError:
        cube.add_aux_coord(typeland_coord, ())
    return cube


def add_scalar_typesea_coord(cube, value='default'):
    """Add scalar coordinate 'typesea' with value of `value`."""
    logger.debug("Adding typesea coordinate (%s)", value)
    typesea_coord = iris.coords.AuxCoord(value,
                                         var_name='type',
                                         standard_name='area_type',
                                         long_name='Ocean area type',
                                         units=Unit('no unit'))
    try:
        cube.coord('area_type')
    except iris.exceptions.CoordinateNotFoundError:
        cube.add_aux_coord(typesea_coord, ())
    return cube


def add_scalar_typesi_coord(cube, value='sea_ice'):
    """Add scalar coordinate 'typesi' with value of `value`."""
    logger.debug("Adding typesi coordinate (%s)", value)
    typesi_coord = iris.coords.AuxCoord(value,
                                        var_name='type',
                                        standard_name='area_type',
                                        long_name='Sea Ice area type',
                                        units=Unit('no unit'))
    try:
        cube.coord('area_type')
    except iris.exceptions.CoordinateNotFoundError:
        cube.add_aux_coord(typesi_coord, ())
    return cube


def cube_to_aux_coord(cube):
    """Convert cube to iris AuxCoord."""
    return iris.coords.AuxCoord(
        points=cube.core_data(),
        var_name=cube.var_name,
        standard_name=cube.standard_name,
        long_name=cube.long_name,
        units=cube.units,
    )


@lru_cache(maxsize=None)
def get_altitude_to_pressure_func():
    """Get function converting altitude [m] to air pressure [Pa].

    Returns
    -------
    callable
        Function that converts altitude to air pressure.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_file = os.path.join(base_dir, 'us_standard_atmosphere.csv')
    data_frame = pd.read_csv(source_file, comment='#')
    func = interp1d(data_frame['Altitude [m]'],
                    data_frame['Pressure [Pa]'],
                    kind='cubic',
                    fill_value='extrapolate')
    return func


def get_bounds_cube(cubes, coord_var_name):
    """Find bound cube for a given variable in a :class:`iris.cube.CubeList`.

    Parameters
    ----------
    cubes : iris.cube.CubeList
        List of cubes containing the coordinate bounds for the desired
        coordinate as single cube.
    coord_var_name : str
        ``var_name`` of the desired coordinate (without suffix ``_bnds`` or
        ``_bounds``).

    Returns
    -------
    iris.cube.Cube
        Bounds cube.

    Raises
    ------
    ValueError
        ``cubes`` do not contain the desired coordinate bounds or multiple
        copies of them.
    """
    for bounds in ('bnds', 'bounds'):
        bound_var = f'{coord_var_name}_{bounds}'
        cube = cubes.extract(NameConstraint(var_name=bound_var))
        if len(cube) == 1:
            return cube[0]
        if len(cube) > 1:
            raise ValueError(
                f"Multiple cubes with var_name '{bound_var}' found")
    raise ValueError(
        f"No bounds for coordinate variable '{coord_var_name}' available in "
        f"cubes\n{cubes}")


@lru_cache(maxsize=None)
def get_pressure_to_altitude_func():
    """Get function converting air pressure [Pa] to altitude [m].

    Returns
    -------
    callable
        Function that converts air pressure to altitude.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_file = os.path.join(base_dir, 'us_standard_atmosphere.csv')
    data_frame = pd.read_csv(source_file, comment='#')
    func = interp1d(data_frame['Pressure [Pa]'],
                    data_frame['Altitude [m]'],
                    kind='cubic',
                    fill_value='extrapolate')
    return func


def fix_bounds(cube, cubes, coord_var_names):
    """Fix bounds for cube that could not be read correctly by :mod:`iris`.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube whose coordinate bounds will be fixed.
    cubes : iris.cube.CubeList
        List of cubes which contains the desired coordinate bounds as single
        cubes.
    coord_var_names : list of str
        ``var_name``s of the desired coordinates (without suffix ``_bnds`` or
        ``_bounds``).

    Raises
    ------
    ValueError
        ``cubes`` do not contain a desired coordinate bounds or multiple copies
        of them.
    """
    for coord_var_name in coord_var_names:
        coord = cube.coord(var_name=coord_var_name)
        if coord.bounds is not None:
            continue
        bounds_cube = get_bounds_cube(cubes, coord_var_name)
        cube.coord(var_name=coord_var_name).bounds = bounds_cube.core_data()
        logger.debug("Fixed bounds of coordinate '%s'", coord_var_name)


def round_coordinates(cubes, decimals=5, coord_names=None):
    """Round all dimensional coordinates of all cubes in place.

    Cubes can be a list of Iris cubes, or an Iris `CubeList`.

    Cubes are modified *in place*. The return value is simply for
    convenience.

    Parameters
    ----------
    cubes : iris.cube.CubeList or list of iris.cube.Cube
        Cubes which are modified in place.

    decimals : int
        Number of decimals to round to.

    coord_names : list of str or None
        If ``None`` (or a falsey value), all dimensional coordinates will be
        rounded. Otherwise, only coordinates given by the names in
        ``coord_names`` are rounded.

    Returns
    -------
    iris.cube.CubeList or list of iris.cube.Cube
        The modified input ``cubes``.
    """
    for cube in cubes:
        if not coord_names:
            coords = cube.coords(dim_coords=True)
        else:
            coords = [cube.coord(c) for c in coord_names if cube.coords(c)]
        for coord in coords:
            coord.points = da.round(da.asarray(coord.core_points()), decimals)
            if coord.bounds is not None:
                coord.bounds = da.round(da.asarray(coord.core_bounds()),
                                        decimals)
    return cubes


def fix_ocean_depth_coord(cube):
    """Fix attributes of ocean vertical level coordinate.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube.
    """
    depth_coord = cube.coord(axis='Z')
    depth_coord.standard_name = 'depth'
    depth_coord.var_name = 'lev'
    depth_coord.units = 'm'
    depth_coord.long_name = 'ocean depth coordinate'
    depth_coord.attributes = {'positive': 'down'}
