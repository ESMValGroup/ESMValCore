"""Shared functions for fixes."""
import logging
import os
import warnings
from functools import lru_cache

import dask.array as da
import iris
import pandas as pd
from cf_units import Unit
from scipy.interpolate import interp1d

from esmvalcore.iris_helpers import var_name_constraint

logger = logging.getLogger(__name__)


class AtmosphereSigmaFactory(iris.aux_factory.AuxCoordFactory):
    """Defines an atmosphere sigma coordinate factory."""

    def __init__(self, pressure_at_top=None, sigma=None,
                 surface_air_pressure=None):
        """Create class instance.

        Creates and atmosphere sigma coordinate factory with the formula:

        p(n, k, j, i) = pressure_at_top + sigma(k) *
                        (surface_air_pressure(n, j, i) - pressure_at_top)
        """
        self._metadata_manager = iris.common.metadata_manager_factory(
            iris.common.CoordMetadata)
        super().__init__()
        self._check_dependencies(pressure_at_top, sigma, surface_air_pressure)
        self.units = pressure_at_top.units
        self.pressure_at_top = pressure_at_top
        self.sigma = sigma
        self.surface_air_pressure = surface_air_pressure
        self.standard_name = 'air_pressure'
        self.attributes = {}

    @staticmethod
    def _check_dependencies(pressure_at_top, sigma, surface_air_pressure):
        """Check for sufficient coordinates."""
        if any([
                pressure_at_top is None,
                sigma is None,
                surface_air_pressure is None,
        ]):
            raise ValueError(
                "Unable to construct atmosphere sigma coordinate factory due "
                "to insufficient source coordinates")

        # Check dimensions
        if pressure_at_top.shape not in ((), (1, )):
            raise ValueError(
                f"Expected scalar 'pressure_at_top' coordinate, got shape "
                f"{pressure_at_top.shape}")

        # Check bounds
        if sigma.nbounds not in (0, 2):
            raise ValueError(
                f"Invalid 'sigma' coordinate: must have either 0 or 2 bounds, "
                f"got {sigma.nbounds:d}")
        for coord in (pressure_at_top, surface_air_pressure):
            if coord.nbounds:
                msg = (f"Coordinate '{coord.name()}' has bounds. These will "
                       "be disregarded")
                warnings.warn(msg, UserWarning, stacklevel=2)

        # Check units
        if sigma.units.is_unknown():
            sigma.units = Unit('1')
        if not sigma.units.is_dimensionless():
            raise ValueError(
                f"Invalid units: 'sigma' must be dimensionless, got "
                f"'{sigma.units}'")
        if pressure_at_top.units != surface_air_pressure.units:
            raise ValueError(
                f"Incompatible units: 'pressure_at_top' and "
                f"'surface_air_pressure' must have the same units, got "
                f"'{pressure_at_top.units}' and "
                f"'{surface_air_pressure.units}'")
        if not pressure_at_top.units.is_convertible('Pa'):
            raise ValueError(
                "Invalid units: 'pressure_at_top' and 'surface_air_pressure' "
                "must have units of pressure")

    @property
    def dependencies(self):
        """Return dependencies."""
        dependencies = {
            'pressure_at_top': self.pressure_at_top,
            'sigma': self.sigma,
            'surface_air_pressure': self.surface_air_pressure,
        }
        return dependencies

    @staticmethod
    def _derive(pressure_at_top, sigma, surface_air_pressure):
        """Derive coordinate."""
        return pressure_at_top + sigma * (surface_air_pressure -
                                          pressure_at_top)

    def make_coord(self, coord_dims_func):
        """Make new :class:`iris.coords.AuxCoord`."""
        # Which dimensions are relevant?
        derived_dims = self.derived_dims(coord_dims_func)
        dependency_dims = self._dependency_dims(coord_dims_func)

        # Build the points array
        nd_points_by_key = self._remap(dependency_dims, derived_dims)
        points = self._derive(nd_points_by_key['pressure_at_top'],
                              nd_points_by_key['sigma'],
                              nd_points_by_key['surface_air_pressure'])

        # Bounds
        bounds = None
        if self.sigma.nbounds:
            nd_values_by_key = self._remap_with_bounds(dependency_dims,
                                                       derived_dims)
            pressure_at_top = nd_values_by_key['pressure_at_top']
            sigma = nd_values_by_key['sigma']
            surface_air_pressure = nd_values_by_key['surface_air_pressure']
            ok_bound_shapes = [(), (1,), (2,)]
            if sigma.shape[-1:] not in ok_bound_shapes:
                raise ValueError("Invalid sigma coordinate bounds")
            if pressure_at_top.shape[-1:] not in [(), (1,)]:
                warnings.warn(
                    "Pressure at top coordinate has bounds. These are being "
                    "disregarded")
                pressure_at_top_pts = nd_points_by_key['pressure_at_top']
                bds_shape = list(pressure_at_top_pts.shape) + [1]
                pressure_at_top = pressure_at_top_pts.reshape(bds_shape)
            if surface_air_pressure.shape[-1:] not in [(), (1,)]:
                warnings.warn(
                    "Surface pressure coordinate has bounds. These are being "
                    "disregarded")
                surface_air_pressure_pts = nd_points_by_key[
                    'surface_air_pressure']
                bds_shape = list(surface_air_pressure_pts.shape) + [1]
                surface_air_pressure = surface_air_pressure_pts.reshape(
                    bds_shape)
            bounds = self._derive(pressure_at_top, sigma, surface_air_pressure)

        # Create coordinate
        return iris.coords.AuxCoord(
            points, standard_name=self.standard_name, long_name=self.long_name,
            var_name=self.var_name, units=self.units, bounds=bounds,
            attributes=self.attributes, coord_system=self.coord_system)

    def update(self, old_coord, new_coord=None):
        """Notify the factory of the removal/replacement of a coordinate."""
        new_dependencies = self.dependencies
        for (name, coord) in self.dependencies.items():
            if old_coord is coord:
                new_dependencies[name] = new_coord
                try:
                    self._check_dependencies(**new_dependencies)
                except ValueError as exc:
                    raise ValueError(f"Failed to update dependencies: {exc}")
                else:
                    setattr(self, name, new_coord)
                break


def add_sigma_factory(cube):
    """Add factory for ``atmosphere_sigma_coordinate``.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube.

    Raises
    ------
    ValueError
        ``cube`` does not contain coordinate ``atmosphere_sigma_coordinate``.
    """
    if cube.coords('atmosphere_sigma_coordinate'):
        aux_factory = AtmosphereSigmaFactory(
            pressure_at_top=cube.coord(var_name='ptop'),
            sigma=cube.coord(var_name='lev'),
            surface_air_pressure=cube.coord(var_name='ps'),
        )
        cube.add_aux_factory(aux_factory)
        return
    raise ValueError(
        "Cannot add 'air_pressure' coordinate, 'atmosphere_sigma_coordinate' "
        "coordinate not available")


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
        coord_cube = cubes.extract(var_name_constraint(coord_name))
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
                                              var_name='plev',
                                              standard_name='air_pressure',
                                              long_name='pressure',
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
                                              var_name='alt',
                                              standard_name='altitude',
                                              long_name='altitude',
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
        cube = cubes.extract(var_name_constraint(bound_var))
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
