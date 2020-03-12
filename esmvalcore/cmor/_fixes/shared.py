"""Shared functions for fixes."""
import logging
import warnings

import dask.array as da
import iris
from cf_units import Unit

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
        super().__init__()
        self._check_dependencies(pressure_at_top, sigma, surface_air_pressure)
        self.pressure_at_top = pressure_at_top
        self.sigma = sigma
        self.surface_air_pressure = surface_air_pressure
        self.standard_name = 'air_pressure'
        self.attributes = {}

    @property
    def units(self):
        """Units."""
        units = self.pressure_at_top.units
        return units

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
                "Invalid 'sigma' coordinate: must have either 0 or 2 bounds, "
                "got {sigma.nbounds:d}")
        for coord in (pressure_at_top, surface_air_pressure):
            if coord.nbounds:
                msg = (f"Coordinate '{coord.name()}' has bounds. These will "
                       "be disregarded")
                warnings.warn(msg, UserWarning, stacklevel=2)

        # Check units
        if not sigma.units.is_dimensionless():
            raise ValueError(
                "Invalid units: 'sigma' must be dimensionless, got "
                "'{sigma.units}'")
        if pressure_at_top.units != surface_air_pressure.units:
            raise ValueError(
                "Incompatible units: 'pressure_at_top' and "
                "'surface_air_pressure' must have the same units, got "
                "'{pressure_at_top.units}' and '{surface_air_pressure.units}'")
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
    """Add factory for ``atmosphere_sigma_coordinate``."""
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


def cube_to_aux_coord(cube):
    """Convert cube to iris AuxCoord"""
    return iris.coords.AuxCoord(
        points=cube.core_data(),
        var_name=cube.var_name,
        standard_name=cube.standard_name,
        long_name=cube.long_name,
        units=cube.units,
    )


def round_coordinates(cubes, decimals=5, coord_names=None):
    """Round all dimensional coordinates of all cubes in place

    Cubes can be a list of Iris cubes, or an Iris `CubeList`.

    Cubes are modified *in place*. The return value is simply for
    convenience.

    Parameters
    ----------
    - cubes: iris.cube.CubeList (or a list of iris.cube.Cube).

    - decimals: number of decimals to round to.

    - coord_names: list of strings, or None.
        If None (or a falsey value), all dimensional coordinates will
        be rounded.
        Otherwise, only coordinates given by the names in
        `coord_names` are rounded.

    Returns
    -------
    The modified input `cubes`

    """

    for cube in cubes:
        if not coord_names:
            coords = cube.coords(dim_coords=True)
        else:
            coords = [cube.coord(name) for name in coord_names]
        for coord in coords:
            coord.points = da.round(da.asarray(coord.core_points()), decimals)
            if coord.bounds is not None:
                coord.bounds = da.round(da.asarray(coord.core_bounds()),
                                        decimals)
    return cubes
