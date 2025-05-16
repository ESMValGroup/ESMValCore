"""Metadata operations on data cubes.

Allows for unit conversions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import dask.array as da
import iris
import numpy as np
from iris.coords import AuxCoord, DimCoord

from esmvalcore.iris_helpers import _try_special_unit_conversions

if TYPE_CHECKING:
    from cf_units import Unit
    from iris.cube import Cube

logger = logging.getLogger(__name__)


def convert_units(cube: Cube, units: str | Unit) -> Cube:
    """Convert the units of a cube to new ones (in-place).

    Note
    ----
    Allows special unit conversions which transforms one quantity to another
    (physically related) quantity, which may also change the input cube's
    :attr:`~iris.cube.Cube.standard_name`. These quantities are identified via
    their ``standard_name`` and their ``units`` (units convertible to the ones
    defined are also supported). For example, this enables conversions between
    precipitation fluxes measured in ``kg m-2 s-1`` and precipitation rates
    measured in ``mm day-1`` (and vice versa).

    Currently, the following special conversions are supported:

    * ``precipitation_flux`` (``kg m-2 s-1``) --
      ``lwe_precipitation_rate`` (``mm day-1``)
    * ``water_evaporation_flux`` (``kg m-2 s-1``) --
      ``lwe_water_evaporation_rate`` (``mm day-1``)
    * ``water_potential_evaporation_flux`` (``kg m-2 s-1``) --
      ``None`` (``mm day-1``)
    * ``equivalent_thickness_at_stp_of_atmosphere_ozone_content`` (``m``) --
      ``equivalent_thickness_at_stp_of_atmosphere_ozone_content`` (``DU``)
    * ``surface_air_pressure`` (``Pa``) --
      ``atmosphere_mass_of_air_per_unit_area`` (``kg m-2``)

    Names in the list correspond to ``standard_names`` of the input data.
    Conversions are allowed from each quantity to any other quantity given in a
    bullet point. The corresponding target quantity is inferred from the
    desired target units. In addition, any other units convertible to the ones
    given are also supported (e.g., instead of ``mm day-1``, ``m s-1`` is also
    supported).

    Note that for precipitation and evaporation variables, a water density of
    ``1000 kg m-3`` is assumed.

    Parameters
    ----------
    cube:
        Input cube (modified in place).
    units:
        New units.

    Returns
    -------
    iris.cube.Cube
        Converted cube. Just returned for convenience; input cube is modified
        in place.

    Raises
    ------
    iris.exceptions.UnitConversionError
        Old units are unknown.
    ValueError
        Old units are not convertible to new units.

    """
    try:
        cube.convert_units(units)
    except ValueError:
        if not _try_special_unit_conversions(cube, units):
            raise

    return cube


def accumulate_coordinate(
    cube: Cube,
    coordinate: str | DimCoord | AuxCoord,
) -> Cube:
    """Weight data using the bounds from a given coordinate.

    The resulting cube will then have units given by
    ``cube_units * coordinate_units``.

    Parameters
    ----------
    cube:
        Data cube for the flux.

    coordinate:
        Name of the coordinate that will be used as weights.

    Returns
    -------
    iris.cube.Cube
        Cube with the aggregated data.

    Raises
    ------
    ValueError
        If the coordinate is not found in the cube.

    NotImplementedError
        If the coordinate is multidimensional.
    """
    try:
        coord = cube.coord(coordinate)
    except iris.exceptions.CoordinateNotFoundError as err:
        msg = (
            f"Requested coordinate {coordinate} not found in cube "
            f"{cube.summary(shorten=True)}"
        )
        raise ValueError(
            msg,
        ) from err

    if coord.ndim > 1:
        msg = f"Multidimensional coordinate {coord} not supported."
        raise NotImplementedError(
            msg,
        )

    array_module = da if coord.has_lazy_bounds() else np
    factor = AuxCoord(
        array_module.diff(coord.core_bounds())[..., -1],
        var_name=coord.var_name,
        long_name=coord.long_name,
        units=coord.units,
    )
    result = cube * factor
    unit = result.units.format().split(" ")[-1]
    result.convert_units(unit)
    result.long_name = f"{cube.long_name} * {factor.long_name}"
    return result
