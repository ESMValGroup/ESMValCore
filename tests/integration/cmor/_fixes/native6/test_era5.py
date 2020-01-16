"""Tests for the fixes of ERA5."""
import iris
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.native6.era5 import AllVars, Evspsbl
from esmvalcore.cmor.fix import Fix, fix_metadata
from esmvalcore.cmor.table import CMOR_TABLES


def test_get_evspsbl_fix():
    """Test whether the right fixes are gathered for a single variable."""
    fix = Fix.get_fixes('native6', 'ERA5', 'E1hr', 'evspsbl')
    assert fix == [Evspsbl(None), AllVars(None)]


def make_src_cubes(units, mip='E1hr'):
    """Make dummy cube that looks like the ERA5 data."""

    # Adapt data and time coordinate for different mips
    data = np.arange(27).reshape(3, 3, 3)
    if mip == 'E1hr':
        timestamps = [788928, 788929, 788930] # 3 consecutive hours
    elif mip == 'Amon':
        timestamps = [788928, 789672, 790344] # 3 consecutive months
    elif mip == 'Fx':
        timestamps = [788928]                 # 1 single timestamp
        data = np.arange(9).reshape(1, 3, 3)

    # Create coordinates
    latitude = iris.coords.DimCoord(np.array([90., 0., -90.]),
                                    standard_name='latitude',
                                    long_name='latitude',
                                    var_name='latitude',
                                    units=Unit('degrees'))
    longitude = iris.coords.DimCoord(np.array([0, 180, 359.75]),
                                     standard_name='longitude',
                                     long_name='longitude',
                                     var_name='longitude',
                                     units=Unit('degrees'),
                                     circular=True)
    time = iris.coords.DimCoord(np.array(timestamps, dtype='int32'),
                                standard_name='time',
                                long_name='time',
                                var_name='time',
                                units=Unit(
                                    'hours since 1900-01-01'
                                    '00:00:00.0',
                                    calendar='gregorian'))

    cube = iris.cube.Cube(
        data,
        long_name='random_long_name',
        var_name='random_var_name',
        units=units,
        dim_coords_and_dims=[(time, 0), (latitude, 1), (longitude, 2)],
    )
    return iris.cube.CubeList([cube])


def make_target_cubes(project, mip, short_name):
    """Make dummy cube that looks like the CMORized data."""
    # Look up variable definition in CMOR table
    cmor_table = CMOR_TABLES[project]
    vardef = cmor_table.get_variable(mip, short_name)

    # Make up time dimension and data (shape)
    data = np.arange(27).reshape(3, 3, 3)[:, ::-1, :]
    bounds = None
    if mip == 'E1hr':
        timestamps = [51134.0, 51134.0416667, 51134.0833333]
        if not 'time1' in vardef.dimensions:
            bounds = np.array([[51133.9791667, 51134.0208333],
                               [51134.0208333, 51134.0625],
                               [51134.0625, 51134.1041667]])
    elif mip == 'Amon':
        timestamps = [51149.5, 51179.0, 51208.5]
        if not 'time1' in vardef.dimensions:
            bounds = np.array([[51134.0, 51165.0],
		                       [51165.0, 51193.0],
		                       [51193.0, 51224.0]])
    elif mip == 'Fx':
        data = np.arange(9).reshape(1, 3, 3)[:, ::-1, :]
        timestamps = [51134.0]
        if not 'time1' in vardef.dimensions:
            bounds = np.array([[51133.9791667, 51134.0208333]])

    # Make lat/lon/time coordinates
    latitude = iris.coords.DimCoord(np.array([-90., 0., 90.]),
                                    standard_name='latitude',
                                    long_name='Latitude',
                                    var_name='lat',
                                    units=Unit('degrees_north'),
                                    bounds=np.array([[-90., -45.], [-45., 45.],
                                                     [45., 90.]]))
    longitude = iris.coords.DimCoord(np.array([0, 180, 359.75]),
                                     standard_name='longitude',
                                     long_name='Longitude',
                                     var_name='lon',
                                     units=Unit('degrees_east'),
                                     bounds=np.array([[-0.125, 90.],
                                                      [90., 269.875],
                                                      [269.875, 359.875]]),
                                     circular=True)
    time = iris.coords.DimCoord(np.array(timestamps, dtype=float),
                                standard_name='time',
                                long_name='time',
                                var_name='time',
                                units=Unit('days since 1850-1-1 00:00:00',
                                           calendar='gregorian'),
                                bounds=bounds)

    # Make dummy cube that's the cmor equivalent of the era5 dummy cube.
    attributes = {}
    if vardef.positive:
        attributes['positive'] = vardef.positive

    cube = iris.cube.Cube(
        data.astype('float32'),
        long_name=vardef.long_name,
        var_name=short_name,  # vardef.cmor_name after #391?,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[(time, 0), (latitude, 1), (longitude, 2)],
        attributes=attributes)

    # Add auxiliary height coordinate for certain variables
    if short_name in ['uas', 'vas', 'tas', 'tasmin', 'tasmax']:
        value = 10. if short_name in ['uas', 'vas'] else 2.
        aux_coord = iris.coords.AuxCoord([value],
                                         long_name="height",
                                         standard_name="height",
                                         units=Unit('m'),
                                         var_name="height",
                                         attributes={'positive': 'up'})
        cube.add_aux_coord(aux_coord, ())

    return [cube]


variables = [
    # short_name, mip, era5_units, ndims
    ['pr', 'E1hr', 'm'],
    ['evspsbl', 'E1hr', 'm'],
    ['mrro', 'E1hr', 'm'],
    ['prsn', 'E1hr', 'm of water equivalent'],
    ['evspsblpot', 'E1hr', 'm'],
    ['rss', 'E1hr', 'J m**-2'],
    ['rsds', 'E1hr', 'J m**-2'],
    ['rsdt', 'E1hr', 'J m**-2'],
    ['rls', 'E1hr', 'W m**-2'],  # variables with explicit fixes
    ['uas', 'E1hr', 'm s**-1'],  # a variable without explicit fixes
    ['pr', 'Amon', 'm'],  # a monthly variable
    # ['ua', 'E1hr', 'm s**-1'], # a 4d variable (we decided not to do this now)
    ['orog', 'Fx', 'm**2 s**-2']  # a 2D variable (but keep time coord)
]


@pytest.mark.parametrize('variable', variables)
def test_cmorization(variable):
    """Verify that cmorization results in the expected target cube."""
    short_name, mip, era5_units = variable

    src_cubes = make_src_cubes(era5_units, mip=mip)
    target_cubes = make_target_cubes('native6', mip, short_name)

    print(src_cubes[0].xml())
    out_cubes = fix_metadata(src_cubes, short_name, 'native6', 'era5', mip)

    print(out_cubes[0].xml())  # for testing purposes
    print(target_cubes[0].xml())  # during development
    assert out_cubes[0].xml() == target_cubes[0].xml()
