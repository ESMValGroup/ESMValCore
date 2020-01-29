"""Tests for the fixes of ERA5."""
import iris
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.native6.era5 import AllVars, Evspsbl, FixEra5
from esmvalcore.cmor.fix import Fix, fix_metadata
from esmvalcore.cmor.table import CMOR_TABLES


def test_get_evspsbl_fix():
    """Test whether the right fixes are gathered for a single variable."""
    fix = Fix.get_fixes('native6', 'ERA5', 'E1hr', 'evspsbl')
    assert fix == [Evspsbl(None), AllVars(None)]


def test_get_frequency_hourly():
    fix = FixEra5(None)
    time = iris.coords.DimCoord([0, 1, 2],
                                standard_name='time',
                                units=Unit('hours since 1900-01-01'))
    cube = iris.cube.Cube([1, 6, 3],
                          var_name='random_var',
                          dim_coords_and_dims=[(time, 0)])
    assert fix._frequency(cube) == 'hourly'
    cube.coord('time').convert_units('days since 1850-1-1 00:00:00.0')
    assert fix._frequency(cube) == 'hourly'


def test_get_frequency_monthly():
    fix = FixEra5(None)
    time = iris.coords.DimCoord([0, 31, 59],
                                standard_name='time',
                                units=Unit('hours since 1900-01-01'))
    cube = iris.cube.Cube([1, 6, 3],
                          var_name='random_var',
                          dim_coords_and_dims=[(time, 0)])
    assert fix._frequency(cube) == 'monthly'
    cube.coord('time').convert_units('days since 1850-1-1 00:00:00.0')
    assert fix._frequency(cube) == 'monthly'


def test_get_frequency_fx():
    fix = FixEra5(None)
    cube = iris.cube.Cube(1., long_name='Cube without time coordinate')
    assert fix._frequency(cube) == 'fx'
    time = iris.coords.DimCoord(0,
                                standard_name='time',
                                units=Unit('hours since 1900-01-01'))
    cube = iris.cube.Cube([1],
                          var_name='cube_with_length_1_time_coord',
                          long_name='Geopotential',
                          dim_coords_and_dims=[(time, 0)])
    assert fix._frequency(cube) == 'fx'
    cube.long_name = 'Not geopotential'
    with pytest.raises(ValueError):
        fix._frequency(cube)


def _era5_latitude():
    return iris.coords.DimCoord(np.array([90., 0., -90.]),
                                standard_name='latitude',
                                long_name='latitude',
                                var_name='latitude',
                                units=Unit('degrees'))


def _era5_longitude():
    return iris.coords.DimCoord(np.array([0, 180, 359.75]),
                                standard_name='longitude',
                                long_name='longitude',
                                var_name='longitude',
                                units=Unit('degrees'),
                                circular=True)


def _era5_time(frequency):
    if frequency == 'invariant':
        timestamps = [788928]
    elif frequency == 'hourly':
        timestamps = [788928, 788929, 788930]
    elif frequency == 'monthly':
        timestamps = [788928, 789672, 790344]
    return iris.coords.DimCoord(np.array(timestamps, dtype='int32'),
                                standard_name='time',
                                long_name='time',
                                var_name='time',
                                units=Unit(
                                    'hours since 1900-01-01'
                                    '00:00:00.0',
                                    calendar='gregorian'))


def _era5_data(frequency):
    if frequency == 'invariant':
        return np.arange(9).reshape(1, 3, 3)
    return np.arange(27).reshape(3, 3, 3)


def _cmor_latitude():
    return iris.coords.DimCoord(np.array([-90., 0., 90.]),
                                standard_name='latitude',
                                long_name='Latitude',
                                var_name='lat',
                                units=Unit('degrees_north'),
                                bounds=np.array([[-90., -45.], [-45., 45.],
                                                 [45., 90.]]))


def _cmor_longitude():
    return iris.coords.DimCoord(np.array([0, 180, 359.75]),
                                standard_name='longitude',
                                long_name='Longitude',
                                var_name='lon',
                                units=Unit('degrees_east'),
                                bounds=np.array([[-0.125, 90.], [90., 269.875],
                                                 [269.875, 359.875]]),
                                circular=True)


def _cmor_time(mip, bounds=None, shifted=False):
    """Provide expected time coordinate after fixes."""
    if mip is 'E1hr':
        if not shifted:
            timestamps = [51134.0, 51134.0416667, 51134.0833333]
            if bounds is not None:
                bounds = np.array([[51133.9791667, 51134.0208333],
                                   [51134.0208333, 51134.0625],
                                   [51134.0625, 51134.1041667]])
        else:
            timestamps = [51133.97916667, 51134.02083333, 51134.0625]
            if bounds is not None:
                bounds = np.array([[51133.95833333, 51134.0],
                                   [51134.0, 51134.04166667],
                                   [51134.04166667, 51134.08333333]])
    elif mip is 'Amon':
        timestamps = np.array([51149.5, 51179.0, 51208.5])
        if bounds is not None:
            bounds = np.array([[51134.0, 51165.0], [51165.0, 51193.0],
                               [51193.0, 51224.0]])

    return iris.coords.DimCoord(np.array(timestamps, dtype=float),
                                standard_name='time',
                                long_name='time',
                                var_name='time',
                                units=Unit('days since 1850-1-1 00:00:00',
                                           calendar='gregorian'),
                                bounds=bounds)


def _cmor_aux_height(value):
    return iris.coords.AuxCoord(value,
                                long_name="height",
                                standard_name="height",
                                units=Unit('m'),
                                var_name="height",
                                attributes={'positive': 'up'})


def _cmor_data(mip):
    if mip is 'fx':
        return np.arange(9).reshape(3, 3)[::-1, :]
    return np.arange(27).reshape(3, 3, 3)[:, ::-1, :]


def pr_era5_monthly():
    time = _era5_time('monthly')
    cube = iris.cube.Cube(
        _era5_data('monthly'),
        long_name='total_precipitation',
        var_name='tp',
        units='m',
        dim_coords_and_dims=[(time, 0), (_era5_latitude(), 1),
                             (_era5_longitude(), 2)],
    )
    return iris.cube.CubeList([cube])


def pr_cmor_amon():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('Amon', 'pr')
    time = _cmor_time('Amon', bounds=True)
    data = _cmor_data('Amon') * 1000.
    cube = iris.cube.Cube(data.astype('float32'),
                          long_name=vardef.long_name,
                          var_name=vardef.short_name,
                          standard_name=vardef.standard_name,
                          units=Unit(vardef.units),
                          dim_coords_and_dims=[(time, 0),
                                               (_cmor_latitude(), 1),
                                               (_cmor_longitude(), 2)])
    return iris.cube.CubeList([cube])


def pr_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='total_precipitation',
        var_name='tp',
        units='m',
        dim_coords_and_dims=[(time, 0), (_era5_latitude(), 1),
                             (_era5_longitude(), 2)],
    )
    return iris.cube.CubeList([cube])


def pr_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'pr')
    time = _cmor_time('E1hr', bounds=True, shifted=True)
    data = _cmor_data('E1hr') * 1000.
    cube = iris.cube.Cube(data.astype('float32'),
                          long_name=vardef.long_name,
                          var_name=vardef.short_name,
                          standard_name=vardef.standard_name,
                          units=Unit(vardef.units),
                          dim_coords_and_dims=[(time, 0),
                                               (_cmor_latitude(), 1),
                                               (_cmor_longitude(), 2)])
    return iris.cube.CubeList([cube])


def orog_era5_hourly():
    time = _era5_time('invariant')
    cube = iris.cube.Cube(
        _era5_data('invariant'),
        long_name='geopotential height',
        var_name='zg',
        units='m**2 s**-2',
        dim_coords_and_dims=[(time, 0), (_era5_latitude(), 1),
                             (_era5_longitude(), 2)],
    )
    return iris.cube.CubeList([cube])


def orog_cmor_fx():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('fx', 'orog')
    data = _cmor_data('fx') / 9.80665
    cube = iris.cube.Cube(data.astype('float32'),
                          long_name=vardef.long_name,
                          var_name=vardef.short_name,
                          standard_name=vardef.standard_name,
                          units=Unit(vardef.units),
                          dim_coords_and_dims=[(_cmor_latitude(), 0),
                                               (_cmor_longitude(), 1)])
    return iris.cube.CubeList([cube])


def uas_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='10m_u_component_of_wind',
        var_name='u10',
        units='m s-1',
        dim_coords_and_dims=[(time, 0), (_era5_latitude(), 1),
                             (_era5_longitude(), 2)],
    )
    return iris.cube.CubeList([cube])


def uas_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'uas')
    time = _cmor_time('E1hr')
    data = _cmor_data('E1hr')
    cube = iris.cube.Cube(data.astype('float32'),
                          long_name=vardef.long_name,
                          var_name=vardef.short_name,
                          standard_name=vardef.standard_name,
                          units=Unit(vardef.units),
                          dim_coords_and_dims=[(time, 0),
                                               (_cmor_latitude(), 1),
                                               (_cmor_longitude(), 2)])
    cube.add_aux_coord(_cmor_aux_height(10.))
    return iris.cube.CubeList([cube])


def tas_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='2m_temperature',
        var_name='t2m',
        units='degC',
        dim_coords_and_dims=[(time, 0), (_era5_latitude(), 1),
                             (_era5_longitude(), 2)],
    )
    return iris.cube.CubeList([cube])


def tas_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'tas')
    time = _cmor_time('E1hr')
    data = _cmor_data('E1hr') + 273.15
    cube = iris.cube.Cube(data.astype('float32'),
                          long_name=vardef.long_name,
                          var_name=vardef.short_name,
                          standard_name=vardef.standard_name,
                          units=Unit(vardef.units),
                          dim_coords_and_dims=[(time, 0),
                                               (_cmor_latitude(), 1),
                                               (_cmor_longitude(), 2)])
    cube.add_aux_coord(_cmor_aux_height(2.))
    return iris.cube.CubeList([cube])


def rsds_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='solar_radiation_downwards',
        var_name='rlwd',
        units='J m**-2',
        dim_coords_and_dims=[(time, 0), (_era5_latitude(), 1),
                             (_era5_longitude(), 2)],
    )
    return iris.cube.CubeList([cube])


def rsds_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'rsds')
    time = _cmor_time('E1hr', shifted=True, bounds=True)
    data = _cmor_data('E1hr') / 3600
    cube = iris.cube.Cube(data.astype('float32'),
                          long_name=vardef.long_name,
                          var_name=vardef.short_name,
                          standard_name=vardef.standard_name,
                          units=Unit(vardef.units),
                          dim_coords_and_dims=[(time, 0),
                                               (_cmor_latitude(), 1),
                                               (_cmor_longitude(), 2)],
                          attributes={'positive': 'down'})
    return iris.cube.CubeList([cube])


def prsn_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='snow',
        var_name='snow',
        units='unknown',
        dim_coords_and_dims=[(time, 0), (_era5_latitude(), 1),
                             (_era5_longitude(), 2)],
    )
    return iris.cube.CubeList([cube])


def prsn_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'prsn')
    time = _cmor_time('E1hr', shifted=True, bounds=True)
    data = _cmor_data('E1hr') * 1000
    cube = iris.cube.Cube(data.astype('float32'),
                          long_name=vardef.long_name,
                          var_name=vardef.short_name,
                          standard_name=vardef.standard_name,
                          units=Unit(vardef.units),
                          dim_coords_and_dims=[(time, 0),
                                               (_cmor_latitude(), 1),
                                               (_cmor_longitude(), 2)])
    return iris.cube.CubeList([cube])


def mrro_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='runoff',
        var_name='runoff',
        units='m',
        dim_coords_and_dims=[(time, 0), (_era5_latitude(), 1),
                             (_era5_longitude(), 2)],
    )
    return iris.cube.CubeList([cube])


def mrro_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'mrro')
    time = _cmor_time('E1hr', shifted=True, bounds=True)
    data = _cmor_data('E1hr') * 1000
    cube = iris.cube.Cube(data.astype('float32'),
                          long_name=vardef.long_name,
                          var_name=vardef.short_name,
                          standard_name=vardef.standard_name,
                          units=Unit(vardef.units),
                          dim_coords_and_dims=[(time, 0),
                                               (_cmor_latitude(), 1),
                                               (_cmor_longitude(), 2)])
    return iris.cube.CubeList([cube])


def rls_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='runoff',
        var_name='runoff',
        units='W m**-2',
        dim_coords_and_dims=[(time, 0), (_era5_latitude(), 1),
                             (_era5_longitude(), 2)],
    )
    return iris.cube.CubeList([cube])


def rls_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'rls')
    time = _cmor_time('E1hr', shifted=False, bounds=True)
    data = _cmor_data('E1hr')
    cube = iris.cube.Cube(data.astype('float32'),
                          long_name=vardef.long_name,
                          var_name=vardef.short_name,
                          standard_name=vardef.standard_name,
                          units=Unit(vardef.units),
                          dim_coords_and_dims=[(time, 0),
                                               (_cmor_latitude(), 1),
                                               (_cmor_longitude(), 2)],
                          attributes={'positive': 'down'})
    return iris.cube.CubeList([cube])

VARIABLES = [
    pytest.param(a, b, c, d, id=c + '_' + d) for (a, b, c, d) in [
        (pr_era5_monthly(), pr_cmor_amon(), 'pr', 'Amon'),
        (pr_era5_hourly(), pr_cmor_e1hr(), 'pr', 'E1hr'),
        (orog_era5_hourly(), orog_cmor_fx(), 'orog', 'fx'),
        (uas_era5_hourly(), uas_cmor_e1hr(), 'uas', 'E1hr'),
        (tas_era5_hourly(), tas_cmor_e1hr(), 'tas', 'E1hr'),
        (rsds_era5_hourly(), rsds_cmor_e1hr(), 'rsds', 'E1hr'),
        (prsn_era5_hourly(), prsn_cmor_e1hr(), 'prsn', 'E1hr'),
        (mrro_era5_hourly(), mrro_cmor_e1hr(), 'mrro', 'E1hr'),
        (rls_era5_hourly(), rls_cmor_e1hr(), 'rls', 'E1hr'),
    ]
]


@pytest.mark.parametrize('era5_cubes, cmor_cubes, var, mip', VARIABLES)
def test_cmorization(era5_cubes, cmor_cubes, var, mip):
    """Verify that cmorization results in the expected target cube."""

    fixed_cubes = fix_metadata(era5_cubes, var, 'native6', 'era5', mip)
    print('era5_cube:', era5_cubes[0].xml())
    print('cmor_cube:', cmor_cubes[0].xml())
    print('fixed_cube:', fixed_cubes[0].xml())
    assert fixed_cubes[0].xml() == cmor_cubes[0].xml()
    assert (fixed_cubes[0].data == cmor_cubes[0].data).all()
    # assert fixed_cubes[0].coords() == cmor_cubes[0].coords()
    # assert fixed_cubes[0] == cmor_cubes[0]
