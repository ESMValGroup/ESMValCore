"""Tests for the fixes of ERA5."""
import datetime

import iris
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.native6.era5 import (
    AllVars,
    Evspsbl,
    Zg,
    get_frequency,
)
from esmvalcore.cmor.fix import Fix, fix_metadata
from esmvalcore.cmor.table import CMOR_TABLES

COMMENT = ('Contains modified Copernicus Climate Change Service Information '
           f'{datetime.datetime.now().year}')


def test_get_evspsbl_fix():
    """Test whether the right fixes are gathered for a single variable."""
    fix = Fix.get_fixes('native6', 'ERA5', 'E1hr', 'evspsbl')
    assert fix == [Evspsbl(None), AllVars(None)]


def test_get_zg_fix():
    """Test whether the right fix gets found again, for zg as well."""
    fix = Fix.get_fixes('native6', 'ERA5', 'Amon', 'zg')
    assert fix == [Zg(None), AllVars(None)]


def test_get_frequency_hourly():
    """Test cubes with hourly frequency."""
    time = iris.coords.DimCoord(
        [0, 1, 2],
        standard_name='time',
        units=Unit('hours since 1900-01-01'),
    )
    cube = iris.cube.Cube(
        [1, 6, 3],
        var_name='random_var',
        dim_coords_and_dims=[(time, 0)],
    )
    assert get_frequency(cube) == 'hourly'
    cube.coord('time').convert_units('days since 1850-1-1 00:00:00.0')
    assert get_frequency(cube) == 'hourly'


def test_get_frequency_monthly():
    """Test cubes with monthly frequency."""
    time = iris.coords.DimCoord(
        [0, 31, 59],
        standard_name='time',
        units=Unit('hours since 1900-01-01'),
    )
    cube = iris.cube.Cube(
        [1, 6, 3],
        var_name='random_var',
        dim_coords_and_dims=[(time, 0)],
    )
    assert get_frequency(cube) == 'monthly'
    cube.coord('time').convert_units('days since 1850-1-1 00:00:00.0')
    assert get_frequency(cube) == 'monthly'


def test_get_frequency_fx():
    """Test cubes with time invariant frequency."""
    cube = iris.cube.Cube(1., long_name='Cube without time coordinate')
    assert get_frequency(cube) == 'fx'
    time = iris.coords.DimCoord(
        0,
        standard_name='time',
        units=Unit('hours since 1900-01-01'),
    )
    cube = iris.cube.Cube(
        [1],
        var_name='cube_with_length_1_time_coord',
        long_name='Geopotential',
        dim_coords_and_dims=[(time, 0)],
    )
    assert get_frequency(cube) == 'fx'
    cube.long_name = 'Not geopotential'
    with pytest.raises(ValueError):
        get_frequency(cube)


def _era5_latitude():
    return iris.coords.DimCoord(
        np.array([90., 0., -90.]),
        standard_name='latitude',
        long_name='latitude',
        var_name='latitude',
        units=Unit('degrees'),
    )


def _era5_longitude():
    return iris.coords.DimCoord(
        np.array([0, 180, 359.75]),
        standard_name='longitude',
        long_name='longitude',
        var_name='longitude',
        units=Unit('degrees'),
        circular=True,
    )


def _era5_time(frequency):
    if frequency == 'invariant':
        timestamps = [788928]  # hours since 1900 at 1 january 1990
    elif frequency == 'hourly':
        timestamps = [788928, 788929, 788930]
    elif frequency == 'monthly':
        timestamps = [788928, 789672, 790344]
    return iris.coords.DimCoord(
        np.array(timestamps, dtype='int32'),
        standard_name='time',
        long_name='time',
        var_name='time',
        units=Unit('hours since 1900-01-01'
                   '00:00:00.0', calendar='gregorian'),
    )


def _era5_plev():
    values = np.array([
        1,
        1000,
    ])
    return iris.coords.DimCoord(
        values,
        long_name="pressure",
        units=Unit("millibars"),
        var_name="level",
        attributes={'positive': 'down'},
    )


def _era5_data(frequency):
    if frequency == 'invariant':
        return np.arange(9).reshape(1, 3, 3)
    return np.arange(27).reshape(3, 3, 3)


def _cmor_latitude():
    return iris.coords.DimCoord(
        np.array([-90., 0., 90.]),
        standard_name='latitude',
        long_name='Latitude',
        var_name='lat',
        units=Unit('degrees_north'),
        bounds=np.array([[-90., -45.], [-45., 45.], [45., 90.]]),
    )


def _cmor_longitude():
    return iris.coords.DimCoord(
        np.array([0, 180, 359.75]),
        standard_name='longitude',
        long_name='Longitude',
        var_name='lon',
        units=Unit('degrees_east'),
        bounds=np.array([[-0.125, 90.], [90., 269.875], [269.875, 359.875]]),
        circular=True,
    )


def _cmor_time(mip, bounds=None, shifted=False):
    """Provide expected time coordinate after fixes."""
    if mip == 'E1hr':
        offset = 51134  # days since 1850 at 1 january 1990
        timestamps = offset + np.arange(3) / 24
        if shifted:
            timestamps -= 1 / 48
        if bounds is not None:
            bounds = [[t - 1 / 48, t + 1 / 48] for t in timestamps]
    elif mip == 'Amon':
        timestamps = np.array([51149.5, 51179., 51208.5])
        if bounds is not None:
            bounds = np.array([[51134., 51165.], [51165., 51193.],
                               [51193., 51224.]])

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


def _cmor_plev():
    values = np.array([
        100000.0,
        100.0,
    ])
    return iris.coords.DimCoord(values,
                                long_name="pressure",
                                standard_name="air_pressure",
                                units=Unit("Pa"),
                                var_name="plev",
                                attributes={'positive': 'down'})


def _cmor_data(mip):
    if mip == 'fx':
        return np.arange(9).reshape(3, 3)[::-1, :]
    return np.arange(27).reshape(3, 3, 3)[:, ::-1, :]


def clt_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='cloud cover fraction',
        var_name='cloud_cover',
        units='unknown',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def clt_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'clt')
    time = _cmor_time('E1hr', bounds=True)
    data = _cmor_data('E1hr') * 100
    cube = iris.cube.Cube(
        data.astype('float32'),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_latitude(), 1),
            (_cmor_longitude(), 2),
        ],
        attributes={'comment': COMMENT},
    )
    return iris.cube.CubeList([cube])


def evspsbl_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='total evapotranspiration',
        var_name='e',
        units='unknown',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def evspsbl_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'evspsbl')
    time = _cmor_time('E1hr', shifted=True, bounds=True)
    data = _cmor_data('E1hr') * 1000 / 3600.
    cube = iris.cube.Cube(
        data.astype('float32'),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_latitude(), 1),
            (_cmor_longitude(), 2),
        ],
        attributes={'comment': COMMENT},
    )
    return iris.cube.CubeList([cube])


def evspsblpot_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='potential evapotranspiration',
        var_name='epot',
        units='unknown',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def evspsblpot_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'evspsblpot')
    time = _cmor_time('E1hr', shifted=True, bounds=True)
    data = _cmor_data('E1hr') * 1000 / 3600.
    cube = iris.cube.Cube(
        data.astype('float32'),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_latitude(), 1),
            (_cmor_longitude(), 2),
        ],
        attributes={'comment': COMMENT},
    )
    return iris.cube.CubeList([cube])


def mrro_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='runoff',
        var_name='runoff',
        units='m',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def mrro_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'mrro')
    time = _cmor_time('E1hr', shifted=True, bounds=True)
    data = _cmor_data('E1hr') * 1000 / 3600.
    cube = iris.cube.Cube(
        data.astype('float32'),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_latitude(), 1),
            (_cmor_longitude(), 2),
        ],
        attributes={'comment': COMMENT},
    )
    return iris.cube.CubeList([cube])


def orog_era5_hourly():
    time = _era5_time('invariant')
    cube = iris.cube.Cube(
        _era5_data('invariant'),
        long_name='geopotential height',
        var_name='zg',
        units='m**2 s**-2',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def orog_cmor_fx():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('fx', 'orog')
    data = _cmor_data('fx') / 9.80665
    cube = iris.cube.Cube(
        data.astype('float32'),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[(_cmor_latitude(), 0), (_cmor_longitude(), 1)],
        attributes={'comment': COMMENT},
    )
    return iris.cube.CubeList([cube])


def pr_era5_monthly():
    time = _era5_time('monthly')
    cube = iris.cube.Cube(
        _era5_data('monthly'),
        long_name='total_precipitation',
        var_name='tp',
        units='m',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def pr_cmor_amon():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('Amon', 'pr')
    time = _cmor_time('Amon', bounds=True)
    data = _cmor_data('Amon') * 1000. / 3600. / 24.
    cube = iris.cube.Cube(
        data.astype('float32'),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_latitude(), 1),
            (_cmor_longitude(), 2),
        ],
        attributes={'comment': COMMENT},
    )
    return iris.cube.CubeList([cube])


def pr_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='total_precipitation',
        var_name='tp',
        units='m',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def pr_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'pr')
    time = _cmor_time('E1hr', bounds=True, shifted=True)
    data = _cmor_data('E1hr') * 1000. / 3600.
    cube = iris.cube.Cube(
        data.astype('float32'),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_latitude(), 1),
            (_cmor_longitude(), 2),
        ],
        attributes={'comment': COMMENT},
    )
    return iris.cube.CubeList([cube])


def prsn_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='snow',
        var_name='snow',
        units='unknown',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def prsn_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'prsn')
    time = _cmor_time('E1hr', shifted=True, bounds=True)
    data = _cmor_data('E1hr') * 1000 / 3600.
    cube = iris.cube.Cube(
        data.astype('float32'),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_latitude(), 1),
            (_cmor_longitude(), 2),
        ],
        attributes={'comment': COMMENT},
    )
    return iris.cube.CubeList([cube])


def ptype_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='snow',
        var_name='snow',
        units='unknown',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def ptype_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'ptype')
    time = _cmor_time('E1hr', shifted=False, bounds=True)
    data = _cmor_data('E1hr')
    cube = iris.cube.Cube(
        data.astype('float32'),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        units=1,
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_latitude(), 1),
            (_cmor_longitude(), 2),
        ],
        attributes={'comment': COMMENT},
    )
    cube.coord('latitude').long_name = 'latitude'
    cube.coord('longitude').long_name = 'longitude'
    return iris.cube.CubeList([cube])


def rlds_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='surface thermal radiation downwards',
        var_name='ssrd',
        units='J m**-2',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def rlds_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'rlds')
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
                          attributes={
                              'comment': COMMENT,
                              'positive': 'down',
                          })
    return iris.cube.CubeList([cube])


def rls_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='runoff',
        var_name='runoff',
        units='W m-2',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def rls_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'rls')
    time = _cmor_time('E1hr', shifted=True, bounds=True)
    data = _cmor_data('E1hr')
    cube = iris.cube.Cube(data.astype('float32'),
                          long_name=vardef.long_name,
                          var_name=vardef.short_name,
                          standard_name=vardef.standard_name,
                          units=Unit(vardef.units),
                          dim_coords_and_dims=[
                              (time, 0),
                              (_cmor_latitude(), 1),
                              (_cmor_longitude(), 2),
                          ],
                          attributes={
                              'comment': COMMENT,
                              'positive': 'down',
                          })
    return iris.cube.CubeList([cube])


def rsds_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='solar_radiation_downwards',
        var_name='rlwd',
        units='J m**-2',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
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
                          attributes={
                              'comment': COMMENT,
                              'positive': 'down',
                          })
    return iris.cube.CubeList([cube])


def rsdt_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='thermal_radiation_downwards',
        var_name='strd',
        units='J m**-2',
        dim_coords_and_dims=[(time, 0), (_era5_latitude(), 1),
                             (_era5_longitude(), 2)],
    )
    return iris.cube.CubeList([cube])


def rsdt_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'rsdt')
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
                          attributes={
                              'comment': COMMENT,
                              'positive': 'down',
                          })
    return iris.cube.CubeList([cube])


def rss_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='net_solar_radiation',
        var_name='ssr',
        units='J m**-2',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def rss_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'rss')
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
                          attributes={
                              'comment': COMMENT,
                              'positive': 'down',
                          })
    return iris.cube.CubeList([cube])


def tas_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='2m_temperature',
        var_name='t2m',
        units='K',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def tas_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'tas')
    time = _cmor_time('E1hr')
    data = _cmor_data('E1hr')
    cube = iris.cube.Cube(data.astype('float32'),
                          long_name=vardef.long_name,
                          var_name=vardef.short_name,
                          standard_name=vardef.standard_name,
                          units=Unit(vardef.units),
                          dim_coords_and_dims=[(time, 0),
                                               (_cmor_latitude(), 1),
                                               (_cmor_longitude(), 2)],
                          attributes={'comment': COMMENT})
    cube.add_aux_coord(_cmor_aux_height(2.))
    return iris.cube.CubeList([cube])


def tas_era5_monthly():
    time = _era5_time('monthly')
    cube = iris.cube.Cube(
        _era5_data('monthly'),
        long_name='2m_temperature',
        var_name='t2m',
        units='K',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def tas_cmor_amon():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('Amon', 'tas')
    time = _cmor_time('Amon', bounds=True)
    data = _cmor_data('Amon')
    cube = iris.cube.Cube(
        data.astype('float32'),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_latitude(), 1),
            (_cmor_longitude(), 2),
        ],
        attributes={'comment': COMMENT},
    )
    cube.add_aux_coord(_cmor_aux_height(2.))
    return iris.cube.CubeList([cube])


def zg_era5_monthly():
    time = _era5_time('monthly')
    data = np.ones((3, 2, 3, 3))
    cube = iris.cube.Cube(
        data,
        long_name='geopotential height',
        var_name='zg',
        units='m**2 s**-2',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_plev(), 1),
            (_era5_latitude(), 2),
            (_era5_longitude(), 3),
        ],
    )
    return iris.cube.CubeList([cube])


def zg_cmor_amon():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('Amon', 'zg')
    time = _cmor_time('Amon', bounds=True)
    data = np.ones((3, 2, 3, 3))
    data = data / 9.80665
    cube = iris.cube.Cube(
        data.astype('float32'),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_plev(), 1),
            (_cmor_latitude(), 2),
            (_cmor_longitude(), 3),
        ],
        attributes={'comment': COMMENT},
    )
    return iris.cube.CubeList([cube])


def tasmax_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='maximum 2m temperature',
        var_name='mx2t',
        units='K',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def tasmax_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'tasmax')
    time = _cmor_time('E1hr', shifted=True, bounds=True)
    data = _cmor_data('E1hr')
    cube = iris.cube.Cube(
        data.astype('float32'),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_latitude(), 1),
            (_cmor_longitude(), 2),
        ],
        attributes={'comment': COMMENT},
    )
    cube.add_aux_coord(_cmor_aux_height(2.))
    return iris.cube.CubeList([cube])


def tasmin_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='minimum 2m temperature',
        var_name='mn2t',
        units='K',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def tasmin_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'tasmin')
    time = _cmor_time('E1hr', shifted=True, bounds=True)
    data = _cmor_data('E1hr')
    cube = iris.cube.Cube(
        data.astype('float32'),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_latitude(), 1),
            (_cmor_longitude(), 2),
        ],
        attributes={'comment': COMMENT},
    )
    cube.add_aux_coord(_cmor_aux_height(2.))
    return iris.cube.CubeList([cube])


def uas_era5_hourly():
    time = _era5_time('hourly')
    cube = iris.cube.Cube(
        _era5_data('hourly'),
        long_name='10m_u_component_of_wind',
        var_name='u10',
        units='m s-1',
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return iris.cube.CubeList([cube])


def uas_cmor_e1hr():
    cmor_table = CMOR_TABLES['native6']
    vardef = cmor_table.get_variable('E1hr', 'uas')
    time = _cmor_time('E1hr')
    data = _cmor_data('E1hr')
    cube = iris.cube.Cube(
        data.astype('float32'),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_latitude(), 1),
            (_cmor_longitude(), 2),
        ],
        attributes={'comment': COMMENT},
    )
    cube.add_aux_coord(_cmor_aux_height(10.))
    return iris.cube.CubeList([cube])


VARIABLES = [
    pytest.param(a, b, c, d, id=c + '_' + d) for (a, b, c, d) in [
        (clt_era5_hourly(), clt_cmor_e1hr(), 'clt', 'E1hr'),
        (evspsbl_era5_hourly(), evspsbl_cmor_e1hr(), 'evspsbl', 'E1hr'),
        (evspsblpot_era5_hourly(), evspsblpot_cmor_e1hr(), 'evspsblpot',
         'E1hr'),
        (mrro_era5_hourly(), mrro_cmor_e1hr(), 'mrro', 'E1hr'),
        (orog_era5_hourly(), orog_cmor_fx(), 'orog', 'fx'),
        (pr_era5_monthly(), pr_cmor_amon(), 'pr', 'Amon'),
        (pr_era5_hourly(), pr_cmor_e1hr(), 'pr', 'E1hr'),
        (prsn_era5_hourly(), prsn_cmor_e1hr(), 'prsn', 'E1hr'),
        (ptype_era5_hourly(), ptype_cmor_e1hr(), 'ptype', 'E1hr'),
        (rlds_era5_hourly(), rlds_cmor_e1hr(), 'rlds', 'E1hr'),
        (rls_era5_hourly(), rls_cmor_e1hr(), 'rls', 'E1hr'),
        (rsds_era5_hourly(), rsds_cmor_e1hr(), 'rsds', 'E1hr'),
        (rsdt_era5_hourly(), rsdt_cmor_e1hr(), 'rsdt', 'E1hr'),
        (rss_era5_hourly(), rss_cmor_e1hr(), 'rss', 'E1hr'),
        (tas_era5_hourly(), tas_cmor_e1hr(), 'tas', 'E1hr'),
        (tas_era5_monthly(), tas_cmor_amon(), 'tas', 'Amon'),
        (tasmax_era5_hourly(), tasmax_cmor_e1hr(), 'tasmax', 'E1hr'),
        (tasmin_era5_hourly(), tasmin_cmor_e1hr(), 'tasmin', 'E1hr'),
        (uas_era5_hourly(), uas_cmor_e1hr(), 'uas', 'E1hr'),
        (zg_era5_monthly(), zg_cmor_amon(), 'zg', 'Amon'),
    ]
]


@pytest.mark.parametrize('era5_cubes, cmor_cubes, var, mip', VARIABLES)
def test_cmorization(era5_cubes, cmor_cubes, var, mip):
    """Verify that cmorization results in the expected target cube."""
    fixed_cubes = fix_metadata(era5_cubes, var, 'native6', 'era5', mip)
    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes[0]
    cmor_cube = cmor_cubes[0]
    if fixed_cube.coords('time'):
        for cube in [fixed_cube, cmor_cube]:
            coord = cube.coord('time')
            coord.points = np.round(coord.points, decimals=7)
            if coord.bounds is not None:
                coord.bounds = np.round(coord.bounds, decimals=7)
    print('cmor_cube:', cmor_cube)
    print('fixed_cube:', fixed_cube)
    assert fixed_cube == cmor_cube
