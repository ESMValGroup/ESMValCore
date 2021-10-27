"""Unit tests for the :func:`esmvalcore.preprocessor._time` module."""

import copy
import unittest
from datetime import datetime
from typing import List, Tuple

import iris
import iris.coord_categorisation
import iris.coords
import iris.exceptions
import iris.fileformats
import numpy as np
import pytest
from cf_units import Unit
from iris.cube import Cube
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_raises,
)

import tests
from esmvalcore.iris_helpers import date2num
from esmvalcore.preprocessor._time import (
    annual_statistics,
    anomalies,
    climate_statistics,
    clip_start_end_year,
    daily_statistics,
    decadal_statistics,
    extract_month,
    extract_season,
    extract_time,
    get_time_weights,
    hourly_statistics,
    monthly_statistics,
    regrid_time,
    resample_hours,
    resample_time,
    seasonal_statistics,
    timeseries_filter,
)


def _create_sample_cube(calendar='gregorian'):
    """Create sample cube."""
    cube = Cube(np.arange(1, 25), var_name='co2', units='J')
    cube.add_dim_coord(
        iris.coords.DimCoord(
            np.arange(15., 720., 30.),
            standard_name='time',
            units=Unit('days since 1950-01-01 00:00:00', calendar=calendar),
        ),
        0,
    )
    return cube


def add_auxiliary_coordinate(cubelist):
    """Add AuxCoords to cubes in cubelist."""
    for cube in cubelist:
        iris.coord_categorisation.add_day_of_month(cube, cube.coord('time'))
        iris.coord_categorisation.add_day_of_year(cube, cube.coord('time'))


class TestExtractMonth(tests.Test):
    """Tests for extract_month."""
    def setUp(self):
        """Prepare tests."""
        self.cube = _create_sample_cube()

    def test_get_january(self):
        """Test january extraction."""
        sliced = extract_month(self.cube, 1)
        assert_array_equal(np.array([1, 1]),
                           sliced.coord('month_number').points)

    def test_raises_if_extracted_cube_is_none(self):
        """Test function for winter."""
        sliced = extract_month(self.cube, 1)
        with assert_raises(ValueError):
            extract_month(sliced, 2)

    def test_get_january_with_existing_coord(self):
        """Test january extraction."""
        iris.coord_categorisation.add_month_number(self.cube, 'time')
        sliced = extract_month(self.cube, 1)
        assert_array_equal(np.array([1, 1]),
                           sliced.coord('month_number').points)

    def test_bad_month_raises(self):
        """Test january extraction."""
        with self.assertRaises(ValueError):
            extract_month(self.cube, 13)
        with self.assertRaises(ValueError):
            extract_month(self.cube, -1)


class TestTimeSlice(tests.Test):
    """Tests for extract_time."""
    def setUp(self):
        """Prepare tests."""
        self.cube = _create_sample_cube()

    def test_raises_if_extracted_cube_is_none(self):
        """Test extract_time."""
        with assert_raises(ValueError):
            extract_time(self.cube, 2000, 1, 1, 2000, 12, 31)

    def test_extract_time(self):
        """Test extract_time."""
        sliced = extract_time(self.cube, 1950, 1, 1, 1950, 12, 31)
        iris.coord_categorisation.add_month_number(sliced, 'time')
        assert_array_equal(np.arange(1, 13, 1),
                           sliced.coord('month_number').points)

    def test_extract_time_limit(self):
        """Test extract time when limits are included."""
        cube = Cube(np.arange(0, 720), var_name='co2', units='J')
        cube.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(0., 720., 1.),
                standard_name='time',
                units=Unit('days since 1950-01-01 00:00:00',
                           calendar='360_day'),
            ),
            0,
        )
        sliced = extract_time(cube, 1950, 1, 1, 1951, 1, 1)
        assert_array_equal(np.arange(0, 360), sliced.coord('time').points)

    def test_extract_time_non_gregorian_day(self):
        """Test extract time when the day is not in the Gregorian calendar."""
        cube = Cube(np.arange(0, 720), var_name='co2', units='J')
        cube.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(0., 720., 1.),
                standard_name='time',
                units=Unit('days since 1950-01-01 00:00:00',
                           calendar='360_day'),
            ),
            0,
        )
        sliced = extract_time(cube, 1950, 2, 30, 1950, 3, 1)
        assert_array_equal(np.array([59]), sliced.coord('time').points)

    def test_extract_time_no_slice(self):
        """Test fail of extract_time."""
        with self.assertRaises(ValueError) as ctx:
            extract_time(self.cube, 2200, 1, 1, 2200, 12, 31)
        msg = ("Time slice 2200-01-01 to 2200-12-31 is outside"
               " cube time bounds 1950-01-16 00:00:00 to 1951-12-07 00:00:00.")
        assert ctx.exception.args == (msg, )

    def test_extract_time_one_time(self):
        """Test extract_time with one time step."""
        cube = _create_sample_cube()
        cube.coord('time').guess_bounds()
        cube = cube.collapsed('time', iris.analysis.MEAN)
        sliced = extract_time(cube, 1950, 1, 1, 1952, 12, 31)
        assert_array_equal(np.array([360.]), sliced.coord('time').points)

    def test_extract_time_no_time(self):
        """Test extract_time with no time step."""
        cube = _create_sample_cube()[0]
        sliced = extract_time(cube, 1950, 1, 1, 1950, 12, 31)
        assert cube == sliced


class TestClipStartEndYear(tests.Test):
    """Tests for clip_start_end_year."""
    def setUp(self):
        """Prepare tests."""
        self.cube = _create_sample_cube()

    def test_clip_start_end_year_1_year(self):
        """Test clip_start_end_year with 1 year."""
        sliced = clip_start_end_year(self.cube, 1950, 1950)
        iris.coord_categorisation.add_month_number(sliced, 'time')
        iris.coord_categorisation.add_year(sliced, 'time')
        assert_array_equal(np.arange(1, 13, 1),
                           sliced.coord('month_number').points)
        assert_array_equal(np.full(12, 1950), sliced.coord('year').points)

    def test_clip_start_end_year_3_years(self):
        """Test clip_start_end_year with 3 years."""
        sliced = clip_start_end_year(self.cube, 1949, 1951)
        assert sliced == self.cube

    def test_clip_start_end_year_no_slice(self):
        """Test fail of clip_start_end_year."""
        with self.assertRaises(ValueError) as ctx:
            clip_start_end_year(self.cube, 2200, 2200)
        msg = ("Time slice 2200-01-01 to 2201-01-01 is outside"
               " cube time bounds 1950-01-16 00:00:00 to 1951-12-07 00:00:00.")
        assert ctx.exception.args == (msg, )

    def test_clip_start_end_year_one_time(self):
        """Test clip_start_end_year with one time step."""
        cube = _create_sample_cube()
        cube.coord('time').guess_bounds()
        cube = cube.collapsed('time', iris.analysis.MEAN)
        sliced = clip_start_end_year(cube, 1950, 1952)
        assert_array_equal(np.array([360.]), sliced.coord('time').points)

    def test_clip_start_end_year_no_time(self):
        """Test clip_start_end_year with no time step."""
        cube = _create_sample_cube()[0]
        sliced = clip_start_end_year(cube, 1950, 1950)
        assert cube == sliced


class TestExtractSeason(tests.Test):
    """Tests for extract_season."""
    def setUp(self):
        """Prepare tests."""
        self.cube = _create_sample_cube()

    def test_get_djf(self):
        """Test function for winter."""
        sliced = extract_season(self.cube, 'DJF')
        iris.coord_categorisation.add_month_number(sliced, 'time')
        assert_array_equal(np.array([1, 2, 12, 1, 2, 12]),
                           sliced.coord('month_number').points)
        with assert_raises(iris.exceptions.CoordinateNotFoundError):
            self.cube.coord('clim_season')
        with assert_raises(iris.exceptions.CoordinateNotFoundError):
            self.cube.coord('season_year')

    def test_raises_if_extracted_cube_is_none(self):
        """Test function for winter."""
        sliced = extract_season(self.cube, 'DJF')
        with assert_raises(ValueError):
            extract_season(sliced, 'MAM')

    def test_get_djf_caps(self):
        """Test function works when season specified in caps."""
        sliced = extract_season(self.cube, 'DJF')
        iris.coord_categorisation.add_month_number(sliced, 'time')
        assert_array_equal(np.array([1, 2, 12, 1, 2, 12]),
                           sliced.coord('month_number').points)
        with assert_raises(iris.exceptions.CoordinateNotFoundError):
            self.cube.coord('clim_season')
        with assert_raises(iris.exceptions.CoordinateNotFoundError):
            self.cube.coord('season_year')

    def test_get_mam(self):
        """Test function for spring."""
        sliced = extract_season(self.cube, 'MAM')
        iris.coord_categorisation.add_month_number(sliced, 'time')
        assert_array_equal(np.array([3, 4, 5, 3, 4, 5]),
                           sliced.coord('month_number').points)
        with assert_raises(iris.exceptions.CoordinateNotFoundError):
            self.cube.coord('clim_season')
        with assert_raises(iris.exceptions.CoordinateNotFoundError):
            self.cube.coord('season_year')

    def test_get_jja(self):
        """Test function for summer."""
        sliced = extract_season(self.cube, 'JJA')
        iris.coord_categorisation.add_month_number(sliced, 'time')
        assert_array_equal(np.array([6, 7, 8, 6, 7, 8]),
                           sliced.coord('month_number').points)

    def test_get_multiple_seasons(self):
        """Test function for two seasons."""
        sliced = [extract_season(self.cube, seas) for seas in ["JJA", "SON"]]
        clim_coords = [sin_sli.coord("clim_season") for sin_sli in sliced]
        assert_array_equal(clim_coords[0].points,
                           ['JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA'])
        assert_array_equal(clim_coords[1].points,
                           ['SON', 'SON', 'SON', 'SON', 'SON', 'SON'])

    def test_get_son(self):
        """Test function for summer."""
        sliced = extract_season(self.cube, 'SON')
        iris.coord_categorisation.add_month_number(sliced, 'time')
        assert_array_equal(np.array([9, 10, 11, 9, 10, 11]),
                           sliced.coord('month_number').points)
        with assert_raises(iris.exceptions.CoordinateNotFoundError):
            self.cube.coord('clim_season')
        with assert_raises(iris.exceptions.CoordinateNotFoundError):
            self.cube.coord('season_year')

    def test_get_jf(self):
        """Test function for custom seasons."""
        sliced = extract_season(self.cube, 'JF')
        iris.coord_categorisation.add_month_number(sliced, 'time')
        assert_array_equal(np.array([1, 2, 1, 2]),
                           sliced.coord('month_number').points)
        with assert_raises(iris.exceptions.CoordinateNotFoundError):
            self.cube.coord('clim_season')
        with assert_raises(iris.exceptions.CoordinateNotFoundError):
            self.cube.coord('season_year')


class TestClimatology(tests.Test):
    """Test class for :func:`esmvalcore.preprocessor._time.climatology`."""
    @staticmethod
    def _create_cube(data, times, bounds):
        time = iris.coords.DimCoord(times,
                                    bounds=bounds,
                                    standard_name='time',
                                    units=Unit('days since 1950-01-01',
                                               calendar='gregorian'))
        cube = iris.cube.Cube(data, dim_coords_and_dims=[(time, 0)])
        return cube

    def test_time_mean(self):
        """Test for time average of a 1D field."""
        data = np.array([1., 1., 1.], dtype=np.float32)
        times = np.array([15., 45., 75.])
        bounds = np.array([[0., 30.], [30., 60.], [60., 90.]])
        cube = self._create_cube(data, times, bounds)

        result = climate_statistics(cube, operator='mean')
        expected = np.array([1.], dtype=np.float32)
        assert_array_equal(result.data, expected)

    def test_time_mean_uneven(self):
        """Test for time average of a 1D field with uneven time boundaries."""
        data = np.array([1., 5.], dtype=np.float32)
        times = np.array([5., 25.])
        bounds = np.array([[0., 1.], [1., 4.]])
        cube = self._create_cube(data, times, bounds)

        result = climate_statistics(cube, operator='mean')
        expected = np.array([4.], dtype=np.float32)
        assert_array_equal(result.data, expected)

    def test_time_mean_365_day(self):
        """Test for time avg of a realistic time axis and 365 day calendar."""
        data = np.array([1., 1., 1., 1., 1., 1.], dtype=np.float32)
        times = np.array([15, 45, 74, 105, 135, 166])
        bounds = np.array([[0, 31], [31, 59], [59, 90], [90, 120], [120, 151],
                           [151, 181]])
        cube = self._create_cube(data, times, bounds)

        result = climate_statistics(cube, operator='mean')
        expected = np.array([1.], dtype=np.float32)
        assert_array_equal(result.data, expected)

    def test_time_sum(self):
        """Test for time sum of a 1D field."""
        data = np.array([1., 2., 1.], dtype=np.float32)
        times = np.array([15., 45., 75.])
        bounds = np.array([[0., 30.], [30., 60.], [60., 90.]])
        cube = self._create_cube(data, times, bounds)

        result = climate_statistics(cube, operator='sum')
        expected = np.array([4.], dtype=np.float32)
        assert_array_equal(result.data, expected)

    def test_time_sum_weighted(self):
        """Test for time sum of a 1D field."""
        data = np.array([1., 2., 1.], dtype=np.float32)
        times = np.array([15., 45., 75.])
        bounds = np.array([[10., 20.], [30., 60.], [73., 77.]])
        cube = self._create_cube(data, times, bounds)

        result = climate_statistics(cube, operator='sum')
        expected = np.array([74.], dtype=np.float32)
        assert_array_equal(result.data, expected)

    def test_time_sum_uneven(self):
        """Test for time sum of a 1D field with uneven time boundaries."""
        data = np.array([1., 5.], dtype=np.float32)
        times = np.array([5., 25.])
        bounds = np.array([[0., 1.], [1., 4.]])
        cube = self._create_cube(data, times, bounds)

        result = climate_statistics(cube, operator='sum')
        expected = np.array([16.0], dtype=np.float32)
        assert_array_equal(result.data, expected)

    def test_time_sum_365_day(self):
        """Test for time sum of a realistic time axis and 365 day calendar."""
        data = np.ones((6, ))
        data[3] = 2.0
        times = np.array([15, 45, 74, 105, 135, 166])
        bounds = np.array([[0, 31], [31, 59], [59, 90], [90, 120], [120, 151],
                           [151, 181]])
        cube = self._create_cube(data, times, bounds)

        result = climate_statistics(cube, operator='sum')
        expected = np.array([211.], dtype=np.float32)
        assert_array_equal(result.data, expected)

    def test_season_climatology(self):
        """Test for time avg of a realistic time axis and 365 day calendar."""
        data = np.array([1., 1., 1., 1., 1., 1.], dtype=np.float32)
        times = np.array([15, 45, 74, 105, 135, 166])
        bounds = np.array([[0, 31], [31, 59], [59, 90], [90, 120], [120, 151],
                           [151, 181]])
        cube = self._create_cube(data, times, bounds)

        result = climate_statistics(cube, operator='mean', period='season')
        expected = np.array([1., 1., 1.], dtype=np.float32)
        assert_array_equal(result.data, expected)

    def test_custom_season_climatology(self):
        """Test for time avg of a realisitc time axis and 365 day calendar."""
        data = np.array([1., 1., 1., 1., 1., 1., 1., 1.], dtype=np.float32)
        times = np.array([15, 45, 74, 105, 135, 166, 195, 225])
        bounds = np.array([[0, 31], [31, 59], [59, 90], [90, 120], [120, 151],
                           [151, 181], [181, 212], [212, 243]])
        cube = self._create_cube(data, times, bounds)

        result = climate_statistics(cube,
                                    operator='mean',
                                    period='season',
                                    seasons=('jfmamj', 'jasond'))
        expected = np.array([1., 1.], dtype=np.float32)
        assert_array_equal(result.data, expected)

    def test_monthly(self):
        """Test for time avg of a realistic time axis and 365 day calendar."""
        data = np.array([1., 1., 1., 1., 1., 1.], dtype=np.float32)
        times = np.array([15, 45, 74, 105, 135, 166])
        bounds = np.array([[0, 31], [31, 59], [59, 90], [90, 120], [120, 151],
                           [151, 181]])
        cube = self._create_cube(data, times, bounds)

        result = climate_statistics(cube, operator='mean', period='mon')
        expected = np.ones((6, ), dtype=np.float32)
        assert_array_equal(result.data, expected)

    def test_day(self):
        """Test for time avg of a realistic time axis and 365 day calendar."""
        data = np.array([1., 1., 1., 1., 1., 1.], dtype=np.float32)
        times = np.array([0.5, 1.5, 2.5, 365.5, 366.5, 367.5])
        bounds = np.array([[0, 1], [1, 2], [2, 3], [365, 366], [366, 367],
                           [367, 368]])
        cube = self._create_cube(data, times, bounds)

        result = climate_statistics(cube, operator='mean', period='day')
        expected = np.array([1, 1, 1], dtype=np.float32)
        assert_array_equal(result.data, expected)

    def test_period_not_supported(self):
        """Test for time avg of a realistic time axis and 365 day calendar."""
        data = np.array([1., 1., 1., 1., 1., 1.], dtype=np.float32)
        times = np.array([15, 45, 74, 105, 135, 166])
        bounds = np.array([[0, 31], [31, 59], [59, 90], [90, 120], [120, 151],
                           [151, 181]])
        cube = self._create_cube(data, times, bounds)

        with self.assertRaises(ValueError):
            climate_statistics(cube, operator='mean', period='bad')

    def test_time_max(self):
        """Test for time max of a 1D field."""
        data = np.array([0., 1., 2.], dtype=np.float32)
        times = np.array([15., 45., 75.])
        bounds = np.array([[0., 30.], [30., 60.], [60., 90.]])
        cube = self._create_cube(data, times, bounds)

        result = climate_statistics(cube, operator='max')
        expected = np.array([2.], dtype=np.float32)
        assert_array_equal(result.data, expected)

    def test_time_min(self):
        """Test for time min of a 1D field."""
        data = np.array([0., 1., 2.], dtype=np.float32)
        times = np.array([15., 45., 75.])
        bounds = np.array([[0., 30.], [30., 60.], [60., 90.]])
        cube = self._create_cube(data, times, bounds)

        result = climate_statistics(cube, operator='min')
        expected = np.array([0.], dtype=np.float32)
        assert_array_equal(result.data, expected)

    def test_time_median(self):
        """Test for time meadian of a 1D field."""
        data = np.array([0., 1., 2.], dtype=np.float32)
        times = np.array([15., 45., 75.])
        bounds = np.array([[0., 30.], [30., 60.], [60., 90.]])
        cube = self._create_cube(data, times, bounds)

        result = climate_statistics(cube, operator='median')
        expected = np.array([1.], dtype=np.float32)
        assert_array_equal(result.data, expected)

    def test_time_rms(self):
        """Test for time rms of a 1D field."""
        data = np.array([0., 1., 2.], dtype=np.float32)
        times = np.array([15., 45., 75.])
        bounds = np.array([[0., 30.], [30., 60.], [60., 90.]])
        cube = self._create_cube(data, times, bounds)

        result = climate_statistics(cube, operator='rms')
        expected = np.array([(5 / 3)**0.5], dtype=np.float32)
        assert_array_equal(result.data, expected)

    def test_time_dependent_fx(self):
        """Test average time dimension in time-dependent fx vars."""
        data = np.ones((3, 3, 3))
        times = np.array([15., 45., 75.])
        bounds = np.array([[0., 30.], [30., 60.], [60., 90.]])
        cube = self._create_cube(data, times, bounds)
        measure = iris.coords.CellMeasure(data,
                                          standard_name='ocean_volume',
                                          var_name='volcello',
                                          units='m3',
                                          measure='volume')
        ancillary_var = iris.coords.AncillaryVariable(
            data,
            standard_name='land_ice_area_fraction',
            var_name='sftgif',
            units='%')
        cube.add_cell_measure(measure, (0, 1, 2))
        cube.add_ancillary_variable(ancillary_var, (0, 1, 2))
        with self.assertLogs(level='DEBUG') as cm:
            result = climate_statistics(cube, operator='mean', period='mon')
        self.assertEqual(cm.records[0].getMessage(),
                         'Averaging time dimension in measure volcello.')
        self.assertEqual(
            cm.records[1].getMessage(),
            'Averaging time dimension in ancillary variable sftgif.')
        self.assertEqual(result.cell_measure('ocean_volume').ndim, 2)
        self.assertEqual(
            result.ancillary_variable('land_ice_area_fraction').ndim, 2)


class TestSeasonalStatistics(tests.Test):
    """Test :func:`esmvalcore.preprocessor._time.seasonal_statistics`."""
    @staticmethod
    def _create_cube(data, times):
        time = iris.coords.DimCoord(times,
                                    standard_name='time',
                                    units=Unit('days since 1950-01-01',
                                               calendar='360_day'))
        time.guess_bounds()
        cube = iris.cube.Cube(data, dim_coords_and_dims=[(time, 0)])
        return cube

    def test_season_mean(self):
        """Test for season average of a 1D field."""
        data = np.arange(12)
        times = np.arange(15, 360, 30)
        cube = self._create_cube(data, times)

        result = seasonal_statistics(cube, 'mean')
        expected = np.array([3., 6., 9.])
        assert_array_equal(result.data, expected)

    def test_season_median(self):
        """Test for season median of a 1D field."""
        data = np.arange(12)
        times = np.arange(15, 360, 30)
        cube = self._create_cube(data, times)

        result = seasonal_statistics(cube, 'median')
        expected = np.array([3., 6., 9.])
        assert_array_equal(result.data, expected)

    def test_season_min(self):
        """Test for season min of a 1D field."""
        data = np.arange(12)
        times = np.arange(15, 360, 30)
        cube = self._create_cube(data, times)

        result = seasonal_statistics(cube, 'min')
        expected = np.array([2., 5., 8.])
        assert_array_equal(result.data, expected)

    def test_season_max(self):
        """Test for season max of a 1D field."""
        data = np.arange(12)
        times = np.arange(15, 360, 30)
        cube = self._create_cube(data, times)

        result = seasonal_statistics(cube, 'max')
        expected = np.array([4., 7., 10.])
        assert_array_equal(result.data, expected)

    def test_season_sum(self):
        """Test for season sum of a 1D field."""
        data = np.arange(12)
        times = np.arange(15, 360, 30)
        cube = self._create_cube(data, times)

        result = seasonal_statistics(cube, 'sum')
        expected = np.array([9., 18., 27.])
        assert_array_equal(result.data, expected)

    def test_season_custom_mean(self):
        """Test for season average of a 1D field."""
        data = np.arange(12)
        times = np.arange(15, 360, 30)
        cube = self._create_cube(data, times)

        result = seasonal_statistics(cube,
                                     'mean',
                                     seasons=('jfmamj', 'jasond'))
        expected = np.array([2.5, 8.5])
        assert_array_equal(result.data, expected)

    def test_season_custom_spans_full_season(self):
        """Test for season average of a 1D field."""
        data = np.ones(12)
        times = np.arange(15, 360, 30)
        cube = self._create_cube(data, times)

        result = seasonal_statistics(cube,
                                     'mean',
                                     seasons=('JJAS', 'ondjfmam'))
        expected = np.array([1])
        assert_array_equal(result.data, expected)

    def test_time_dependent_fx(self):
        """Test average time dimension in time-dependent fx vars."""
        data = np.ones((12, 3, 3))
        times = np.arange(15, 360, 30)
        cube = self._create_cube(data, times)
        measure = iris.coords.CellMeasure(data,
                                          standard_name='ocean_volume',
                                          var_name='volcello',
                                          units='m3',
                                          measure='volume')
        ancillary_var = iris.coords.AncillaryVariable(
            data,
            standard_name='land_ice_area_fraction',
            var_name='sftgif',
            units='%')
        cube.add_cell_measure(measure, (0, 1, 2))
        cube.add_ancillary_variable(ancillary_var, (0, 1, 2))
        with self.assertLogs(level='DEBUG') as cm:
            result = seasonal_statistics(cube, operator='mean')
        self.assertEqual(cm.records[0].getMessage(),
                         'Averaging time dimension in measure volcello.')
        self.assertEqual(
            cm.records[1].getMessage(),
            'Averaging time dimension in ancillary variable sftgif.')
        self.assertEqual(result.cell_measure('ocean_volume').ndim, 2)
        self.assertEqual(
            result.ancillary_variable('land_ice_area_fraction').ndim, 2)


class TestMonthlyStatistics(tests.Test):
    """Test :func:`esmvalcore.preprocessor._time.monthly_statistics`."""
    @staticmethod
    def _create_cube(data, times):
        time = iris.coords.DimCoord(times,
                                    standard_name='time',
                                    units=Unit('days since 1950-01-01',
                                               calendar='360_day'))
        time.guess_bounds()
        cube = iris.cube.Cube(data, dim_coords_and_dims=[(time, 0)])
        return cube

    def test_mean(self):
        """Test average of a 1D field."""
        data = np.arange(24)
        times = np.arange(7, 360, 15)
        cube = self._create_cube(data, times)

        result = monthly_statistics(cube, 'mean')
        expected = np.array([
            0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5, 22.5
        ])
        assert_array_equal(result.data, expected)

    def test_median(self):
        """Test median of a 1D field."""
        data = np.arange(24)
        times = np.arange(7, 360, 15)
        cube = self._create_cube(data, times)

        result = monthly_statistics(cube, 'median')
        expected = np.array([
            0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5, 22.5
        ])
        assert_array_equal(result.data, expected)

    def test_min(self):
        """Test min of a 1D field."""
        data = np.arange(24)
        times = np.arange(7, 360, 15)
        cube = self._create_cube(data, times)

        result = monthly_statistics(cube, 'min')
        expected = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
        assert_array_equal(result.data, expected)

    def test_max(self):
        """Test max of a 1D field."""
        data = np.arange(24)
        times = np.arange(7, 360, 15)
        cube = self._create_cube(data, times)

        result = monthly_statistics(cube, 'max')
        expected = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23])
        assert_array_equal(result.data, expected)

    def test_sum(self):
        """Test sum of a 1D field."""
        data = np.arange(24)
        times = np.arange(7, 360, 15)
        cube = self._create_cube(data, times)

        result = monthly_statistics(cube, 'sum')
        expected = np.array([1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45])
        assert_array_equal(result.data, expected)

    def test_time_dependent_fx(self):
        """Test average time dimension in time-dependent fx vars."""
        data = np.ones((3, 3, 3))
        times = np.array([15., 45., 75.])
        cube = self._create_cube(data, times)
        measure = iris.coords.CellMeasure(data,
                                          standard_name='ocean_volume',
                                          var_name='volcello',
                                          units='m3',
                                          measure='volume')
        ancillary_var = iris.coords.AncillaryVariable(
            data,
            standard_name='land_ice_area_fraction',
            var_name='sftgif',
            units='%')
        cube.add_cell_measure(measure, (0, 1, 2))
        cube.add_ancillary_variable(ancillary_var, (0, 1, 2))
        with self.assertLogs(level='DEBUG') as cm:
            result = monthly_statistics(cube, operator='mean')
        self.assertEqual(cm.records[0].getMessage(),
                         'Averaging time dimension in measure volcello.')
        self.assertEqual(
            cm.records[1].getMessage(),
            'Averaging time dimension in ancillary variable sftgif.')
        self.assertEqual(result.cell_measure('ocean_volume').ndim, 2)
        self.assertEqual(
            result.ancillary_variable('land_ice_area_fraction').ndim, 2)


class TestHourlyStatistics(tests.Test):
    """Test :func:`esmvalcore.preprocessor._time.hourly_statistics`."""
    @staticmethod
    def _create_cube(data, times):
        time = iris.coords.DimCoord(times,
                                    standard_name='time',
                                    units=Unit('hours since 1950-01-01',
                                               calendar='360_day'))
        time.guess_bounds()
        cube = iris.cube.Cube(data, dim_coords_and_dims=[(time, 0)])
        return cube

    def test_mean(self):
        """Test average of a 1D field."""
        data = np.arange(8)
        times = np.arange(0, 48, 6)
        cube = self._create_cube(data, times)

        result = hourly_statistics(cube, 12, 'mean')
        expected = np.array([0.5, 2.5, 4.5, 6.5])
        assert_array_equal(result.data, expected)

    def test_median(self):
        """Test median of a 1D field."""
        data = np.arange(8)
        times = np.arange(0, 48, 6)
        cube = self._create_cube(data, times)

        result = hourly_statistics(cube, 12, 'median')
        expected = np.array([0.5, 2.5, 4.5, 6.5])
        assert_array_equal(result.data, expected)

    def test_min(self):
        """Test min of a 1D field."""
        data = np.arange(8)
        times = np.arange(0, 48, 6)
        cube = self._create_cube(data, times)

        result = hourly_statistics(cube, 12, 'min')
        expected = np.array([0., 2., 4., 6.])
        assert_array_equal(result.data, expected)

    def test_max(self):
        """Test max of a 1D field."""
        data = np.arange(8)
        times = np.arange(0, 48, 6)
        cube = self._create_cube(data, times)

        result = hourly_statistics(cube, 12, 'max')
        expected = np.array([1., 3., 5., 7.])
        assert_array_equal(result.data, expected)

    def test_sum(self):
        """Test sum of a 1D field."""
        data = np.arange(8)
        times = np.arange(0, 48, 6)
        cube = self._create_cube(data, times)

        result = hourly_statistics(cube, 12, 'sum')
        expected = np.array([1., 5., 9., 13.])
        assert_array_equal(result.data, expected)


class TestDailyStatistics(tests.Test):
    """Test :func:`esmvalcore.preprocessor._time.monthly_statistics`."""
    @staticmethod
    def _create_cube(data, times):
        time = iris.coords.DimCoord(times,
                                    standard_name='time',
                                    units=Unit('hours since 1950-01-01',
                                               calendar='360_day'))
        time.guess_bounds()
        cube = iris.cube.Cube(data, dim_coords_and_dims=[(time, 0)])
        return cube

    def test_mean(self):
        """Test average of a 1D field."""
        data = np.arange(8)
        times = np.arange(0, 48, 6)
        cube = self._create_cube(data, times)

        result = daily_statistics(cube, 'mean')
        expected = np.array([1.5, 5.5])
        assert_array_equal(result.data, expected)

    def test_median(self):
        """Test median of a 1D field."""
        data = np.arange(8)
        times = np.arange(0, 48, 6)
        cube = self._create_cube(data, times)

        result = daily_statistics(cube, 'median')
        expected = np.array([1.5, 5.5])
        assert_array_equal(result.data, expected)

    def test_min(self):
        """Test min of a 1D field."""
        data = np.arange(8)
        times = np.arange(0, 48, 6)
        cube = self._create_cube(data, times)

        result = daily_statistics(cube, 'min')
        expected = np.array([0., 4.])
        assert_array_equal(result.data, expected)

    def test_max(self):
        """Test max of a 1D field."""
        data = np.arange(8)
        times = np.arange(0, 48, 6)
        cube = self._create_cube(data, times)

        result = daily_statistics(cube, 'max')
        expected = np.array([3., 7.])
        assert_array_equal(result.data, expected)

    def test_sum(self):
        """Test sum of a 1D field."""
        data = np.arange(8)
        times = np.arange(0, 48, 6)
        cube = self._create_cube(data, times)

        result = daily_statistics(cube, 'sum')
        expected = np.array([6., 22.])
        assert_array_equal(result.data, expected)


class TestRegridTimeYearly(tests.Test):
    """Tests for regrid_time with monthly frequency."""
    def setUp(self):
        """Prepare tests."""
        self.cube_1 = _create_sample_cube()
        self.cube_2 = _create_sample_cube()
        self.cube_2.data = self.cube_2.data * 2.
        self.cube_2.remove_coord('time')
        self.cube_1.remove_coord('time')
        self.cube_1.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(11., 8770., 365.),
                standard_name='time',
                units=Unit('days since 1950-01-01 00:00:00',
                           calendar='gregorian'),
            ),
            0,
        )
        self.cube_2.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(91., 8851., 365.),
                standard_name='time',
                units=Unit('days since 1950-01-01 00:00:00',
                           calendar='gregorian'),
            ),
            0,
        )
        add_auxiliary_coordinate([self.cube_1, self.cube_2])

    def test_regrid_time_year(self):
        """Test changes to cubes."""
        # test yearly
        newcube_1 = regrid_time(self.cube_1, frequency='yr')
        newcube_2 = regrid_time(self.cube_2, frequency='yr')
        # no changes to core data
        assert_array_equal(newcube_1.data, self.cube_1.data)
        assert_array_equal(newcube_2.data, self.cube_2.data)
        # no changes to number of coords and aux_coords
        assert len(newcube_1.coords()) == len(self.cube_1.coords())
        assert len(newcube_1.aux_coords) == len(self.cube_1.aux_coords)
        # test difference; also diff is zero
        expected = self.cube_1.data
        diff_cube = newcube_2 - newcube_1
        assert_array_equal(diff_cube.data, expected)
        # test bounds are set at [01-01-YEAR 00:00, 01-01-NEXT_YEAR 00:00]
        timeunit_1 = newcube_1.coord('time').units
        for i, time in enumerate(newcube_1.coord('time').points):
            year_1 = timeunit_1.num2date(time).year
            expected_minbound = date2num(datetime(year_1, 1, 1),
                                         timeunit_1)
            expected_maxbound = date2num(datetime(year_1 + 1, 1, 1),
                                         timeunit_1)
            assert_array_equal(
                newcube_1.coord('time').bounds[i],
                np.array([expected_minbound, expected_maxbound]))


class TestRegridTimeMonthly(tests.Test):
    """Tests for regrid_time with monthly frequency."""
    def setUp(self):
        """Prepare tests."""
        self.cube_1 = _create_sample_cube()
        self.cube_2 = _create_sample_cube()
        self.cube_2.data = self.cube_2.data * 2.
        self.cube_2.remove_coord('time')
        self.cube_2.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(14., 719., 30.),
                standard_name='time',
                units=Unit('days since 1950-01-01 00:00:00',
                           calendar='gregorian'),
            ),
            0,
        )
        add_auxiliary_coordinate([self.cube_1, self.cube_2])

    def test_regrid_time_mon(self):
        """Test changes to cubes."""
        # test monthly
        newcube_1 = regrid_time(self.cube_1, frequency='mon')
        newcube_2 = regrid_time(self.cube_2, frequency='mon')
        # no changes to core data
        assert_array_equal(newcube_1.data, self.cube_1.data)
        assert_array_equal(newcube_2.data, self.cube_2.data)
        # no changes to number of coords and aux_coords
        assert len(newcube_1.coords()) == len(self.cube_1.coords())
        assert len(newcube_1.aux_coords) == len(self.cube_1.aux_coords)
        # test difference; also diff is zero
        expected = self.cube_1.data
        diff_cube = newcube_2 - newcube_1
        assert_array_equal(diff_cube.data, expected)
        # test bounds are set at
        # [01-MONTH-YEAR 00:00, 01-NEXT_MONTH-YEAR 00:00]
        timeunit_1 = newcube_1.coord('time').units
        for i, time in enumerate(newcube_1.coord('time').points):
            month_1 = timeunit_1.num2date(time).month
            year_1 = timeunit_1.num2date(time).year
            next_month = month_1 + 1
            next_year = year_1
            if month_1 == 12:
                next_month = 1
                next_year += 1
            expected_minbound = date2num(datetime(year_1, month_1, 1),
                                         timeunit_1)
            expected_maxbound = date2num(datetime(next_year, next_month, 1),
                                         timeunit_1)
            assert_array_equal(
                newcube_1.coord('time').bounds[i],
                np.array([expected_minbound, expected_maxbound]))

    def test_regrid_time_different_calendar_bounds(self):
        """Test bounds in different calendars."""
        cube_360 = _create_sample_cube(calendar='360_day')
        # Same cubes but differing time units
        newcube_360 = regrid_time(cube_360, frequency='mon')
        newcube_gregorian = regrid_time(self.cube_1, frequency='mon')
        bounds_360 = newcube_360.coord('time').bounds
        bounds_gregorian = newcube_gregorian.coord('time').bounds
        # test value of the bounds is not the same
        assert (bounds_360 != bounds_gregorian).any()
        # assert length of the 360_day bounds interval is 30 days
        assert (bounds_360[:, 1] - bounds_360[:, 0] == 30).all()


class TestRegridTimeDaily(tests.Test):
    """Tests for regrid_time with daily frequency."""
    def setUp(self):
        """Prepare tests."""
        self.cube_1 = _create_sample_cube()
        self.cube_2 = _create_sample_cube()
        self.cube_2.data = self.cube_2.data * 2.
        self.cube_1.remove_coord('time')
        self.cube_2.remove_coord('time')
        self.cube_1.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(14. * 24. + 6., 38. * 24. + 6., 24.),
                standard_name='time',
                units=Unit('hours since 1950-01-01 00:00:00',
                           calendar='gregorian'),
            ),
            0,
        )
        self.cube_2.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(14. * 24. + 3., 38. * 24. + 3., 24.),
                standard_name='time',
                units=Unit('hours since 1950-01-01 00:00:00',
                           calendar='gregorian'),
            ),
            0,
        )
        add_auxiliary_coordinate([self.cube_1, self.cube_2])

    def test_regrid_time_day(self):
        """Test changes to cubes."""
        # test daily
        newcube_1 = regrid_time(self.cube_1, frequency='day')
        newcube_2 = regrid_time(self.cube_2, frequency='day')
        # no changes to core data
        self.assert_array_equal(newcube_1.data, self.cube_1.data)
        self.assert_array_equal(newcube_2.data, self.cube_2.data)
        # no changes to number of coords and aux_coords
        assert len(newcube_1.coords()) == len(self.cube_1.coords())
        assert len(newcube_1.aux_coords) == len(self.cube_1.aux_coords)
        # test difference; also diff is zero
        expected = self.cube_1.data
        diff_cube = newcube_2 - newcube_1
        self.assert_array_equal(diff_cube.data, expected)
        # test bounds are set with a dt = 12/24 days
        for i, time in enumerate(newcube_1.coord('time').points):
            expected_minbound = time - 12 / 24
            expected_maxbound = time + 12 / 24
            assert_array_equal(
                newcube_1.coord('time').bounds[i],
                np.array([expected_minbound, expected_maxbound]))


class TestRegridTime6Hourly(tests.Test):
    """Tests for regrid_time with 6-hourly frequency."""
    def setUp(self):
        """Prepare tests."""
        self.cube_1 = _create_sample_cube()
        self.cube_2 = _create_sample_cube()
        self.cube_2.data = self.cube_2.data * 2.
        self.cube_1.remove_coord('time')
        self.cube_2.remove_coord('time')
        self.cube_1.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(10. * 6. + 5., 34. * 6. + 5., 6.),
                standard_name='time',
                units=Unit('hours since 1950-01-01 00:00:00',
                           calendar='360_day'),
            ),
            0,
        )
        self.cube_2.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(10. * 6. + 2., 34. * 6. + 2., 6.),
                standard_name='time',
                units=Unit('hours since 1950-01-01 00:00:00',
                           calendar='360_day'),
            ),
            0,
        )
        add_auxiliary_coordinate([self.cube_1, self.cube_2])

    def test_regrid_time_6hour(self):
        """Test changes to cubes."""
        # test 6-hourly
        newcube_1 = regrid_time(self.cube_1, frequency='6hr')
        newcube_2 = regrid_time(self.cube_2, frequency='6hr')
        # no changes to core data
        self.assert_array_equal(newcube_1.data, self.cube_1.data)
        self.assert_array_equal(newcube_2.data, self.cube_2.data)
        # no changes to number of coords and aux_coords
        assert len(newcube_1.coords()) == len(self.cube_1.coords())
        assert len(newcube_1.aux_coords) == len(self.cube_1.aux_coords)
        # test difference; also diff is zero
        expected = self.cube_1.data
        diff_cube = newcube_2 - newcube_1
        self.assert_array_equal(diff_cube.data, expected)
        # test bounds are set with a dt = 3/24 days
        for i, time in enumerate(newcube_1.coord('time').points):
            expected_minbound = time - 3 / 24
            expected_maxbound = time + 3 / 24
            assert_array_equal(
                newcube_1.coord('time').bounds[i],
                np.array([expected_minbound, expected_maxbound]))


class TestRegridTime3Hourly(tests.Test):
    """Tests for regrid_time with 3-hourly frequency."""
    def setUp(self):
        """Prepare tests."""
        self.cube_1 = _create_sample_cube()
        self.cube_2 = _create_sample_cube()
        self.cube_2.data = self.cube_2.data * 2.
        self.cube_1.remove_coord('time')
        self.cube_2.remove_coord('time')
        self.cube_1.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(17. * 180. + 40., 41. * 180. + 40., 180.),
                standard_name='time',
                units=Unit('minutes since 1950-01-01 00:00:00',
                           calendar='gregorian'),
            ),
            0,
        )
        self.cube_2.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(17. * 180. + 150., 41. * 180. + 150., 180.),
                standard_name='time',
                units=Unit('minutes since 1950-01-01 00:00:00',
                           calendar='gregorian'),
            ),
            0,
        )
        add_auxiliary_coordinate([self.cube_1, self.cube_2])

    def test_regrid_time_3hour(self):
        """Test changes to cubes."""
        # test 3-hourly
        newcube_1 = regrid_time(self.cube_1, frequency='3hr')
        newcube_2 = regrid_time(self.cube_2, frequency='3hr')
        # no changes to core data
        self.assert_array_equal(newcube_1.data, self.cube_1.data)
        self.assert_array_equal(newcube_2.data, self.cube_2.data)
        # no changes to number of coords and aux_coords
        assert len(newcube_1.coords()) == len(self.cube_1.coords())
        assert len(newcube_1.aux_coords) == len(self.cube_1.aux_coords)
        # test difference; also diff is zero
        expected = self.cube_1.data
        diff_cube = newcube_2 - newcube_1
        self.assert_array_equal(diff_cube.data, expected)
        # test bounds are set with a dt = 1.5/24 days
        for i, time in enumerate(newcube_1.coord('time').points):
            expected_minbound = time - 1.5 / 24
            expected_maxbound = time + 1.5 / 24
            assert_array_equal(
                newcube_1.coord('time').bounds[i],
                np.array([expected_minbound, expected_maxbound]))


class TestRegridTime1Hourly(tests.Test):
    """Tests for regrid_time with hourly frequency."""
    def setUp(self):
        """Prepare tests."""
        self.cube_1 = _create_sample_cube()
        self.cube_2 = _create_sample_cube()
        self.cube_2.data = self.cube_2.data * 2.
        self.cube_1.remove_coord('time')
        self.cube_2.remove_coord('time')
        self.cube_1.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(14. * 60. + 6., 38. * 60. + 6., 60.),
                standard_name='time',
                units=Unit('minutes since 1950-01-01 00:00:00',
                           calendar='360_day'),
            ),
            0,
        )
        self.cube_2.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(14. * 60. + 34., 38. * 60. + 34., 60.),
                standard_name='time',
                units=Unit('minutes since 1950-01-01 00:00:00',
                           calendar='360_day'),
            ),
            0,
        )
        add_auxiliary_coordinate([self.cube_1, self.cube_2])

    def test_regrid_time_hour(self):
        """Test changes to cubes."""
        # test hourly
        newcube_1 = regrid_time(self.cube_1, frequency='1hr')
        newcube_2 = regrid_time(self.cube_2, frequency='1hr')
        # no changes to core data
        self.assert_array_equal(newcube_1.data, self.cube_1.data)
        self.assert_array_equal(newcube_2.data, self.cube_2.data)
        # no changes to number of coords and aux_coords
        assert len(newcube_1.coords()) == len(self.cube_1.coords())
        assert len(newcube_1.aux_coords) == len(self.cube_1.aux_coords)
        # test difference; also diff is zero
        expected = self.cube_1.data
        diff_cube = newcube_2 - newcube_1
        self.assert_array_equal(diff_cube.data, expected)
        # test bounds are set with a dt = 0.5/24 days
        for i, time in enumerate(newcube_1.coord('time').points):
            expected_minbound = time - 0.5 / 24
            expected_maxbound = time + 0.5 / 24
            assert_array_equal(
                newcube_1.coord('time').bounds[i],
                np.array([expected_minbound, expected_maxbound]))


class TestTimeseriesFilter(tests.Test):
    """Tests for regrid_time with hourly frequency."""
    def setUp(self):
        """Prepare tests."""
        self.cube = _create_sample_cube()

    def test_timeseries_filter_simple(self):
        """Test timeseries_filter func."""
        filtered_cube = timeseries_filter(self.cube,
                                          7,
                                          14,
                                          filter_type='lowpass',
                                          filter_stats='sum')
        expected_data = np.array([
            2.44824568, 3.0603071, 3.67236852, 4.28442994, 4.89649137,
            5.50855279, 6.12061421, 6.73267563, 7.34473705, 7.95679847,
            8.56885989, 9.18092131, 9.79298273, 10.40504415, 11.01710557,
            11.62916699, 12.24122841, 12.85328983
        ])
        assert_array_almost_equal(filtered_cube.data, expected_data)
        assert len(filtered_cube.coord('time').points) == 18

    def test_timeseries_filter_timecoord(self):
        """Test missing time axis."""
        new_cube = self.cube.copy()
        new_cube.remove_coord(new_cube.coord('time'))
        with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
            timeseries_filter(new_cube,
                              7,
                              14,
                              filter_type='lowpass',
                              filter_stats='sum')

    def test_timeseries_filter_implemented(self):
        """Test a not implemnted filter."""
        with self.assertRaises(NotImplementedError):
            timeseries_filter(self.cube,
                              7,
                              14,
                              filter_type='bypass',
                              filter_stats='sum')


def make_time_series(number_years=2):
    """Make a cube with time only dimension."""
    times = np.array([i * 30 + 15 for i in range(0, 12 * number_years, 1)])
    bounds = np.array([i * 30 for i in range(0, 12 * number_years + 1, 1)])
    bounds = np.array([[bnd, bounds[index + 1]]
                       for index, bnd in enumerate(bounds[:-1])])
    data = np.ones_like(times)
    time = iris.coords.DimCoord(times,
                                bounds=bounds,
                                standard_name='time',
                                units=Unit('days since 1950-01-01',
                                           calendar='360_day'))
    cube = iris.cube.Cube(data, dim_coords_and_dims=[(time, 0)])
    return cube


@pytest.mark.parametrize('existing_coord', [True, False])
def test_annual_average(existing_coord):
    """Test for annual average."""
    cube = make_time_series(number_years=2)
    if existing_coord:
        iris.coord_categorisation.add_year(cube, 'time')

    result = annual_statistics(cube)
    expected = np.array([1., 1.])
    assert_array_equal(result.data, expected)
    expected_time = np.array([180., 540.])
    assert_array_equal(result.coord('time').points, expected_time)


@pytest.mark.parametrize('existing_coord', [True, False])
def test_annual_sum(existing_coord):
    """Test for annual sum."""
    cube = make_time_series(number_years=2)
    if existing_coord:
        iris.coord_categorisation.add_year(cube, 'time')

    result = annual_statistics(cube, 'sum')
    expected = np.array([12., 12.])
    assert_array_equal(result.data, expected)
    expected_time = np.array([180., 540.])
    assert_array_equal(result.coord('time').points, expected_time)


@pytest.mark.parametrize('existing_coord', [True, False])
def test_decadal_average(existing_coord):
    """Test for decadal average."""
    cube = make_time_series(number_years=20)
    if existing_coord:

        def get_decade(coord, value):
            """Get decades from cube."""
            date = coord.units.num2date(value)
            return date.year - date.year % 10

        iris.coord_categorisation.add_categorised_coord(
            cube, 'decade', 'time', get_decade)

    result = decadal_statistics(cube)
    expected = np.array([1., 1.])
    assert_array_equal(result.data, expected)
    expected_time = np.array([1800., 5400.])
    assert_array_equal(result.coord('time').points, expected_time)


@pytest.mark.parametrize('existing_coord', [True, False])
def test_decadal_average_time_dependent_fx(existing_coord):
    """Test for decadal average."""
    cube = make_time_series(number_years=20)
    measure = iris.coords.CellMeasure(cube.data,
                                      standard_name='ocean_volume',
                                      var_name='volcello',
                                      units='m3',
                                      measure='volume')
    ancillary_var = iris.coords.AncillaryVariable(
        cube.data,
        standard_name='land_ice_area_fraction',
        var_name='sftgif',
        units='%')
    cube.add_cell_measure(measure, 0)
    cube.add_ancillary_variable(ancillary_var, 0)
    if existing_coord:
        def get_decade(coord, value):
            """Get decades from cube."""
            date = coord.units.num2date(value)
            return date.year - date.year % 10

        iris.coord_categorisation.add_categorised_coord(
            cube, 'decade', 'time', get_decade)
    result = decadal_statistics(cube)
    assert result.cell_measure('ocean_volume').data.shape == (1,)
    assert result.ancillary_variable(
        'land_ice_area_fraction').data.shape == (1,)


@pytest.mark.parametrize('existing_coord', [True, False])
def test_decadal_sum(existing_coord):
    """Test for decadal average."""
    cube = make_time_series(number_years=20)
    if existing_coord:

        def get_decade(coord, value):
            """Get decades from cube."""
            date = coord.units.num2date(value)
            return date.year - date.year % 10

        iris.coord_categorisation.add_categorised_coord(
            cube, 'decade', 'time', get_decade)

    result = decadal_statistics(cube, 'sum')
    expected = np.array([120., 120.])
    assert_array_equal(result.data, expected)
    expected_time = np.array([1800., 5400.])
    assert_array_equal(result.coord('time').points, expected_time)


def make_map_data(number_years=2):
    """Make a cube with time, lat and lon dimensions."""
    times = np.arange(0.5, number_years * 360)
    bounds = np.stack(((times - 0.5), (times + 0.5)), 1)
    time = iris.coords.DimCoord(times,
                                bounds=bounds,
                                standard_name='time',
                                units=Unit('days since 1950-01-01',
                                           calendar='360_day'))
    lat = iris.coords.DimCoord(
        range(2),
        standard_name='latitude',
    )
    lon = iris.coords.DimCoord(
        range(2),
        standard_name='longitude',
    )
    data = np.array([[0, 1], [1, 0]]) * times[:, None, None]
    cube = iris.cube.Cube(
        data,
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
    )
    return cube


PARAMETERS: List[Tuple] = []
for period in ('full', 'day', 'month', 'season'):
    PARAMETERS.append((period, None))
    if period == 'season':
        PARAMETERS.append((period, {
            "start_year": 1950,
            'start_month': 3,
            'start_day': 1,
            "end_year": 1951,
            'end_month': 3,
            'end_day': 1,
        }))
    else:
        PARAMETERS.append((period, {
            "start_year": 1950,
            'start_month': 1,
            'start_day': 1,
            "end_year": 1951,
            'end_month': 1,
            'end_day': 1,
        }))


@pytest.mark.parametrize('period', ['full'])
def test_standardized_anomalies(period, standardize=True):
    """Test standardized ``anomalies``."""
    cube = make_map_data(number_years=2)
    result = anomalies(cube, period, standardize=standardize)
    if period == 'full':
        expected_anomalies = (cube.data -
                              np.mean(cube.data, axis=0, keepdims=True))
        if standardize:
            # NB: default behaviour for np.std is ddof=0, whereas
            #     default behaviour for iris.analysis.STD_DEV is ddof=1
            expected_stdanomalies = expected_anomalies / np.std(
                expected_anomalies, axis=0, keepdims=True, ddof=1)
            expected = np.ma.masked_invalid(expected_stdanomalies)
            assert_array_equal(result.data, expected)
            assert result.units == '1'
        else:
            expected = np.ma.masked_invalid(expected_anomalies)
            assert_array_equal(result.data, expected)


@pytest.mark.parametrize('period, reference', PARAMETERS)
def test_anomalies_preserve_metadata(period, reference, standardize=False):
    """Test that ``anomalies`` preserves metadata."""
    cube = make_map_data(number_years=2)
    cube.var_name = "si"
    cube.units = "m"
    metadata = copy.deepcopy(cube.metadata)
    result = anomalies(cube, period, reference, standardize=standardize)
    assert result.metadata == metadata
    for coord_cube, coord_res in zip(cube.coords(), result.coords()):
        if coord_cube.has_bounds() and coord_res.has_bounds():
            assert_array_equal(coord_cube.bounds, coord_res.bounds)
        assert coord_cube == coord_res


@pytest.mark.parametrize('period, reference', PARAMETERS)
def test_anomalies(period, reference, standardize=False):
    """Test ``anomalies``."""
    cube = make_map_data(number_years=2)
    result = anomalies(cube, period, reference, standardize=standardize)
    if reference is None:
        if period == 'full':
            anom = np.arange(-359.5, 360)
        elif period == 'day':
            anom = np.concatenate((np.ones(360) * -180, np.ones(360) * 180))
        elif period == 'month':
            anom1 = np.concatenate(
                [np.arange(-194.5, -165) for x in range(12)])
            anom2 = np.concatenate([np.arange(165.5, 195) for x in range(12)])
            anom = np.concatenate((anom1, anom2))
        elif period == 'season':
            anom = np.concatenate((
                np.arange(-314.5, -255),
                np.arange(-224.5, -135),
                np.arange(-224.5, -135),
                np.arange(-224.5, -135),
                np.arange(15.5, 105),
                np.arange(135.5, 225),
                np.arange(135.5, 225),
                np.arange(135.5, 225),
                np.arange(375.5, 405),
            ))
    else:
        if period == 'full':
            anom = np.arange(-179.5, 540)
        elif period == 'day':
            anom = np.concatenate((np.zeros(360), np.ones(360) * 360))
        elif period == 'month':
            anom1 = np.concatenate([np.arange(-14.5, 15) for x in range(12)])
            anom2 = np.concatenate([np.arange(345.5, 375) for x in range(12)])
            anom = np.concatenate((anom1, anom2))
        elif period == 'season':
            anom = np.concatenate((
                np.arange(-374.5, -315),
                np.arange(-44.5, 45),
                np.arange(-44.5, 45),
                np.arange(-44.5, 45),
                np.arange(-44.5, 45),
                np.arange(315.5, 405),
                np.arange(315.5, 405),
                np.arange(315.5, 405),
                np.arange(315.5, 345),
            ))
    expected = anom[:, None, None] * [[0, 1], [1, 0]]
    assert_array_equal(result.data, expected)
    assert_array_equal(result.coord('time').points, cube.coord('time').points)


def test_anomalies_custom_season():
    """Test ``anomalies`` with custom season."""
    cube = make_map_data(number_years=2)
    result = anomalies(cube, 'season', seasons=('jfmamj', 'jasond'))
    anom = np.concatenate((
        np.arange(-269.5, -90),
        np.arange(-269.5, -90),
        np.arange(90.5, 270),
        np.arange(90.5, 270),
    ))
    expected = anom[:, None, None] * [[0, 1], [1, 0]]
    assert_array_equal(result.data, expected)
    assert_array_equal(result.coord('time').points, cube.coord('time').points)


def get_0d_time():
    """Get 0D time coordinate."""
    time = iris.coords.AuxCoord(15.0,
                                bounds=[0.0, 30.0],
                                standard_name='time',
                                units='days since 1850-01-01 00:00:00')
    return time


def get_1d_time():
    """Get 1D time coordinate."""
    time = iris.coords.DimCoord([20., 45.],
                                standard_name='time',
                                bounds=[[15., 30.], [30., 60.]],
                                units=Unit('days since 1950-01-01',
                                           calendar='gregorian'))
    return time


def get_2d_time():
    """Get 2D time coordinate."""
    time = iris.coords.AuxCoord([[20., 45.]],
                                standard_name='time',
                                bounds=[[[15., 30.], [30., 60.]]],
                                units=Unit('days since 1950-01-01',
                                           calendar='gregorian'))
    return time


def get_lon_coord():
    """Get longitude coordinate."""
    lons = iris.coords.DimCoord([1.5, 2.5, 3.5],
                                standard_name='longitude',
                                long_name='longitude',
                                bounds=[[1., 2.], [2., 3.], [3., 4.]],
                                units='degrees_east')
    return lons


def _make_cube():
    """Make a test cube."""
    coord_sys = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
    data2 = np.ma.ones((2, 1, 1, 3))

    time = get_1d_time()
    zcoord = iris.coords.DimCoord([0.5],
                                  standard_name='air_pressure',
                                  long_name='air_pressure',
                                  bounds=[[0., 2.5]],
                                  units='Pa',
                                  attributes={'positive': 'down'})
    lats = iris.coords.DimCoord([1.5],
                                standard_name='latitude',
                                long_name='latitude',
                                bounds=[[1., 2.]],
                                units='degrees_north',
                                coord_system=coord_sys)
    lons = get_lon_coord()
    coords_spec4 = [(time, 0), (zcoord, 1), (lats, 2), (lons, 3)]
    cube1 = iris.cube.Cube(data2, dim_coords_and_dims=coords_spec4)
    return cube1


def test_get_time_weights():
    """Test ``get_time_weights`` for complex cube."""
    cube = _make_cube()
    weights = get_time_weights(cube)
    assert weights.shape == (2, )
    np.testing.assert_allclose(weights, [15.0, 30.0])


def test_get_time_weights_0d_time():
    """Test ``get_time_weights`` for 0D time coordinate."""
    time = get_0d_time()
    cube = iris.cube.Cube(0.0,
                          var_name='x',
                          units='K',
                          aux_coords_and_dims=[(time, ())])
    weights = get_time_weights(cube)
    assert weights.shape == (1, )
    np.testing.assert_allclose(weights, [30.0])


def test_get_time_weights_0d_time_1d_lon():
    """Test ``get_time_weights`` for 0D time and 1D longitude coordinate."""
    time = get_0d_time()
    lons = get_lon_coord()
    cube = iris.cube.Cube([0.0, 0.0, 0.0],
                          var_name='x',
                          units='K',
                          aux_coords_and_dims=[(time, ())],
                          dim_coords_and_dims=[(lons, 0)])
    weights = get_time_weights(cube)
    assert weights.shape == (1, )
    np.testing.assert_allclose(weights, [30.0])


def test_get_time_weights_1d_time():
    """Test ``get_time_weights`` for 1D time coordinate."""
    time = get_1d_time()
    cube = iris.cube.Cube([0.0, 1.0],
                          var_name='x',
                          units='K',
                          dim_coords_and_dims=[(time, 0)])
    weights = get_time_weights(cube)
    assert weights.shape == (2, )
    np.testing.assert_allclose(weights, [15.0, 30.0])


def test_get_time_weights_1d_time_1d_lon():
    """Test ``get_time_weights`` for 1D time and 1D longitude coordinate."""
    time = get_1d_time()
    lons = get_lon_coord()
    cube = iris.cube.Cube([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                          var_name='x',
                          units='K',
                          dim_coords_and_dims=[(time, 0), (lons, 1)])
    weights = get_time_weights(cube)
    assert weights.shape == (2, )
    np.testing.assert_allclose(weights, [15.0, 30.0])


def test_get_time_weights_2d_time():
    """Test ``get_time_weights`` for 1D time coordinate."""
    time = get_2d_time()
    cube = iris.cube.Cube([[0.0, 1.0]],
                          var_name='x',
                          units='K',
                          aux_coords_and_dims=[(time, (0, 1))])
    with pytest.raises(ValueError):
        get_time_weights(cube)


def test_climate_statistics_0d_time_1d_lon():
    """Test climate statistics."""
    time = iris.coords.DimCoord([1.0],
                                bounds=[[0.0, 2.0]],
                                var_name='time',
                                standard_name='time',
                                units='days since 1850-01-01 00:00:00')
    lons = get_lon_coord()
    cube = iris.cube.Cube([[1.0, -1.0, 42.0]],
                          var_name='x',
                          units='K',
                          dim_coords_and_dims=[(time, 0), (lons, 1)])
    new_cube = climate_statistics(cube, operator='sum', period='full')
    assert cube.shape == (1, 3)
    assert new_cube.shape == (3, )
    np.testing.assert_allclose(new_cube.data, [1.0, -1.0, 42.0])


def test_climate_statistics_complex_cube_sum():
    """Test climate statistics."""
    cube = _make_cube()
    new_cube = climate_statistics(cube, operator='sum', period='full')
    assert cube.shape == (2, 1, 1, 3)
    assert new_cube.shape == (1, 1, 3)
    np.testing.assert_allclose(new_cube.data, [[[45.0, 45.0, 45.0]]])


def test_climate_statistics_complex_cube_mean():
    """Test climate statistics."""
    cube = _make_cube()
    new_cube = climate_statistics(cube, operator='mean', period='full')
    assert cube.shape == (2, 1, 1, 3)
    assert new_cube.shape == (1, 1, 3)
    np.testing.assert_allclose(new_cube.data, [[[1.0, 1.0, 1.0]]])


class TestResampleHours(tests.Test):
    """Test :func:`esmvalcore.preprocessor._time.resample_hours`."""
    @staticmethod
    def _create_cube(data, times):
        time = iris.coords.DimCoord(times,
                                    standard_name='time',
                                    units=Unit('hours since 1950-01-01',
                                               calendar='360_day'))
        time.guess_bounds()
        cube = iris.cube.Cube(data, dim_coords_and_dims=[(time, 0)])
        return cube

    def test_resample_1_to_6(self):
        """Test average of a 1D field."""
        data = np.arange(0, 48, 1)
        times = np.arange(0, 48, 1)
        cube = self._create_cube(data, times)

        result = resample_hours(cube, 6)
        expected = np.arange(0, 48, 6)
        assert_array_equal(result.data, expected)

    def test_resample_3_to_6(self):
        """Test average of a 1D field."""
        data = np.arange(0, 48, 3)
        times = np.arange(0, 48, 3)
        cube = self._create_cube(data, times)

        result = resample_hours(cube, 6)
        expected = np.arange(0, 48, 6)
        assert_array_equal(result.data, expected)

    def test_resample_1_to_3(self):
        """Test average of a 1D field."""
        data = np.arange(0, 48, 1)
        times = np.arange(0, 48, 1)
        cube = self._create_cube(data, times)

        result = resample_hours(cube, 3)
        expected = np.arange(0, 48, 3)
        assert_array_equal(result.data, expected)

    def test_resample_1_to_3_with_offset2(self):
        """Test average of a 1D field."""
        data = np.arange(0, 48, 1)
        times = np.arange(0, 48, 1)
        cube = self._create_cube(data, times)

        result = resample_hours(cube, 3, 2)
        expected = np.arange(2, 48, 3)
        assert_array_equal(result.data, expected)

    def test_resample_invalid(self):
        """Test average of a 1D field."""
        data = np.arange(0, 48, 1)
        times = np.arange(0, 48, 1)
        cube = self._create_cube(data, times)

        with self.assertRaises(ValueError):
            resample_hours(cube, 5)

    def test_resample_invalid_offset(self):
        """Test average of a 1D field."""
        data = np.arange(0, 48, 1)
        times = np.arange(0, 48, 1)
        cube = self._create_cube(data, times)

        with self.assertRaises(ValueError):
            resample_hours(cube, interval=3, offset=6)

    def test_resample_shorter_interval(self):
        """Test average of a 1D field."""
        data = np.arange(0, 48, 12)
        times = np.arange(0, 48, 12)
        cube = self._create_cube(data, times)

        with self.assertRaises(ValueError):
            resample_hours(cube, interval=3)

    def test_resample_same_interval(self):
        """Test average of a 1D field."""
        data = np.arange(0, 48, 12)
        times = np.arange(0, 48, 12)
        cube = self._create_cube(data, times)

        with self.assertRaises(ValueError):
            resample_hours(cube, interval=12)


class TestResampleTime(tests.Test):
    """Test :func:`esmvalcore.preprocessor._time.resample_hours`."""
    @staticmethod
    def _create_cube(data, times):
        time = iris.coords.DimCoord(times,
                                    standard_name='time',
                                    units=Unit('hours since 1950-01-01',
                                               calendar='360_day'))
        time.guess_bounds()
        cube = iris.cube.Cube(data, dim_coords_and_dims=[(time, 0)])
        return cube

    def test_resample_hourly_to_daily(self):
        """Test average of a 1D field."""
        data = np.arange(0, 48, 1)
        times = np.arange(0, 48, 1)
        cube = self._create_cube(data, times)

        result = resample_time(cube, hour=12)
        expected = np.arange(12, 48, 24)
        assert_array_equal(result.data, expected)

    def test_resample_hourly_to_monthly(self):
        """Test average of a 1D field."""
        data = np.arange(0, 24 * 60, 3)
        times = np.arange(0, 24 * 60, 3)
        cube = self._create_cube(data, times)

        result = resample_time(cube, hour=12, day=15)
        expected = np.array([12 + 14 * 24, 12 + 44 * 24])
        assert_array_equal(result.data, expected)

    def test_resample_daily_to_monthly(self):
        """Test average of a 1D field."""
        data = np.arange(0, 60 * 24, 24)
        times = np.arange(0, 60 * 24, 24)
        cube = self._create_cube(data, times)

        result = resample_time(cube, day=15)
        expected = np.array([
            14 * 24,
            44 * 24,
        ])
        assert_array_equal(result.data, expected)


if __name__ == '__main__':
    unittest.main()
