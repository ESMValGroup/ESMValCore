"""Unit tests for the CMORCheck class."""

import unittest
from copy import deepcopy

import iris
import iris.coord_categorisation
import iris.coords
import iris.cube
import iris.util
import numpy as np
from cf_units import Unit

from esmvalcore.cmor.check import CheckLevels, CMORCheck, CMORCheckError


class VariableInfoMock:
    """Mock for the variables definition."""

    def __init__(self):
        self.table_type = 'CMIP5'
        self.short_name = 'short_name'
        self.standard_name = 'age_of_sea_ice'  # Iris don't accept fakes ...
        self.long_name = 'Long Name'
        self.units = 'years'  # ... nor in the units
        self.valid_min = '0'
        self.valid_max = '100'
        self.frequency = 'day'
        self.positive = ''

        generic_level = CoordinateInfoMock('depth')
        generic_level.generic_level = True
        generic_level.axis = 'Z'

        requested = CoordinateInfoMock('air_pressure')
        requested.requested = [str(number) for number in range(20)]

        self.coordinates = {
            'time': CoordinateInfoMock('time'),
            'lat': CoordinateInfoMock('lat'),
            'lon': CoordinateInfoMock('lon'),
            'air_pressure': requested,
            'depth': generic_level,
        }


class CoordinateInfoMock:
    """Mock for the coordinates info."""

    def __init__(self, name):
        self.name = name
        self.generic_level = False

        self.axis = ""
        self.value = ""
        standard_names = {'lat': 'latitude', 'lon': 'longitude'}
        if name in standard_names:
            self.standard_name = standard_names[name]
        else:
            self.standard_name = name
        self.long_name = "Long name"
        self.out_name = self.name
        self.var_name = self.name

        units = {
            'lat': 'degrees_north',
            'lon': 'degrees_east',
            'time': 'days since 1950-01-01 00:00:00'
        }
        if name in units:
            self.units = units[name]
        else:
            self.units = "units"

        self.stored_direction = "increasing"
        self.must_have_bounds = "yes"
        self.requested = []
        self.generic_lev_coords = {}
        self.generic_lev_name = ""

        valid_limits = {'lat': ('-90', '90'), 'lon': ('0', '360')}
        if name in valid_limits:
            self.valid_min = valid_limits[name][0]
            self.valid_max = valid_limits[name][1]
        else:
            self.valid_min = ""
            self.valid_max = ""


class TestCMORCheck(unittest.TestCase):
    """Test CMORCheck class."""

    def setUp(self):
        """Prepare tests."""
        self.var_info = VariableInfoMock()
        self.cube = self.get_cube(self.var_info)

    def test_report_error(self):
        """Test report error function."""
        checker = CMORCheck(self.cube, self.var_info)
        self.assertFalse(checker.has_errors())
        checker.report_critical('New error: {}', 'something failed')
        self.assertTrue(checker.has_errors())

    def test_fail_on_error(self):
        """Test exception is raised if fail_on_error is activated."""
        checker = CMORCheck(self.cube, self.var_info, fail_on_error=True)
        with self.assertRaises(CMORCheckError):
            checker.report_critical('New error: {}', 'something failed')

    def test_report_warning(self):
        """Test report warning function."""
        checker = CMORCheck(self.cube, self.var_info)
        self.assertFalse(checker.has_errors())
        checker.report_warning('New error: {}', 'something failed')
        self.assertTrue(checker.has_warnings())

    def test_warning_fail_on_error(self):
        """Test report warning function with fail_on_error."""
        checker = CMORCheck(self.cube, self.var_info, fail_on_error=True)
        with self.assertLogs(level='WARNING') as cm:
            checker.report_warning('New error: {}', 'something failed')
            self.assertEqual(
                cm.output,
                ['WARNING:esmvalcore.cmor.check:New error: something failed', ]
            )

    def test_report_debug_message(self):
        """"Test report debug message function"""
        checker = CMORCheck(self.cube, self.var_info)
        self.assertFalse(checker.has_debug_messages())
        checker.report_debug_message('New debug message')
        self.assertTrue(checker.has_debug_messages())

    def test_check(self):
        """Test checks succeeds for a good cube."""
        self._check_cube()

    def _check_cube(self, automatic_fixes=False, frequency=None,
                    check_level=CheckLevels.DEFAULT):
        """Apply checks and optionally automatic fixes to self.cube."""
        def checker(cube):
            return CMORCheck(
                cube,
                self.var_info,
                automatic_fixes=automatic_fixes,
                frequency=frequency,
                check_level=check_level)

        self.cube = checker(self.cube).check_metadata()
        self.cube = checker(self.cube).check_data()

    def _check_cube_metadata(self, automatic_fixes=False, frequency=None,
                             check_level=CheckLevels.DEFAULT):
        """Apply checks and optionally automatic fixes to self.cube."""
        def checker(cube):
            return CMORCheck(
                cube,
                self.var_info,
                automatic_fixes=automatic_fixes,
                frequency=frequency,
                check_level=check_level)

        self.cube = checker(self.cube).check_metadata()

    def test_check_with_month_number(self):
        """Test checks succeeds for a good cube with month number."""
        iris.coord_categorisation.add_month_number(self.cube, 'time')
        self._check_cube()

    def test_check_with_day_of_month(self):
        """Test checks succeeds for a good cube with day of month."""
        iris.coord_categorisation.add_day_of_month(self.cube, 'time')
        self._check_cube()

    def test_check_with_day_of_year(self):
        """Test checks succeeds for a good cube with day of year."""
        iris.coord_categorisation.add_day_of_year(self.cube, 'time')
        self._check_cube()

    def test_check_with_year(self):
        """Test checks succeeds for a good cube with year."""
        iris.coord_categorisation.add_year(self.cube, 'time')
        self._check_cube()

    def test_check_with_time_attributes(self):
        """Test checks succeeds for a good cube with year."""
        self.cube.coord('time').attributes['time_origin'] = "cow"
        assert self.cube.coord('time').attributes['time_origin'] == "cow"
        self._check_cube()
        assert 'time_origin' not in self.cube.coord('time').attributes

    def test_check_no_multiple_coords_same_stdname(self):
        """Test checks fails if two coords have the same standard_name."""
        self.cube.add_aux_coord(
            iris.coords.AuxCoord(
                np.reshape(np.linspace(-90, 90, num=20*20), (20, 20)),
                var_name='bad_name',
                standard_name='latitude',
                units='degrees_north'
            ),
            (1, 2)
        )
        self._check_fails_in_metadata()

    def test_check_bad_standard_name_auto_fix(self):
        """Test check pass for a bad standard_name with automatic fixes."""
        self.cube.standard_name = 'wind_speed'
        self._check_cube(automatic_fixes=True)
        self._check_cube()

    def test_check_bad_standard_name(self):
        """Test check fails for a bad short_name."""
        self.cube.standard_name = 'wind_speed'
        self._check_fails_in_metadata(automatic_fixes=False)

    def test_check_bad_long_name_auto_fix(self):
        """Test check pass for a bad standard_name with automatic fixes."""
        self.cube.long_name = 'bad_name'
        self._check_cube(automatic_fixes=True)
        self._check_cube()

    def test_check_bad_long_name_auto_fix_report_warning(self):
        """Test check pass for a bad standard_name with automatic fixes."""
        self.cube.long_name = 'bad_name'
        self._check_warnings_on_metadata(automatic_fixes=True)

    def test_check_bad_long_name(self):
        """Test check fails for a bad short_name."""
        self.cube.long_name = 'bad_name'
        self._check_fails_in_metadata()

    def test_check_with_unit_conversion(self):
        """Test check succeeds for a good cube requiring unit conversion."""
        self.cube.units = 'days'
        self._check_cube()

    def test_check_with_psu_units(self):
        """Test check succeeds for a good cube with psu units."""
        self.var_info.units = 'psu'
        self.cube = self.get_cube(self.var_info)
        self._check_cube(automatic_fixes=True)

    def test_check_with_positive(self):
        """Check variable with positive attribute."""
        self.var_info.positive = 'up'
        self.cube = self.get_cube(self.var_info)
        self._check_cube()

    def test_check_with_no_positive_cmip5(self):
        """Check CMIP5 variable with no positive attribute report warning."""
        self.cube = self.get_cube(self.var_info)
        self.var_info.positive = 'up'
        self._check_warnings_on_metadata()

    def test_check_with_no_positive_cmip6(self):
        """Check CMIP6 variable with no positive attribute report warning."""
        self.var_info.positive = 'up'
        self.var_info.table_type = 'CMIP6'
        self._check_warnings_on_metadata()

    def test_invalid_rank(self):
        """Test check fails in metadata step when rank is not correct."""
        lat = iris.coords.AuxCoord.from_coord(self.cube.coord('latitude'))
        self.cube.remove_coord('latitude')
        self.cube.add_aux_coord(lat, self.cube.coord_dims('longitude'))
        self._check_fails_in_metadata()

    def test_rank_with_aux_coords(self):
        """Check succeeds even if a required coordinate is an aux coord."""
        iris.util.demote_dim_coord_to_aux_coord(self.cube, 'latitude')
        self._check_cube()

    def test_rank_with_scalar_coords(self):
        """Check succeeds even if a required coordinate is a scalar coord."""
        self.cube = self.cube.extract(
            iris.Constraint(time=self.cube.coord('time').cell(0)))
        self._check_cube()

    def test_rank_unstructured_grid(self):
        """Check succeeds even if two required coordinates share dimensions."""
        self.cube = self._get_unstructed_grid_cube()
        self._check_cube()

    def test_unstructured_grid_automatic_fixes(self):
        """Check succeeds even if two required coordinates share dimensions."""
        self.cube = self._get_unstructed_grid_cube(correct_lon=False,
                                                   n_bounds=3)

        lon_coord = self.cube.coord('longitude')
        np.testing.assert_allclose(lon_coord.points.min(), -180.0)
        np.testing.assert_allclose(lon_coord.points.max(), 162.0)
        np.testing.assert_allclose(lon_coord.bounds.min(), -180.0)
        np.testing.assert_allclose(lon_coord.bounds.max(), 171.0)

        self._check_cube(automatic_fixes=True)

        lon_coord = self.cube.coord('longitude')
        np.testing.assert_allclose(lon_coord.points.min(), 0.0)
        np.testing.assert_allclose(lon_coord.points.max(), 342.0)
        np.testing.assert_allclose(lon_coord.bounds.min(), 0.0)
        np.testing.assert_allclose(lon_coord.bounds.max(), 351.0)

    def test_bad_generic_level(self):
        """Test check fails in metadata if generic level coord
        has wrong var_name."""
        depth_coord = CoordinateInfoMock('depth')
        depth_coord.axis = 'Z'
        depth_coord.generic_lev_name = 'olevel'
        depth_coord.out_name = 'lev'
        depth_coord.name = 'depth_coord'
        depth_coord.long_name = 'ocean depth coordinate'
        self.var_info.coordinates['depth'].generic_lev_coords = {
            'depth_coord': depth_coord}
        self.var_info.coordinates['depth'].out_name = ""
        self._check_fails_in_metadata()

    def test_valid_generic_level(self):
        """Test valid generic level coordinate."""
        self._setup_generic_level_var()
        checker = CMORCheck(self.cube, self.var_info)
        checker.check_metadata()
        checker.check_data()

    def test_invalid_generic_level(self):
        """Test invalid generic level coordinate."""
        self._setup_generic_level_var()
        self.cube.remove_coord('atmosphere_sigma_coordinate')
        self._check_fails_in_metadata()

    def test_generic_level_alternative_cmip3(self):
        """Test valid alternative for generic level coords (CMIP3)."""
        self.var_info.table_type = 'CMIP3'
        self._setup_generic_level_var()
        self.var_info.coordinates['zlevel'] = self.var_info.coordinates.pop(
            'alevel')
        self._add_plev_to_cube()
        self._check_warnings_on_metadata()

    def test_generic_level_alternative_cmip5(self):
        """Test valid alternative for generic level coords (CMIP5)."""
        self.var_info.table_type = 'CMIP5'
        self._setup_generic_level_var()
        self._add_plev_to_cube()
        self._check_warnings_on_metadata()

    def test_generic_level_alternative_cmip6(self):
        """Test valid alternative for generic level coords (CMIP6)."""
        self.var_info.table_type = 'CMIP6'
        self._setup_generic_level_var()
        self._add_plev_to_cube()
        self._check_warnings_on_metadata()

    def test_generic_level_alternative_obs4mips(self):
        """Test valid alternative for generic level coords (obs4MIPs)."""
        self.var_info.table_type = 'obs4MIPs'
        self._setup_generic_level_var()
        self._add_plev_to_cube()
        self._check_warnings_on_metadata()

    def test_generic_level_invalid_alternative(self):
        """Test invalid alternative for generic level coords."""
        self.var_info.table_type = 'CMIP6'
        self._setup_generic_level_var()
        self._add_plev_to_cube()
        self.cube.coord('air_pressure').standard_name = 'altitude'
        self._check_fails_in_metadata()

    def test_check_bad_var_standard_name_strict_flag(self):
        """Test check fails for a bad variable standard_name with
        --cmor-check strict."""
        self.cube.standard_name = 'wind_speed'
        self._check_fails_in_metadata(automatic_fixes=False)

    def test_check_bad_var_long_name_strict_flag(self):
        """Test check fails for a bad variable long_name with
        --cmor-check strict."""
        self.cube.long_name = "Near-Surface Wind Speed"
        self._check_fails_in_metadata(automatic_fixes=False)

    def test_check_bad_var_units_strict_flag(self):
        """Test check fails for a bad variable units with
        --cmor-check strict."""
        self.cube.units = "kg"
        self._check_fails_in_metadata(automatic_fixes=False)

    def test_check_bad_attributes_strict_flag(self):
        """Test check fails for a bad variable attribute with
        --cmor-check strict."""
        self.var_info.standard_name = "surface_upward_latent_heat_flux"
        self.var_info.positive = "up"
        self.cube = self.get_cube(self.var_info)
        self.cube.attributes['positive'] = "Wrong attribute"
        self._check_fails_in_metadata(automatic_fixes=False)

    def test_check_bad_rank_strict_flag(self):
        """Test check fails for a bad variable rank with
        --cmor-check strict."""
        lat = iris.coords.AuxCoord.from_coord(self.cube.coord('latitude'))
        self.cube.remove_coord('latitude')
        self.cube.add_aux_coord(lat, self.cube.coord_dims('longitude'))
        self._check_fails_in_metadata(automatic_fixes=False)

    def test_check_bad_coord_var_name_strict_flag(self):
        """Test check fails for bad coord var_name with
        --cmor-check strict"""
        self.var_info.table_type = 'CMIP5'
        self.cube.coord('longitude').var_name = 'bad_name'
        self._check_fails_in_metadata(automatic_fixes=False)

    def test_check_missing_lon_strict_flag(self):
        """Test check fails for missing longitude with --cmor-check strict"""
        self.var_info.table_type = 'CMIP5'
        self.cube.remove_coord('longitude')
        self._check_fails_in_metadata(automatic_fixes=False)

    def test_check_missing_lat_strict_flag(self):
        """Test check fails for missing latitude with --cmor-check strict"""
        self.var_info.table_type = 'CMIP5'
        self.cube.remove_coord('latitude')
        self._check_fails_in_metadata(automatic_fixes=False)

    def test_check_missing_time_strict_flag(self):
        """Test check fails for missing time with --cmor-check strict"""
        self.var_info.table_type = 'CMIP5'
        self.cube.remove_coord('time')
        self._check_fails_in_metadata(automatic_fixes=False)

    def test_check_missing_coord_strict_flag(self):
        """Test check fails for missing coord other than lat and lon
         with --cmor-check strict"""
        self.var_info.coordinates.update(
            {'height2m': CoordinateInfoMock('height2m')}
        )
        self._check_fails_in_metadata(automatic_fixes=False)

    def test_check_bad_var_standard_name_relaxed_flag(self):
        """Test check reports warning for a bad variable standard_name with
        --cmor-check relaxed."""
        self.cube.standard_name = 'wind_speed'
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.RELAXED)

    def test_check_bad_var_long_name_relaxed_flag(self):
        """Test check reports warning for a bad variable long_name with
        --cmor-check relaxed."""
        self.cube.long_name = "Near-Surface Wind Speed"
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.RELAXED)

    def test_check_bad_var_units_relaxed_flag(self):
        """Test check reports warning for a bad variable units with
        --cmor-check relaxed."""
        self.cube.units = "kg"
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.RELAXED)

    def test_check_bad_attributes_relaxed_flag(self):
        """Test check report warnings for a bad variable attribute with
        --cmor-check relaxed."""
        self.var_info.standard_name = "surface_upward_latent_heat_flux"
        self.var_info.positive = "up"
        self.cube = self.get_cube(self.var_info)
        self.cube.attributes['positive'] = "Wrong attribute"
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.RELAXED)

    def test_check_bad_rank_relaxed_flag(self):
        """Test check report warnings for a bad variable rank with
        --cmor-check relaxed."""
        lat = iris.coords.AuxCoord.from_coord(self.cube.coord('latitude'))
        self.cube.remove_coord('latitude')
        self.cube.add_aux_coord(lat, self.cube.coord_dims('longitude'))
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.RELAXED)

    def test_check_bad_coord_standard_name_relaxed_flag(self):
        """Test check reports warning for bad coord var_name with
        --cmor-check relaxed"""
        self.var_info.table_type = 'CMIP5'
        self.cube.coord('longitude').var_name = 'bad_name'
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.RELAXED)

    def test_check_missing_lon_relaxed_flag(self):
        """Test check fails for missing longitude with --cmor-check relaxed"""
        self.var_info.table_type = 'CMIP5'
        self.cube.remove_coord('longitude')
        self._check_fails_in_metadata(automatic_fixes=False,
                                      check_level=CheckLevels.RELAXED)

    def test_check_missing_lat_relaxed_flag(self):
        """Test check fails for missing latitude with --cmor-check relaxed"""
        self.var_info.table_type = 'CMIP5'
        self.cube.remove_coord('latitude')
        self._check_fails_in_metadata(automatic_fixes=False,
                                      check_level=CheckLevels.RELAXED)

    def test_check_missing_time_relaxed_flag(self):
        """Test check fails for missing latitude with --cmor-check relaxed"""
        self.var_info.table_type = 'CMIP5'
        self.cube.remove_coord('time')
        self._check_fails_in_metadata(automatic_fixes=False,
                                      check_level=CheckLevels.RELAXED)

    def test_check_missing_coord_relaxed_flag(self):
        """Test check reports warning for missing coord other than lat and lon
        with --cmor-check relaxed"""
        self.var_info.coordinates.update(
            {'height2m': CoordinateInfoMock('height2m')}
        )
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.RELAXED)

    def test_check_bad_var_standard_name_none_flag(self):
        """Test check reports warning for a bad variable standard_name with
        --cmor-check ignore."""
        self.cube.standard_name = 'wind_speed'
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.IGNORE)

    def test_check_bad_var_long_name_none_flag(self):
        """Test check reports warning for a bad variable long_name with
        --cmor-check ignore."""
        self.cube.long_name = "Near-Surface Wind Speed"
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.IGNORE)

    def test_check_bad_var_units_none_flag(self):
        """Test check reports warning for a bad variable unit with
        --cmor-check ignore."""
        self.cube.units = "kg"
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.IGNORE)

    def test_check_bad_attributes_none_flag(self):
        """Test check reports warning for a bad variable attribute with
        --cmor-check ignore."""
        self.var_info.standard_name = "surface_upward_latent_heat_flux"
        self.var_info.positive = "up"
        self.cube = self.get_cube(self.var_info)
        self.cube.attributes['positive'] = "Wrong attribute"
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.IGNORE)

    def test_check_bad_rank_none_flag(self):
        """Test check reports warning for a bad variable rank with
        --cmor-check ignore."""
        lat = iris.coords.AuxCoord.from_coord(self.cube.coord('latitude'))
        self.cube.remove_coord('latitude')
        self.cube.add_aux_coord(lat, self.cube.coord_dims('longitude'))
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.IGNORE)

    def test_check_bad_coord_standard_name_none_flag(self):
        """Test check reports warning for bad coord var_name with
        --cmor-check ignore."""
        self.var_info.table_type = 'CMIP5'
        self.cube.coord('longitude').var_name = 'bad_name'
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.IGNORE)

    def test_check_missing_lon_none_flag(self):
        """Test check reports warning for missing longitude with
        --cmor-check ignore"""
        self.var_info.table_type = 'CMIP5'
        self.cube.remove_coord('longitude')
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.IGNORE)

    def test_check_missing_lat_none_flag(self):
        """Test check reports warning for missing latitude with
        --cmor-check ignore"""
        self.var_info.table_type = 'CMIP5'
        self.cube.remove_coord('latitude')
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.IGNORE)

    def test_check_missing_time_none_flag(self):
        """Test check reports warning for missing time
        with --cmor-check ignore"""
        self.var_info.table_type = 'CMIP5'
        self.cube.remove_coord('time')
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.IGNORE)

    def test_check_missing_coord_none_flag(self):
        """Test check reports warning for missing coord other than lat, lon and
        time with --cmor-check ignore"""
        self.var_info.coordinates.update(
            {'height2m': CoordinateInfoMock('height2m')}
        )
        self._check_warnings_on_metadata(automatic_fixes=False,
                                         check_level=CheckLevels.IGNORE)

    def _check_fails_in_metadata(self, automatic_fixes=False, frequency=None,
                                 check_level=CheckLevels.DEFAULT):
        checker = CMORCheck(
            self.cube,
            self.var_info,
            automatic_fixes=automatic_fixes,
            frequency=frequency,
            check_level=check_level)
        with self.assertRaises(CMORCheckError):
            checker.check_metadata()

    def _check_warnings_on_metadata(self, automatic_fixes=False,
                                    check_level=CheckLevels.DEFAULT):
        checker = CMORCheck(
            self.cube, self.var_info, automatic_fixes=automatic_fixes,
            check_level=check_level
        )
        checker.check_metadata()
        self.assertTrue(checker.has_warnings())

    def _check_debug_messages_on_metadata(self, automatic_fixes=False):
        checker = CMORCheck(
            self.cube, self.var_info, automatic_fixes=automatic_fixes)
        checker.check_metadata()
        self.assertTrue(checker.has_debug_messages())

    def test_non_requested(self):
        """
        Warning if requested values are not present.

        Check issue a warning if a values requested
        for a coordinate are not correct in the metadata step
        """
        coord = self.cube.coord('air_pressure')
        values = np.linspace(0, 40, len(coord.points))
        self._update_coordinate_values(self.cube, coord, values)
        checker = CMORCheck(self.cube, self.var_info)
        checker.check_metadata()
        self.assertTrue(checker.has_warnings())

    def test_almost_requested(self):
        """Automatically fix tiny deviations from the requested values."""
        coord = self.cube.coord('air_pressure')
        values = np.array(coord.points)
        values[0] += 1.e-7
        values[-1] += 1.e-7
        self._update_coordinate_values(self.cube, coord, values)
        checker = CMORCheck(self.cube, self.var_info, automatic_fixes=True)
        checker.check_metadata()
        reference = np.array(
            self.var_info.coordinates['air_pressure'].requested, dtype=float)
        np.testing.assert_array_equal(
            self.cube.coord('air_pressure').points, reference)

    def test_requested_str_values(self):
        """
        Warning if requested values are not present.

        Check issue a warning if a values requested
        for a coordinate are not correct in the metadata step
        """
        region_coord = CoordinateInfoMock('basin')
        region_coord.standard_name = 'region'
        region_coord.units = ''
        region_coord.requested = [
            "atlantic_arctic_ocean",
            "indian_pacific_ocean",
            "global_ocean"
        ]
        self.var_info.coordinates['region'] = region_coord
        self.cube = self.get_cube(self.var_info)
        self._check_cube()

    def test_requested_non_1d(self):
        """Warning if requested values in non-1d cannot be checked."""
        coord = self.cube.coord('air_pressure')
        values = np.linspace(0, 40, len(coord.points))
        values = np.broadcast_to(values, (20, 20))
        bounds = np.moveaxis(np.stack((values - 0.01, values + 0.01)), 0, -1)
        new_plev_coord = iris.coords.AuxCoord(
            values, bounds=bounds, var_name=coord.var_name,
            standard_name=coord.standard_name, long_name=coord.long_name,
            units=coord.units)
        self.cube.remove_coord('air_pressure')
        self.cube.add_aux_coord(new_plev_coord, (2, 3))
        checker = CMORCheck(self.cube, self.var_info)
        checker.check_metadata()
        self.assertFalse(checker.has_debug_messages())
        self.assertTrue(checker.has_warnings())

    def test_non_increasing(self):
        """Fail in metadata if increasing coordinate is decreasing."""
        coord = self.cube.coord('latitude')
        values = np.linspace(
            coord.points[-1],
            coord.points[0],
            len(coord.points)
        )
        self._update_coordinate_values(self.cube, coord, values)
        self._check_fails_in_metadata()

    def test_non_increasing_fix(self):
        """Check automatic fix for direction."""
        coord = self.cube.coord('latitude')
        values = np.linspace(
            coord.points[-1],
            coord.points[0],
            len(coord.points)
        )
        self._update_coordinate_values(self.cube, coord, values)
        self._check_cube(automatic_fixes=True)
        self._check_cube()
        # test bounds are contiguous
        bounds = self.cube.coord('latitude').bounds
        right_bounds = bounds[:-2, 1]
        left_bounds = bounds[1:-1, 0]
        self.assertTrue(np.all(left_bounds == right_bounds))

    def test_non_decreasing(self):
        """Fail in metadata if decreasing coordinate is increasing."""
        self.var_info.coordinates['lat'].stored_direction = 'decreasing'
        self._check_fails_in_metadata()

    def test_non_decreasing_fix(self):
        """Check automatic fix for non decreasing coordinate."""
        self.cube.data[0, 0, 0, 0, 0] = 70
        self.var_info.coordinates['lat'].stored_direction = 'decreasing'
        self._check_cube(automatic_fixes=True)
        self._check_cube()
        index = [0, 0, 0, 0, 0]
        index[self.cube.coord_dims('latitude')[0]] = -1
        self.assertEqual(self.cube.data.item(tuple(index)), 70)
        self.assertEqual(self.cube.data[0, 0, 0, 0, 0], 50)
        cube_points = self.cube.coord('latitude').points
        reference = np.linspace(90, -90, 20, endpoint=True)
        for index in range(20):
            self.assertTrue(
                iris.util.approx_equal(cube_points[index], reference[index]))
        # test bounds are contiguous
        bounds = self.cube.coord('latitude').bounds
        right_bounds = bounds[:-2, 1]
        left_bounds = bounds[1:-1, 0]
        self.assertTrue(np.all(left_bounds == right_bounds))

    def test_not_bounds(self):
        """Warning if bounds are not available."""
        self.cube.coord('longitude').bounds = None
        self._check_warnings_on_metadata(automatic_fixes=False)
        self.assertFalse(self.cube.coord('longitude').has_bounds())

    def test_guess_bounds_with_fixes(self):
        """Warning if bounds added with automatic fixes."""
        self.cube.coord('longitude').bounds = None
        self._check_warnings_on_metadata(automatic_fixes=True)
        self.assertTrue(self.cube.coord('longitude').has_bounds())

    def test_not_guess_bounds_with_fixes(self):
        """Warning if bounds added with automatic fixes."""
        self.cube.coord('longitude').bounds = None
        self.var_info.coordinates['lon'].must_have_bounds = "no"
        self._check_cube(automatic_fixes=True)
        self.assertFalse(self.cube.coord('longitude').has_bounds())

    def test_not_correct_lons(self):
        """Fail if longitudes are not correct in metadata step."""
        self.cube = self.cube.intersection(longitude=(-180., 180.))
        self._check_fails_in_metadata()

    def test_lons_automatic_fix(self):
        """Test automatic fixes for bad longitudes."""
        self.cube = self.cube.intersection(longitude=(-180., 180.))
        self._check_cube(automatic_fixes=True)

    def test_lons_automatic_fix_with_bounds(self):
        """Test automatic fixes for bad longitudes with added bounds."""
        self.cube.coord('longitude').bounds = None
        self.cube = self.cube.intersection(longitude=(-180., 180.))
        self._check_cube(automatic_fixes=True)
        self.assertTrue(self.cube.coord('longitude').points.min() >= 0.)
        self.assertTrue(self.cube.coord('longitude').points.max() <= 360.)
        self.assertTrue(self.cube.coord('longitude').has_bounds())

    def test_high_lons_automatic_fix(self):
        """Test automatic fixes for high longitudes."""
        self.cube = self.cube.intersection(longitude=(180., 520.))
        self._check_cube(automatic_fixes=True)

    def test_too_high_lons_automatic_fix_not_done(self):
        """Test automatic fixes for bad longitudes."""
        self.cube = self.cube.intersection(longitude=(720., 1080.))
        self._check_fails_in_metadata(automatic_fixes=True)

    def test_too_low_lons_automatic_fix_not_done(self):
        """Test automatic fixes for bad longitudes."""
        self.cube = self.cube.intersection(longitude=(-720., -360.))
        self._check_fails_in_metadata(automatic_fixes=True)

    def test_not_valid_min(self):
        """Fail if coordinate values below valid_min."""
        coord = self.cube.coord('latitude')
        values = np.linspace(
            coord.points[0] - 1,
            coord.points[-1],
            len(coord.points)
        )
        self._update_coordinate_values(self.cube, coord, values)
        self._check_fails_in_metadata()

    def test_not_valid_max(self):
        """Fail if coordinate values above valid_max."""
        coord = self.cube.coord('latitude')
        values = np.linspace(
            coord.points[0],
            coord.points[-1] + 1,
            len(coord.points)
        )
        self._update_coordinate_values(self.cube, coord, values)
        self._check_fails_in_metadata()

    @staticmethod
    def _update_coordinate_values(cube, coord, values):
        [dimension] = cube.coord_dims(coord)
        cube.remove_coord(coord)
        new_coord = iris.coords.DimCoord(
            values,
            standard_name=coord.standard_name,
            long_name=coord.long_name,
            var_name=coord.var_name,
            units=coord.units)
        cube.add_dim_coord(new_coord, dimension)

    def test_bad_units(self):
        """Fail if coordinates have bad units."""
        self.cube.coord('latitude').units = 'degrees_n'
        self._check_fails_in_metadata()

    def test_units_automatic_fix(self):
        """Test automatic fix for bad coordinate units."""
        self.cube.coord('latitude').units = 'degrees_n'
        self._check_cube(automatic_fixes=True)

    def test_units_automatic_fix_failed(self):
        """Test automatic fix fail for incompatible coordinate units."""
        self.cube.coord('latitude').units = 'degC'
        self._check_fails_in_metadata(automatic_fixes=True)

    def test_bad_time(self):
        """Fail if time have bad units."""
        self.cube.coord('time').units = 'days'
        self._check_fails_in_metadata()

    def test_time_units_automatic_fix(self):
        """Test automatic fix for time units."""
        self.cube.coord('time').units = 'days since 1860-1-1 00:00:00'
        self.cube.attributes['parent_time_units'] = 'days since ' \
                                                    '1850-1-1 00:00:00'
        self.cube.attributes['branch_time_in_parent'] = 0.
        self.cube.attributes['branch_time_in_child'] = 0.
        self._check_cube()
        assert (self.cube.coord('time').units.origin ==
                'days since 1850-1-1 00:00:00')
        assert self.cube.attributes['parent_time_units'] == 'days since ' \
                                                            '1850-1-1 00:00:00'
        assert self.cube.attributes['branch_time_in_parent'] == 0.
        assert self.cube.attributes['branch_time_in_child'] == 3652.

    def test_time_units_no_parent_exp(self):
        """Test non-conversion of time units attributes for no parent exps."""
        self.cube.coord('time').units = 'days since 1860-1-1 00:00:00'
        self.cube.attributes['parent_time_units'] = 'no parent'
        self.cube.attributes['branch_time_in_parent'] = 0.
        self.cube.attributes['branch_time_in_child'] = 0.
        self._check_cube()
        assert (self.cube.coord('time').units.origin ==
                'days since 1850-1-1 00:00:00')
        assert self.cube.attributes['parent_time_units'] == 'no parent'
        assert self.cube.attributes['branch_time_in_parent'] == 0.
        assert self.cube.attributes['branch_time_in_child'] == 0.

    def test_wrong_parent_time_unit(self):
        """Test fail for wrong parent time units."""
        self.cube.coord('time').units = 'days since 1860-1-1 00:00:00'
        self.cube.attributes['parent_time_units'] = 'days since ' \
                                                    '1860-1-1-00-00-00'
        self.cube.attributes['branch_time_in_parent'] = 0.
        self.cube.attributes['branch_time_in_child'] = 0.
        self._check_warnings_on_metadata()
        assert self.cube.attributes['branch_time_in_parent'] == 0.
        assert self.cube.attributes['branch_time_in_child'] == 0

    def test_time_automatic_fix_failed(self):
        """Test automatic fix fail for incompatible time units."""
        self.cube.coord('time').units = 'K'
        self._check_fails_in_metadata(automatic_fixes=True)

    def test_time_non_monotonic(self):
        """Test automatic fix fail for non monotonic times."""
        time = self.cube.coord('time')
        points = np.array(time.points)
        points[-1] = points[0]
        dims = self.cube.coord_dims(time)
        self.cube.remove_coord(time)
        time = iris.coords.AuxCoord.from_coord(time)
        self.cube.add_aux_coord(time.copy(points), dims)
        self._check_fails_in_metadata(automatic_fixes=True)

    def test_bad_standard_name(self):
        """Fail if coordinates have bad standard names at metadata step."""
        self.cube.coord('time').standard_name = 'region'
        self._check_fails_in_metadata()

    def test_bad_out_name_multidim_latitude(self):
        """Warning if multidimensional lat has bad var_name at metadata"""
        self.var_info.table_type = 'CMIP6'
        self.cube.remove_coord('latitude')
        self.cube.add_aux_coord(
            iris.coords.AuxCoord(
                np.reshape(np.linspace(-90, 90, num=20*20), (20, 20)),
                var_name='bad_name',
                standard_name='latitude',
                units='degrees_north'
            ),
            (1, 2)
        )
        self._check_debug_messages_on_metadata()

    def test_bad_out_name_multidim_longitude(self):
        """Warning if multidimensional lon has bad var_name at metadata"""
        self.var_info.table_type = 'CMIP6'
        self.cube.remove_coord('longitude')
        self.cube.add_aux_coord(
            iris.coords.AuxCoord(
                np.reshape(np.linspace(-90, 90, num=20*20), (20, 20)),
                var_name='bad_name',
                standard_name='longitude',
                units='degrees_east'
            ),
            (1, 2)
        )
        self._check_debug_messages_on_metadata(automatic_fixes=True)

    def test_bad_bounds_in_multidim_longitude(self):
        """Warning if multidimensional lon has bad var_name at metadata"""
        self.var_info.table_type = 'CMIP6'
        self.cube.remove_coord('longitude')
        lons = np.reshape(np.linspace(180, 540, num=20*20), (20, 20))
        bounds = np.stack([lons.copy() - 0.5, lons.copy() + 0.5], -1)
        self.cube.add_aux_coord(
            iris.coords.AuxCoord(
                lons,
                var_name='bad_name',
                standard_name='longitude',
                units='degrees_east',
                bounds=bounds,
            ),
            (1, 2)
        )
        self._check_debug_messages_on_metadata(automatic_fixes=True)

    def test_bad_out_name_region_area_type(self):
        """Debug message if region/area_type AuxCoord has bad var_name at
        metadata."""
        region_coord = CoordinateInfoMock('basin')
        region_coord.standard_name = 'region'
        self.var_info.coordinates['region'] = region_coord

        self.cube = self.get_cube(self.var_info)
        self.cube.coord("region").var_name = 'sector'

        self._check_debug_messages_on_metadata()

    def test_bad_out_name_onedim_latitude(self):
        """Warning if onedimensional lat has bad var_name at metadata"""
        self.var_info.table_type = 'CMIP6'
        self.cube.coord('latitude').var_name = 'bad_name'
        self._check_fails_in_metadata()

    def test_bad_out_name_onedim_longitude(self):
        """Warning if onedimensional lon has bad var_name at metadata"""
        self.var_info.table_type = 'CMIP6'
        self.cube.coord('longitude').var_name = 'bad_name'
        self._check_fails_in_metadata()

    def test_bad_out_name_other(self):
        """Warning if general coordinate has bad var_name at metadata"""
        self.var_info.table_type = 'CMIP6'
        self.cube.coord('time').var_name = 'bad_name'
        self._check_fails_in_metadata()

    def test_bad_out_name(self):
        """Fail if coordinates have bad short names at metadata step."""
        self.cube.coord('latitude').var_name = 'region'
        self._check_fails_in_metadata()

    def test_bad_data_units(self):
        """Fail if data has bad units at metadata step."""
        self.cube.units = 'hPa'
        self._check_fails_in_metadata()

    def test_bad_positive(self):
        """Fail if positive value is incorrect at metadata step."""
        self.cube.attributes['positive'] = 'up'
        self.var_info.positive = 'down'
        self._check_fails_in_metadata()

    def test_bad_standard_name_genlevel(self):
        """Check if generic level has a different."""
        self.cube.coord('depth').standard_name = None
        self._check_cube()

    def test_frequency_month_not_same_day(self):
        """Fail at metadata if frequency (day) not matches data frequency."""
        self.cube = self.get_cube(self.var_info, frequency='mon')
        time = self.cube.coord('time')
        points = np.array(time.points)
        points[1] = points[1] + 12
        dims = self.cube.coord_dims(time)
        self.cube.remove_coord(time)
        self.cube.add_dim_coord(time.copy(points), dims)
        self._check_cube(frequency='mon')

    def test_check_pt_freq(self):
        """Test checks succeeds for a good Pt frequency."""
        self.var_info.frequency = 'dayPt'
        self._check_cube()

    def test_check_pt_lowercase_freq(self):
        """Test checks succeeds for a good Pt frequency."""
        self.var_info.frequency = 'daypt'
        self._check_cube()

    def test_bad_frequency_day(self):
        """Fail at metadata if frequency (day) not matches data frequency."""
        self.cube = self.get_cube(self.var_info, frequency='mon')
        self._check_fails_in_metadata(frequency='day')

    def test_bad_frequency_subhr(self):
        """Fail at metadata if frequency (subhr) not matches data frequency."""
        self._check_fails_in_metadata(frequency='subhr')

    def test_bad_frequency_dec(self):
        """Fail at metadata if frequency (dec) not matches data frequency."""
        self._check_fails_in_metadata(frequency='d')

    def test_bad_frequency_yr(self):
        """Fail at metadata if frequency (yr) not matches data frequency."""
        self._check_fails_in_metadata(frequency='yr')

    def test_bad_frequency_mon(self):
        """Fail at metadata if frequency (mon) not matches data frequency."""
        self._check_fails_in_metadata(frequency='mon')

    def test_bad_frequency_hourly(self):
        """Fail at metadata if frequency (3hr) not matches data frequency."""
        self._check_fails_in_metadata(frequency='3hr')

    def test_frequency_not_supported(self):
        """Fail at metadata if frequency is not supported."""
        self._check_fails_in_metadata(frequency='wrong_freq')

    def test_time_bounds(self):
        """Test time bounds are guessed for all frequencies"""
        freqs = ['hr', '3hr', '6hr', 'day', 'mon', 'yr', 'dec']
        guessed = []
        for freq in freqs:
            cube = self.get_cube(self.var_info, frequency=freq)
            cube.coord('time').bounds = None
            self.cube = cube
            self._check_cube(automatic_fixes=True, frequency=freq)
            if cube.coord('time').bounds is not None:
                guessed.append(True)
        assert len(guessed) == len(freqs)

    def test_no_time_bounds(self):
        """Test time bounds are not guessed for instantaneous data"""
        self.var_info.coordinates['time'].must_have_bounds = 'no'
        self.var_info.coordinates['time1'] = (
            self.var_info.coordinates.pop('time'))
        self.cube.coord('time').bounds = None
        self._check_cube(automatic_fixes=True)
        guessed_bounds = self.cube.coord('time').bounds
        assert guessed_bounds is None

    def _check_fails_on_data(self):
        checker = CMORCheck(self.cube, self.var_info)
        checker.check_metadata()
        with self.assertRaises(CMORCheckError):
            checker.check_data()

    def _check_warnings_on_data(self):
        checker = CMORCheck(self.cube, self.var_info)
        checker.check_metadata()
        checker.check_data()
        self.assertTrue(checker.has_warnings())

    def get_cube(self,
                 var_info,
                 set_time_units="days since 1850-1-1 00:00:00",
                 frequency=None):
        """
        Create a cube based on a specification.

        Parameters
        ----------
        var_info:
            variable specification
        set_time_units: str
            units for the time coordinate
        frequency: None or str
            frequency of the generated data

        Returns
        -------
        iris.cube.Cube

        """
        coords = []
        scalar_coords = []
        index = 0
        if not frequency:
            frequency = var_info.frequency
        for dim_spec in var_info.coordinates.values():
            coord = self._create_coord_from_spec(
                dim_spec, set_time_units, frequency)
            if dim_spec.value:
                scalar_coords.append(coord)
            else:
                coords.append((coord, index))
                index += 1

        valid_min, valid_max = self._get_valid_limits(var_info)
        var_data = (np.ones(len(coords) * [20], 'f') *
                    (valid_min + (valid_max - valid_min) / 2))

        if var_info.units == 'psu':
            units = None
            attributes = {'invalid_units': 'psu'}
        else:
            units = var_info.units
            attributes = None

        cube = iris.cube.Cube(
            var_data,
            standard_name=var_info.standard_name,
            long_name=var_info.long_name,
            var_name=var_info.short_name,
            units=units,
            attributes=attributes,
        )
        if var_info.positive:
            cube.attributes['positive'] = var_info.positive

        for coord, i in coords:
            if isinstance(coord, iris.coords.DimCoord):
                cube.add_dim_coord(coord, i)
            else:
                cube.add_aux_coord(coord, i)

        for coord in scalar_coords:
            cube.add_aux_coord(coord)

        return cube

    def _get_unstructed_grid_cube(self, correct_lon=True, n_bounds=2):
        """Get cube with unstructured grid."""
        assert n_bounds in (2, 3), "Only 2 or 3 bounds per cell supported"

        cube = self.get_cube(self.var_info)
        cube = cube.extract(
            iris.Constraint(latitude=cube.coord('latitude').points[0]))
        lat_points = cube.coord('longitude').points
        lat_points = lat_points / 3.0 - 50.
        cube.remove_coord('latitude')
        iris.util.demote_dim_coord_to_aux_coord(cube, 'longitude')
        lat_points = np.concatenate(
            (
                cube.coord('longitude').points[0:10] / 4,
                cube.coord('longitude').points[0:10] / 4
            ),
            axis=0
        )
        lat_bounds = np.concatenate(
            (
                cube.coord('longitude').bounds[0:10] / 4,
                cube.coord('longitude').bounds[0:10] / 4
            ),
            axis=0
        )
        new_lat = iris.coords.AuxCoord(
            points=lat_points,
            bounds=lat_bounds,
            var_name='lat',
            standard_name='latitude',
            long_name='Latitude',
            units='degrees_north',
        )
        cube.add_aux_coord(new_lat, 1)

        # Add additional bound if desired
        if n_bounds == 3:
            for coord_name in ('latitude', 'longitude'):
                coord = cube.coord(coord_name)
                new_bounds = np.stack((
                    coord.bounds[:, 0],
                    0.5 * (coord.bounds[:, 0] + coord.bounds[:, 1]),
                    coord.bounds[:, 1],
                ))
                coord.bounds = np.swapaxes(new_bounds, 0, 1)

        # Convert longitude to [-180, 180] to check automatic fixes if desired
        if correct_lon:
            return cube
        new_lon = iris.coords.AuxCoord(
            points=cube.coord('longitude').points,
            bounds=cube.coord('longitude').bounds,
            var_name='lon',
            standard_name='longitude',
            long_name='Longitude',
            units='degrees_east',
        )
        new_lon.points = np.where(new_lon.points < 180.0, new_lon.points,
                                  new_lon.points - 360.0)
        new_lon.bounds = np.where(new_lon.bounds < 180.0, new_lon.bounds,
                                  new_lon.bounds - 360.0)
        cube.remove_coord('longitude')
        cube.add_aux_coord(new_lon, 1)
        return cube

    def _setup_generic_level_var(self):
        """Setup var_info and cube with generic alevel coordinate."""
        self.var_info.coordinates.pop('depth')
        self.var_info.coordinates.pop('air_pressure')

        # Create cube with sigma coordinate
        sigma_coord = CoordinateInfoMock('standard_sigma')
        sigma_coord.axis = 'Z'
        sigma_coord.out_name = 'lev'
        sigma_coord.standard_name = 'atmosphere_sigma_coordinate'
        sigma_coord.long_name = 'sigma coordinate'
        sigma_coord.generic_lev_name = 'alevel'
        var_info_for_cube = deepcopy(self.var_info)
        var_info_for_cube.coordinates['standard_sigma'] = sigma_coord
        self.cube = self.get_cube(var_info_for_cube)

        # Create var_info with alevel coord that contains sigma coordinate in
        # generic_lev_coords dict (just like it is the case for the true CMOR
        # tables)
        gen_lev_coord = CoordinateInfoMock('alevel')
        gen_lev_coord.standard_name = None
        gen_lev_coord.generic_level = True
        gen_lev_coord.generic_lev_coords = {'standard_sigma': sigma_coord}
        self.var_info.coordinates['alevel'] = gen_lev_coord

    def _add_plev_to_cube(self):
        """Add plev coordinate to cube."""
        if self.cube.coords('atmosphere_sigma_coordinate'):
            self.cube.remove_coord('atmosphere_sigma_coordinate')
        plevs = [100000.0, 92500.0, 85000.0, 70000.0, 60000.0, 50000.0,
                 40000.0, 30000.0, 25000.0, 20000.0, 15000.0, 10000.0, 7000.0,
                 5000.0, 3000.0, 2000.0, 1000.0, 900.0, 800.0, 700.0]
        coord = iris.coords.DimCoord(
            plevs,
            var_name='plev',
            standard_name='air_pressure',
            units='Pa',
            attributes={'positive': 'down'},
        )
        coord.guess_bounds()
        self.cube.add_dim_coord(coord, 3)

    def _get_valid_limits(self, var_info):
        if var_info.valid_min:
            valid_min = float(var_info.valid_min)
        else:
            valid_min = 0

        if var_info.valid_max:
            valid_max = float(var_info.valid_max)
        else:
            valid_max = valid_min + 100
        return valid_min, valid_max

    @staticmethod
    def _construct_scalar_coord(coord_spec):
        return iris.coords.AuxCoord(
            coord_spec.value,
            standard_name=coord_spec.standard_name,
            long_name=coord_spec.long_name,
            var_name=coord_spec.out_name,
            units=coord_spec.units,
            attributes=None)

    def _create_coord_from_spec(self, coord_spec, set_time_units, frequency):
        if coord_spec.units.startswith("days since "):
            coord_spec.units = set_time_units
        coord_spec.frequency = frequency

        if coord_spec.value:
            return self._construct_scalar_coord(coord_spec)

        if coord_spec.requested:
            try:
                float(coord_spec.requested[0])
            except ValueError:
                return self._construct_array_coord(coord_spec, aux=True)
        return self._construct_array_coord(coord_spec)

    def _construct_array_coord(self, dim_spec, aux=False):
        if dim_spec.units.startswith("days since "):
            values = self._get_time_values(dim_spec)
            unit = Unit(dim_spec.units, calendar='360_day')
        else:
            values = self._get_values(dim_spec)
            unit = Unit(dim_spec.units)
        # Set up attributes dictionary
        coord_atts = {'stored_direction': dim_spec.stored_direction}
        if aux:
            coord = iris.coords.AuxCoord(
                values,
                standard_name=dim_spec.standard_name,
                long_name=dim_spec.long_name,
                var_name=dim_spec.out_name,
                attributes=coord_atts,
                units=unit,
            )
        else:
            coord = iris.coords.DimCoord(
                values,
                standard_name=dim_spec.standard_name,
                long_name=dim_spec.long_name,
                var_name=dim_spec.out_name,
                attributes=coord_atts,
                units=unit,
            )
            coord.guess_bounds()
        return coord

    @staticmethod
    def _get_values(dim_spec):
        if dim_spec.requested:
            try:
                float(dim_spec.requested[0])
            except ValueError:
                return dim_spec.requested + [
                    f'Value{x}' for x in range(len(dim_spec.requested), 20)]
        valid_min = dim_spec.valid_min
        if valid_min:
            valid_min = float(valid_min)
        else:
            valid_min = 0.0
        valid_max = dim_spec.valid_max
        if valid_max:
            valid_max = float(valid_max)
        else:
            valid_max = 100.0
        decreasing = dim_spec.stored_direction == 'decreasing'
        endpoint = not dim_spec.standard_name == 'longitude'
        if decreasing:
            values = np.linspace(
                valid_max, valid_min, 20, endpoint=endpoint)
        else:
            values = np.linspace(
                valid_min, valid_max, 20, endpoint=endpoint)
        values = np.array(values)
        if dim_spec.requested:
            requested = [float(val) for val in dim_spec.requested]
            requested.sort(reverse=decreasing)
            for j, request in enumerate(requested):
                values[j] = request
            if decreasing:
                extra_values = np.linspace(
                    len(requested), valid_min, 20 - len(requested))
            else:
                extra_values = np.linspace(
                    len(requested), valid_max, 20 - len(requested))

            for j in range(len(requested), 20):
                values[j] = extra_values[j - len(requested)]

        return values

    @staticmethod
    def _get_time_values(dim_spec):
        frequency = dim_spec.frequency
        if frequency == 'mon':
            delta = 30
        elif frequency == 'day':
            delta = 1
        elif frequency == 'yr':
            delta = 360
        elif frequency == 'dec':
            delta = 3600
        elif frequency.endswith('hr'):
            if frequency == 'hr':
                frequency = '1hr'
            delta = float(frequency[:-2]) / 24
        else:
            raise Exception('Frequency {} not supported'.format(frequency))
        start = 0
        end = start + delta * 20
        return np.arange(start, end, step=delta)


if __name__ == "__main__":
    unittest.main()
