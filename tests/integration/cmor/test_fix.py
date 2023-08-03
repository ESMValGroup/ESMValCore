"""Integration tests for fixes."""

from cf_units import Unit
import numpy as np
import pytest
from iris.coords import DimCoord, AuxCoord
from iris.cube import Cube, CubeList
import dask.array as da

from esmvalcore.preprocessor import (
    fix_metadata,
    fix_data,
    cmor_check_data,
    cmor_check_metadata,
)
from esmvalcore.cmor.fix import AutomaticFix


@pytest.fixture
def only_automatic_fixes(monkeypatch):
    """Only use automatic fixes."""

    def return_empty_list(*_, **__):
        """Return empty list."""
        return []

    monkeypatch.setattr(AutomaticFix, 'get_fixes', return_empty_list)


class TestAutomaticFix:
    """Tests for ``AutomaticFix``."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        """Setup tests."""
        self.mock_debug = mocker.patch.object(
            AutomaticFix, '_debug_msg', autospec=True
        )
        self.mock_warning = mocker.patch.object(
            AutomaticFix, '_warning_msg', autospec=True
        )

        # Create sample data with CMOR errors
        self.time_coord = DimCoord(
            [15, 45],
            standard_name='time',
            var_name='time',
            units=Unit('days since 1851-01-01', calendar='noleap'),
            attributes={'test': 1, 'time_origin': 'will_be_removed'},
        )
        self.plev_coord_rev = DimCoord(
            [250, 500, 850],
            standard_name='air_pressure',
            var_name='plev',
            units='hPa',
        )
        self.lat_coord = DimCoord(
            [0, 10],
            standard_name='latitude',
            var_name='lat',
            units='degrees',
        )
        self.lat_coord_rev = DimCoord(
            [10, -10],
            standard_name='latitude',
            var_name='lat',
            units='degrees',
        )
        self.lon_coord = DimCoord(
            [-180, 0],
            standard_name='longitude',
            var_name='lon',
            units='degrees',
        )
        self.lon_coord_unstructured = AuxCoord(
            [-180, 0],
            bounds=[[-200, -180, -160], [-20, 0, 20]],
            standard_name='longitude',
            var_name='lon',
            units='degrees',
        )
        self.height2m_coord = AuxCoord(
            2.0,
            standard_name='height',
            var_name='height',
            units='m',
        )

        coord_spec_3d = [
            (self.time_coord, 0),
            (self.lat_coord, 1),
            (self.lon_coord, 2),
        ]
        self.cube_3d = Cube(
            da.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2),
            standard_name='air_pressure',
            long_name='Air Pressure',
            var_name='tas',
            units='celsius',
            dim_coords_and_dims=coord_spec_3d,
            aux_coords_and_dims=[(self.height2m_coord, ())],
            attributes={},
        )

        coord_spec_4d = [
            (self.time_coord, 0),
            (self.plev_coord_rev, 1),
            (self.lat_coord_rev, 2),
            (self.lon_coord, 3),
        ]
        self.cube_4d = Cube(
            da.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2),
            standard_name='air_pressure',
            long_name='Air Pressure',
            var_name='ta',
            units='celsius',
            dim_coords_and_dims=coord_spec_4d,
            attributes={},
        )
        self.cubes_4d = CubeList([self.cube_4d])

        coord_spec_unstrucutred = [
            (self.height2m_coord, ()),
            (self.lat_coord_rev, 1),
            (self.lon_coord_unstructured, 1),
        ]
        self.cube_unstructured = Cube(
            da.zeros((2, 2)),
            standard_name='air_pressure',
            long_name='Air Pressure',
            var_name='tas',
            units='celsius',
            dim_coords_and_dims=[(self.time_coord, 0)],
            aux_coords_and_dims=coord_spec_unstrucutred,
            attributes={},
        )
        self.cubes_unstructured = CubeList([self.cube_unstructured])

    def assert_ta_metadata(self, cube, time_has_bounds=True):
        """Assert ta metadata is correct."""
        # Variable metadata
        assert cube.standard_name == 'air_temperature'
        assert cube.long_name == 'Air Temperature'
        assert cube.var_name == 'ta'
        assert cube.units == 'K'
        assert cube.attributes == {}

        # Variable data
        assert cube.has_lazy_data()
        np.testing.assert_allclose(
            cube.data,
            [[[[284.15, 283.15],
               [282.15, 281.15]],
              [[280.15, 279.15],
               [278.15, 277.15]],
              [[276.15, 275.15],
               [274.15, 273.15]]],
             [[[296.15, 295.15],
               [294.15, 293.15]],
              [[292.15, 291.15],
               [290.15, 289.15]],
              [[288.15, 287.15],
               [286.15, 285.15]]]],
        )

        # Time
        assert cube.coord('time').standard_name == 'time'
        assert cube.coord('time').var_name == 'time'
        assert cube.coord('time').units == Unit(
            'days since 1850-01-01', calendar='365_day'
        )
        np.testing.assert_allclose(cube.coord('time').points, [380, 410])
        if time_has_bounds:
            np.testing.assert_allclose(
                cube.coord('time').bounds, [[365, 396], [396, 424]],
            )
        else:
            assert cube.coord('time').bounds is None
        assert cube.coord('time').attributes == {'test': 1}

        # Air pressure
        assert cube.coord('air_pressure').standard_name == 'air_pressure'
        assert cube.coord('air_pressure').var_name == 'plev'
        assert cube.coord('air_pressure').units == 'Pa'
        np.testing.assert_allclose(
            cube.coord('air_pressure').points,
            [85000.0, 50000.0, 25000.0],
            atol=1e-8,
        )
        assert cube.coord('air_pressure').bounds is None
        assert cube.coord('air_pressure').attributes == {}

        # Latitude
        assert cube.coord('latitude').standard_name == 'latitude'
        assert cube.coord('latitude').var_name == 'lat'
        assert str(cube.coord('latitude').units) == 'degrees_north'
        np.testing.assert_allclose(
            cube.coord('latitude').points, [-10.0, 10.0]
        )
        np.testing.assert_allclose(
            cube.coord('latitude').bounds, [[-20.0, 0.0], [0.0, 20.0]],
        )
        assert cube.coord('latitude').attributes == {}

        # Longitude
        assert cube.coord('longitude').standard_name == 'longitude'
        assert cube.coord('longitude').var_name == 'lon'
        assert str(cube.coord('longitude').units) == 'degrees_east'
        np.testing.assert_allclose(
            cube.coord('longitude').points, [0.0, 180.0]
        )
        np.testing.assert_allclose(
            cube.coord('longitude').bounds, [[-90.0, 90.0], [90.0, 270.0]],
        )
        assert cube.coord('longitude').attributes == {}

    def test_fix_metadata_amon_ta(self):
        """Test ``fix_metadata``."""
        short_name = 'ta'
        project = 'CMIP6'
        dataset = '__MODEL_WITH_NO_EXPLICIT_FIX__'
        mip = 'Amon'

        fixed_cubes = fix_metadata(
            self.cubes_4d,
            short_name,
            project,
            dataset,
            mip,
        )

        assert len(fixed_cubes) == 1
        fixed_cube = fixed_cubes[0]

        self.assert_ta_metadata(fixed_cube)
        cmor_check_metadata(fixed_cube, project, mip, short_name)

        assert self.mock_debug.call_count == 3
        assert self.mock_warning.call_count == 9

    def test_fix_metadata_cfmon_ta(self):
        """Test ``fix_metadata`` with alternative generic level coordinate."""
        short_name = 'ta'
        project = 'CMIP6'
        dataset = '__MODEL_WITH_NO_EXPLICIT_FIX__'
        mip = 'CFmon'

        fixed_cubes = fix_metadata(
            self.cubes_4d,
            short_name,
            project,
            dataset,
            mip,
        )

        assert len(fixed_cubes) == 1
        fixed_cube = fixed_cubes[0]

        self.assert_ta_metadata(fixed_cube)
        cmor_check_metadata(fixed_cube, project, mip, short_name)

        assert self.mock_debug.call_count == 4
        assert self.mock_warning.call_count == 9

    def test_fix_metadata_e1hr_ta(self):
        """Test ``fix_metadata`` with plev3."""
        short_name = 'ta'
        project = 'CMIP6'
        dataset = '__MODEL_WITH_NO_EXPLICIT_FIX__'
        mip = 'E1hr'

        # Slighly adapt plev to test fixing of requested levels
        self.cubes_4d[0].coord('air_pressure').points = [
            250.0 + 9e-8, 500.0 + 9e-8, 850.0 + 9e-8
        ]

        fixed_cubes = fix_metadata(
            self.cubes_4d,
            short_name,
            project,
            dataset,
            mip,
            frequency='mon',
        )

        assert len(fixed_cubes) == 1
        fixed_cube = fixed_cubes[0]

        self.assert_ta_metadata(fixed_cube, time_has_bounds=False)
        cmor_check_metadata(
            fixed_cube, project, mip, short_name, frequency='mon'
        )

        assert self.mock_debug.call_count == 4
        assert self.mock_warning.call_count == 8

    def test_fix_metadata_amon_tas_unstructured(self):
        """Test ``fix_metadata``."""
        short_name = 'tas'
        project = 'CMIP6'
        dataset = '__MODEL_WITH_NO_EXPLICIT_FIX__'
        mip = 'Amon'

        fixed_cubes = fix_metadata(
            self.cubes_unstructured,
            short_name,
            project,
            dataset,
            mip,
        )

        assert len(fixed_cubes) == 1
        fixed_cube = fixed_cubes[0]

        # Variable metadata
        assert fixed_cube.standard_name == 'air_temperature'
        assert fixed_cube.long_name == 'Near-Surface Air Temperature'
        assert fixed_cube.var_name == 'tas'
        assert fixed_cube.units == 'K'
        assert fixed_cube.attributes == {}

        # Variable data
        assert fixed_cube.has_lazy_data()
        np.testing.assert_allclose(
            fixed_cube.data, [[273.15, 273.15], [273.15, 273.15]],
        )

        # Time
        assert fixed_cube.coord('time').standard_name == 'time'
        assert fixed_cube.coord('time').var_name == 'time'
        assert fixed_cube.coord('time').units == Unit(
            'days since 1850-01-01', calendar='365_day'
        )
        np.testing.assert_allclose(fixed_cube.coord('time').points, [380, 410])
        np.testing.assert_allclose(
            fixed_cube.coord('time').bounds, [[365, 396], [396, 424]],
        )
        assert fixed_cube.coord('time').attributes == {'test': 1}

        # Latitude
        assert fixed_cube.coord('latitude').standard_name == 'latitude'
        assert fixed_cube.coord('latitude').var_name == 'lat'
        assert str(fixed_cube.coord('latitude').units) == 'degrees_north'
        np.testing.assert_allclose(
            fixed_cube.coord('latitude').points, [10.0, -10.0]
        )
        assert fixed_cube.coord('latitude').bounds is None
        assert fixed_cube.coord('latitude').attributes == {}

        # Longitude
        assert fixed_cube.coord('longitude').standard_name == 'longitude'
        assert fixed_cube.coord('longitude').var_name == 'lon'
        assert str(fixed_cube.coord('longitude').units) == 'degrees_east'
        np.testing.assert_allclose(
            fixed_cube.coord('longitude').points, [180.0, 0.0]
        )
        np.testing.assert_allclose(
            fixed_cube.coord('longitude').bounds,
            [[160.0, 180.0, 200.0], [340.0, 0.0, 20.0]],
        )
        assert fixed_cube.coord('longitude').attributes == {}

        cmor_check_metadata(fixed_cube, project, mip, short_name)

        assert self.mock_debug.call_count == 2
        assert self.mock_warning.call_count == 6

    def test_fix_data_amon_tas(self):
        """Test ``fix_data``."""
        short_name = 'tas'
        project = 'CMIP6'
        dataset = '__MODEL_WITH_NO_EXPLICIT_FIX__'
        mip = 'Amon'

        fixed_cube = fix_data(
            self.cube_3d,
            short_name,
            project,
            dataset,
            mip,
        )

        assert fixed_cube.has_lazy_data()

        cmor_check_data(fixed_cube, project, mip, short_name)

        assert self.mock_debug.call_count == 0
        assert self.mock_warning.call_count == 0
