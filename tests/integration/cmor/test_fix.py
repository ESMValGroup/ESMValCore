"""Integration tests for fixes."""

import dask.array as da
import numpy as np
import pytest
from cf_units import Unit
from iris.aux_factory import HybridHeightFactory, HybridPressureFactory
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.cmor.check import CheckLevels, CMORCheckError
from esmvalcore.exceptions import ESMValCoreDeprecationWarning
from esmvalcore.preprocessor import (
    cmor_check_data,
    cmor_check_metadata,
    fix_data,
    fix_metadata,
)


# TODO: remove in v2.12
@pytest.fixture(autouse=True)
def disable_fix_cmor_checker(mocker):
    """Disable the CMOR checker in fixes (will be default in v2.12)."""
    class MockChecker:

        def __init__(self, cube):
            self._cube = cube

        def check_metadata(self):
            return self._cube

        def check_data(self):
            return self._cube

    mock = mocker.patch('esmvalcore.cmor.fix._get_cmor_checker')
    mock.return_value = MockChecker


class TestGenericFix:
    """Tests for ``GenericFix``."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        """Setup tests."""
        self.mock_debug = mocker.patch(
            'esmvalcore.cmor._fixes.fix.GenericFix._debug_msg', autospec=True
        )
        self.mock_warning = mocker.patch(
            'esmvalcore.cmor._fixes.fix.GenericFix._warning_msg',
            autospec=True,
        )

        # Create sample data with CMOR errors
        time_coord = DimCoord(
            [15, 45],
            standard_name='time',
            var_name='time',
            units=Unit('days since 1851-01-01', calendar='noleap'),
            attributes={'test': 1, 'time_origin': 'will_be_removed'},
        )
        plev_coord_rev = DimCoord(
            [250, 500, 850],
            standard_name='air_pressure',
            var_name='plev',
            units='hPa',
        )
        lev_coord_hybrid_height = DimCoord(
            [1.0, 0.5, 0.0],
            standard_name='atmosphere_hybrid_height_coordinate',
            var_name='lev',
            units='m',
        )
        lev_coord_hybrid_pressure = DimCoord(
            [0.0, 0.5, 1.0],
            standard_name='atmosphere_hybrid_sigma_pressure_coordinate',
            var_name='lev',
            units='1',
        )
        ap_coord = AuxCoord(
            [0.0, 0.0, 0.0],
            var_name='ap',
            units='Pa',
        )
        b_coord = AuxCoord(
            [0.0, 0.5, 1.0],
            var_name='b',
            units='1',
        )
        ps_coord = AuxCoord(
            np.full((2, 2, 2), 10),
            var_name='ps',
            units='Pa',
        )
        orog_coord = AuxCoord(
            np.full((2, 2), 10),
            var_name='orog',
            units='m',
        )
        hybrid_height_factory = HybridHeightFactory(
            delta=lev_coord_hybrid_height,
            sigma=b_coord,
            orography=orog_coord,
        )
        hybrid_pressure_factory = HybridPressureFactory(
            delta=ap_coord,
            sigma=lev_coord_hybrid_pressure,
            surface_air_pressure=ps_coord,
        )
        lat_coord = DimCoord(
            [0, 10],
            standard_name='latitude',
            var_name='lat',
            units='degrees',
        )
        lat_coord_rev = DimCoord(
            [10, -10],
            standard_name='latitude',
            var_name='lat',
            units='degrees',
        )
        lat_coord_2d = AuxCoord(
            [[10, -10]],
            standard_name='latitude',
            var_name='wrong_name',
            units='degrees',
        )
        lon_coord = DimCoord(
            [-180, 0],
            standard_name='longitude',
            var_name='lon',
            units='degrees',
        )
        lon_coord_unstructured = AuxCoord(
            [-180, 0],
            bounds=[[-200, -180, -160], [-20, 0, 20]],
            standard_name='longitude',
            var_name='lon',
            units='degrees',
        )
        lon_coord_2d = AuxCoord(
            [[370, 380]],
            standard_name='longitude',
            var_name='wrong_name',
            units='degrees',
        )
        height2m_coord = AuxCoord(
            2.0,
            standard_name='height',
            var_name='height',
            units='m',
        )

        coord_spec_3d = [
            (time_coord, 0),
            (lat_coord, 1),
            (lon_coord, 2),
        ]
        self.cube_3d = Cube(
            da.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2),
            standard_name='air_pressure',
            long_name='Air Pressure',
            var_name='tas',
            units='celsius',
            dim_coords_and_dims=coord_spec_3d,
            aux_coords_and_dims=[(height2m_coord, ())],
            attributes={},
        )

        coord_spec_4d = [
            (time_coord, 0),
            (plev_coord_rev, 1),
            (lat_coord_rev, 2),
            (lon_coord, 3),
        ]
        cube_4d = Cube(
            da.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2),
            standard_name='air_pressure',
            long_name='Air Pressure',
            var_name='ta',
            units='celsius',
            dim_coords_and_dims=coord_spec_4d,
            attributes={},
        )
        self.cubes_4d = CubeList([cube_4d])

        coord_spec_hybrid_height_4d = [
            (time_coord, 0),
            (lev_coord_hybrid_height, 1),
            (lat_coord_rev, 2),
            (lon_coord, 3),
        ]
        aux_coord_spec_hybrid_height_4d = [
            (b_coord, 1),
            (orog_coord, (2, 3)),
        ]
        cube_hybrid_height_4d = Cube(
            da.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2),
            standard_name='air_pressure',
            long_name='Air Pressure',
            var_name='ta',
            units='celsius',
            dim_coords_and_dims=coord_spec_hybrid_height_4d,
            aux_coords_and_dims=aux_coord_spec_hybrid_height_4d,
            aux_factories=[hybrid_height_factory],
            attributes={},
        )
        self.cubes_hybrid_height_4d = CubeList([cube_hybrid_height_4d])

        coord_spec_hybrid_pressure_4d = [
            (time_coord, 0),
            (lev_coord_hybrid_pressure, 1),
            (lat_coord_rev, 2),
            (lon_coord, 3),
        ]
        aux_coord_spec_hybrid_pressure_4d = [
            (ap_coord, 1),
            (ps_coord, (0, 2, 3)),
        ]
        cube_hybrid_pressure_4d = Cube(
            da.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2),
            standard_name='air_pressure',
            long_name='Air Pressure',
            var_name='ta',
            units='celsius',
            dim_coords_and_dims=coord_spec_hybrid_pressure_4d,
            aux_coords_and_dims=aux_coord_spec_hybrid_pressure_4d,
            aux_factories=[hybrid_pressure_factory],
            attributes={},
        )
        self.cubes_hybrid_pressure_4d = CubeList([cube_hybrid_pressure_4d])

        coord_spec_unstrucutred = [
            (height2m_coord, ()),
            (lat_coord_rev, 1),
            (lon_coord_unstructured, 1),
        ]
        cube_unstructured = Cube(
            da.zeros((2, 2)),
            standard_name='air_pressure',
            long_name='Air Pressure',
            var_name='tas',
            units='celsius',
            dim_coords_and_dims=[(time_coord, 0)],
            aux_coords_and_dims=coord_spec_unstrucutred,
            attributes={},
        )
        self.cubes_unstructured = CubeList([cube_unstructured])

        coord_spec_2d = [
            (height2m_coord, ()),
            (lat_coord_2d, (1, 2)),
            (lon_coord_2d, (1, 2)),
        ]
        cube_2d_latlon = Cube(
            da.zeros((2, 1, 2)),
            standard_name='air_pressure',
            long_name='Air Pressure',
            var_name='tas',
            units='celsius',
            dim_coords_and_dims=[(time_coord, 0)],
            aux_coords_and_dims=coord_spec_2d,
            attributes={},
        )
        self.cubes_2d_latlon = CubeList([cube_2d_latlon])

    def assert_time_metadata(self, cube):
        """Assert time metadata is correct."""
        assert cube.coord('time').standard_name == 'time'
        assert cube.coord('time').var_name == 'time'
        assert cube.coord('time').units == Unit(
            'days since 1850-01-01', calendar='365_day'
        )
        assert cube.coord('time').attributes == {'test': 1}

    def assert_time_data(self, cube, time_has_bounds=True):
        """Assert time data is correct."""
        np.testing.assert_allclose(cube.coord('time').points, [380, 410])
        if time_has_bounds:
            np.testing.assert_allclose(
                cube.coord('time').bounds, [[365, 396], [396, 424]],
            )
        else:
            assert cube.coord('time').bounds is None

    def assert_plev_metadata(self, cube):
        """Assert plev metadata is correct."""
        assert cube.coord('air_pressure').standard_name == 'air_pressure'
        assert cube.coord('air_pressure').var_name == 'plev'
        assert cube.coord('air_pressure').units == 'Pa'
        assert cube.coord('air_pressure').attributes == {}

    def assert_lat_metadata(self, cube):
        """Assert lat metadata is correct."""
        assert cube.coord('latitude').standard_name == 'latitude'
        assert cube.coord('latitude').var_name == 'lat'
        assert str(cube.coord('latitude').units) == 'degrees_north'
        assert cube.coord('latitude').attributes == {}

    def assert_lon_metadata(self, cube):
        """Assert lon metadata is correct."""
        assert cube.coord('longitude').standard_name == 'longitude'
        assert cube.coord('longitude').var_name == 'lon'
        assert str(cube.coord('longitude').units) == 'degrees_east'
        assert cube.coord('longitude').attributes == {}

    def assert_ta_metadata(self, cube):
        """Assert ta metadata is correct."""
        # Variable metadata
        assert cube.standard_name == 'air_temperature'
        assert cube.long_name == 'Air Temperature'
        assert cube.var_name == 'ta'
        assert cube.units == 'K'
        assert cube.attributes == {}

    def assert_ta_data(self, cube, time_has_bounds=True):
        """Assert ta data is correct."""
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
        self.assert_time_data(cube, time_has_bounds=time_has_bounds)

        # Air pressure
        np.testing.assert_allclose(
            cube.coord('air_pressure').points,
            [85000.0, 50000.0, 25000.0],
            atol=1e-8,
        )
        assert cube.coord('air_pressure').bounds is None

        # Latitude
        np.testing.assert_allclose(
            cube.coord('latitude').points, [-10.0, 10.0]
        )
        np.testing.assert_allclose(
            cube.coord('latitude').bounds, [[-20.0, 0.0], [0.0, 20.0]]
        )

        # Longitude
        np.testing.assert_allclose(
            cube.coord('longitude').points, [0.0, 180.0]
        )
        np.testing.assert_allclose(
            cube.coord('longitude').bounds, [[-90.0, 90.0], [90.0, 270.0]]
        )

    def assert_tas_metadata(self, cube):
        """Assert tas metadata is correct."""
        assert cube.standard_name == 'air_temperature'
        assert cube.long_name == 'Near-Surface Air Temperature'
        assert cube.var_name == 'tas'
        assert cube.units == 'K'
        assert cube.attributes == {}

        # Height 2m coordinate
        assert cube.coord('height').standard_name == 'height'
        assert cube.coord('height').var_name == 'height'
        assert cube.coord('height').units == 'm'
        assert cube.coord('height').attributes == {}
        np.testing.assert_allclose(cube.coord('height').points, 2.0)
        assert cube.coord('height').bounds is None

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
        self.assert_time_metadata(fixed_cube)
        self.assert_plev_metadata(fixed_cube)
        self.assert_lat_metadata(fixed_cube)
        self.assert_lon_metadata(fixed_cube)
        self.assert_ta_data(fixed_cube)

        cmor_check_metadata(fixed_cube, project, mip, short_name)

        assert self.mock_debug.call_count == 3
        assert self.mock_warning.call_count == 9

    def test_fix_metadata_amon_ta_wrong_lat_units(self):
        """Test ``fix_metadata``."""
        short_name = 'ta'
        project = 'CMIP6'
        dataset = '__MODEL_WITH_NO_EXPLICIT_FIX__'
        mip = 'Amon'

        # Change units of latitude
        self.cubes_4d[0].coord('latitude').units = 'K'

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
        self.assert_time_metadata(fixed_cube)
        self.assert_plev_metadata(fixed_cube)
        self.assert_lon_metadata(fixed_cube)
        self.assert_ta_data(fixed_cube)

        # CMOR check will fail because of wrong latitude units
        assert fixed_cube.coord('latitude').units == 'K'
        with pytest.raises(CMORCheckError):
            cmor_check_metadata(fixed_cube, project, mip, short_name)

        print(self.mock_debug.mock_calls)
        print(self.mock_warning.mock_calls)

        assert self.mock_debug.call_count == 3
        assert self.mock_warning.call_count == 9

    def test_fix_metadata_cfmon_ta_hybrid_height(self):
        """Test ``fix_metadata`` with hybrid height coordinate."""
        short_name = 'ta'
        project = 'CMIP6'
        dataset = '__MODEL_WITH_NO_EXPLICIT_FIX__'
        mip = 'CFmon'

        fixed_cubes = fix_metadata(
            self.cubes_hybrid_height_4d,
            short_name,
            project,
            dataset,
            mip,
        )

        assert len(fixed_cubes) == 1
        fixed_cube = fixed_cubes[0]

        hybrid_coord = fixed_cube.coord('atmosphere_hybrid_height_coordinate')
        assert hybrid_coord.var_name == 'lev'
        assert hybrid_coord.long_name is None
        assert hybrid_coord.units == 'm'
        np.testing.assert_allclose(hybrid_coord.points, [0.0, 0.5, 1.0])
        assert fixed_cube.coords('altitude')
        assert fixed_cube.coord_dims('altitude') == (1, 2, 3)

        self.assert_ta_metadata(fixed_cube)
        self.assert_time_metadata(fixed_cube)
        self.assert_lat_metadata(fixed_cube)
        self.assert_lon_metadata(fixed_cube)

        cmor_check_metadata(fixed_cube, project, mip, short_name)

        assert self.mock_debug.call_count == 4
        assert self.mock_warning.call_count == 9

    def test_fix_metadata_cfmon_ta_hybrid_pressure(self):
        """Test ``fix_metadata`` with hybrid pressure coordinate."""
        short_name = 'ta'
        project = 'CMIP6'
        dataset = '__MODEL_WITH_NO_EXPLICIT_FIX__'
        mip = 'CFmon'

        fixed_cubes = fix_metadata(
            self.cubes_hybrid_pressure_4d,
            short_name,
            project,
            dataset,
            mip,
        )

        assert len(fixed_cubes) == 1
        fixed_cube = fixed_cubes[0]

        hybrid_coord = fixed_cube.coord(
            'atmosphere_hybrid_sigma_pressure_coordinate'
        )
        assert hybrid_coord.var_name == 'lev'
        assert hybrid_coord.long_name is None
        assert hybrid_coord.units == '1'
        np.testing.assert_allclose(hybrid_coord.points, [1.0, 0.5, 0.0])
        assert fixed_cube.coords('air_pressure')
        assert fixed_cube.coord_dims('air_pressure') == (0, 1, 2, 3)

        self.assert_ta_metadata(fixed_cube)
        self.assert_time_metadata(fixed_cube)
        self.assert_lat_metadata(fixed_cube)
        self.assert_lon_metadata(fixed_cube)

        cmor_check_metadata(fixed_cube, project, mip, short_name)

        assert self.mock_debug.call_count == 4
        assert self.mock_warning.call_count == 9

    def test_fix_metadata_cfmon_ta_alternative(self):
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
        self.assert_time_metadata(fixed_cube)
        self.assert_plev_metadata(fixed_cube)
        self.assert_lat_metadata(fixed_cube)
        self.assert_lon_metadata(fixed_cube)
        self.assert_ta_data(fixed_cube)

        cmor_check_metadata(fixed_cube, project, mip, short_name)

        assert self.mock_debug.call_count == 4
        assert self.mock_warning.call_count == 9

    def test_fix_metadata_cfmon_ta_no_alternative(self, mocker):
        """Test ``fix_metadata`` with  no alternative coordinate."""
        short_name = 'ta'
        project = 'CMIP6'
        dataset = '__MODEL_WITH_NO_EXPLICIT_FIX__'
        mip = 'CFmon'

        # Remove alternative coordinate
        self.cubes_4d[0].remove_coord('air_pressure')

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
        self.assert_time_metadata(fixed_cube)
        self.assert_time_data(fixed_cube)
        self.assert_lat_metadata(fixed_cube)
        self.assert_lon_metadata(fixed_cube)

        # CMOR check will fail because of missing alevel coordinate
        assert not fixed_cube.coords('air_pressure')
        with pytest.raises(CMORCheckError):
            cmor_check_metadata(fixed_cube, project, mip, short_name)

        assert self.mock_debug.call_count == 2
        assert self.mock_warning.call_count == 8

    def test_fix_metadata_e1hr_ta(self):
        """Test ``fix_metadata`` with plev3."""
        short_name = 'ta'
        project = 'CMIP6'
        dataset = '__MODEL_WITH_NO_EXPLICIT_FIX__'
        mip = 'E1hr'

        # Slightly adapt plev to test fixing of requested levels
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

        self.assert_ta_metadata(fixed_cube)
        self.assert_time_metadata(fixed_cube)
        self.assert_plev_metadata(fixed_cube)
        self.assert_lat_metadata(fixed_cube)
        self.assert_lon_metadata(fixed_cube)
        self.assert_ta_data(fixed_cube, time_has_bounds=False)

        cmor_check_metadata(
            fixed_cube, project, mip, short_name, frequency='mon'
        )

        assert self.mock_debug.call_count == 4
        assert self.mock_warning.call_count == 8

    def test_fix_metadata_amon_tas_unstructured(self):
        """Test ``fix_metadata`` with unstructured grid."""
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

        self.assert_tas_metadata(fixed_cube)
        self.assert_time_metadata(fixed_cube)
        self.assert_lat_metadata(fixed_cube)
        self.assert_lon_metadata(fixed_cube)

        # Latitude
        np.testing.assert_allclose(
            fixed_cube.coord('latitude').points, [10.0, -10.0]
        )
        assert fixed_cube.coord('latitude').bounds is None

        # Longitude
        np.testing.assert_allclose(
            fixed_cube.coord('longitude').points, [180.0, 0.0]
        )
        np.testing.assert_allclose(
            fixed_cube.coord('longitude').bounds,
            [[160.0, 180.0, 200.0], [340.0, 0.0, 20.0]],
        )

        # Variable data
        assert fixed_cube.has_lazy_data()
        np.testing.assert_allclose(
            fixed_cube.data, [[273.15, 273.15], [273.15, 273.15]]
        )

        cmor_check_metadata(fixed_cube, project, mip, short_name)

        assert self.mock_debug.call_count == 2
        assert self.mock_warning.call_count == 6

    def test_fix_metadata_amon_tas_2d_latlon(self):
        """Test ``fix_metadata`` with 2D latitude/longitude."""
        short_name = 'tas'
        project = 'CMIP6'
        dataset = '__MODEL_WITH_NO_EXPLICIT_FIX__'
        mip = 'Amon'

        fixed_cubes = fix_metadata(
            self.cubes_2d_latlon,
            short_name,
            project,
            dataset,
            mip,
        )

        assert len(fixed_cubes) == 1
        fixed_cube = fixed_cubes[0]

        self.assert_tas_metadata(fixed_cube)
        self.assert_time_metadata(fixed_cube)
        self.assert_lat_metadata(fixed_cube)
        self.assert_lon_metadata(fixed_cube)

        # Latitude
        np.testing.assert_allclose(
            fixed_cube.coord('latitude').points, [[10.0, -10.0]]
        )
        assert fixed_cube.coord('latitude').bounds is None

        # Longitude
        np.testing.assert_allclose(
            fixed_cube.coord('longitude').points, [[10.0, 20.0]]
        )
        assert fixed_cube.coord('longitude').bounds is None

        # Variable data
        assert fixed_cube.has_lazy_data()
        np.testing.assert_allclose(
            fixed_cube.data, [[[273.15, 273.15]], [[273.15, 273.15]]]
        )

        cmor_check_metadata(fixed_cube, project, mip, short_name)

        assert self.mock_debug.call_count == 3
        assert self.mock_warning.call_count == 8

    def test_fix_metadata_amon_tas_invalid_time_units(self):
        """Test ``fix_metadata`` with invalid time units."""
        short_name = 'tas'
        project = 'CMIP6'
        dataset = '__MODEL_WITH_NO_EXPLICIT_FIX__'
        mip = 'Amon'

        self.cubes_2d_latlon[0].remove_coord('time')
        aux_time_coord = AuxCoord(
            [1, 2],
            standard_name='time',
            var_name='time',
            units='kg',
        )
        self.cubes_2d_latlon[0].add_aux_coord(aux_time_coord, 0)

        fixed_cubes = fix_metadata(
            self.cubes_2d_latlon,
            short_name,
            project,
            dataset,
            mip,
        )

        assert len(fixed_cubes) == 1
        fixed_cube = fixed_cubes[0]

        self.assert_lat_metadata(fixed_cube)
        self.assert_lon_metadata(fixed_cube)

        assert fixed_cube.coord('time').units == 'kg'

        # CMOR checks fail because calendar is not defined
        with pytest.raises(ValueError):
            cmor_check_metadata(fixed_cube, project, mip, short_name)

        assert self.mock_debug.call_count == 3
        assert self.mock_warning.call_count == 7

    def test_fix_metadata_amon_tas_invalid_time_attrs(self):
        """Test ``fix_metadata`` with invalid time attributes."""
        short_name = 'tas'
        project = 'CMIP6'
        dataset = '__MODEL_WITH_NO_EXPLICIT_FIX__'
        mip = 'Amon'

        self.cubes_2d_latlon[0].attributes = {
            'parent_time_units': 'this is certainly not a unit',
            'branch_time_in_parent': 'BRANCH TIME IN PARENT',
            'branch_time_in_child': 'BRANCH TIME IN CHILD',
        }

        fixed_cubes = fix_metadata(
            self.cubes_2d_latlon,
            short_name,
            project,
            dataset,
            mip,
        )

        assert len(fixed_cubes) == 1
        fixed_cube = fixed_cubes[0]

        self.assert_time_metadata(fixed_cube)
        self.assert_lat_metadata(fixed_cube)
        self.assert_lon_metadata(fixed_cube)

        assert fixed_cube.attributes == {
            'parent_time_units': 'this is certainly not a unit',
            'branch_time_in_parent': 'BRANCH TIME IN PARENT',
            'branch_time_in_child': 'BRANCH TIME IN CHILD',
        }

        cmor_check_metadata(fixed_cube, project, mip, short_name)

        assert self.mock_debug.call_count == 3
        assert self.mock_warning.call_count == 8

    def test_fix_metadata_oimon_ssi(self):
        """Test ``fix_metadata`` with psu units."""
        short_name = 'ssi'
        project = 'CMIP5'
        dataset = '__MODEL_WITH_NO_EXPLICIT_FIX__'
        mip = 'OImon'

        self.cubes_2d_latlon[0].var_name = 'ssi'
        self.cubes_2d_latlon[0].attributes = {
            'invalid_units': 'psu',
            'parent_time_units': 'no parent',
        }

        # Also test 2D longitude that already has bounds
        self.cubes_2d_latlon[0].coord('latitude').var_name = 'lat'
        self.cubes_2d_latlon[0].coord('longitude').var_name = 'lon'
        self.cubes_2d_latlon[0].coord('longitude').bounds = [
            [[365.0, 375.0], [375.0, 400.0]]
        ]

        fixed_cubes = fix_metadata(
            self.cubes_2d_latlon,
            short_name,
            project,
            dataset,
            mip,
        )

        assert len(fixed_cubes) == 1
        fixed_cube = fixed_cubes[0]

        # Variable metadata
        assert fixed_cube.standard_name == 'sea_ice_salinity'
        assert fixed_cube.long_name == 'Sea Ice Salinity'
        assert fixed_cube.var_name == 'ssi'
        assert fixed_cube.units == '1'
        assert fixed_cube.attributes == {'parent_time_units': 'no parent'}

        # Coordinates
        self.assert_time_metadata(fixed_cube)
        self.assert_lat_metadata(fixed_cube)
        self.assert_lon_metadata(fixed_cube)

        # Latitude
        np.testing.assert_allclose(
            fixed_cube.coord('latitude').points, [[10.0, -10.0]]
        )
        assert fixed_cube.coord('latitude').bounds is None

        # Longitude
        np.testing.assert_allclose(
            fixed_cube.coord('longitude').points, [[10.0, 20.0]]
        )
        np.testing.assert_allclose(
            fixed_cube.coord('longitude').bounds,
            [[[5.0, 15.0], [15.0, 40.0]]],
        )

        # Variable data
        assert fixed_cube.has_lazy_data()
        np.testing.assert_allclose(
            fixed_cube.data, [[[0.0, 0.0]], [[0.0, 0.0]]],
        )

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

    def test_deprecate_check_level_fix_metadata(self):
        """Test deprecation of check level in ``fix_metadata``."""
        with pytest.warns(ESMValCoreDeprecationWarning):
            fix_metadata(
                self.cubes_4d,
                'ta',
                'CMIP6',
                'MODEL',
                'Amon',
                check_level=CheckLevels.RELAXED,
            )

    def test_deprecate_check_level_fix_data(self):
        """Test deprecation of check level in ``fix_data``."""
        with pytest.warns(ESMValCoreDeprecationWarning):
            fix_metadata(
                self.cubes_4d,
                'ta',
                'CMIP6',
                'MODEL',
                'Amon',
                check_level=CheckLevels.RELAXED,
            )
