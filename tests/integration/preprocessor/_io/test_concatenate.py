"""Integration tests for :func:`esmvalcore.preprocessor._io.concatenate`."""

import unittest
import warnings
from unittest.mock import call

import numpy as np
import pytest
from cf_units import Unit
from iris.aux_factory import (
    AtmosphereSigmaFactory,
    HybridHeightFactory,
    HybridPressureFactory,
)
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.preprocessor import _io


def get_hybrid_pressure_cube():
    """Return cube with hybrid pressure coordinate."""
    ap_coord = AuxCoord([1.0], bounds=[[0.0, 2.0]], var_name='ap', units='Pa')
    b_coord = AuxCoord([0.0], bounds=[[-0.5, 1.5]],
                       var_name='b', units=Unit('1'))
    ps_coord = AuxCoord([[[100000]]], var_name='ps', units='Pa')
    x_coord = AuxCoord(
        0.0,
        var_name='x',
        standard_name='atmosphere_hybrid_sigma_pressure_coordinate',
    )
    cube = Cube([[[[0.0]]]], var_name='x',
                aux_coords_and_dims=[(ap_coord, 1), (b_coord, 1),
                                     (ps_coord, (0, 2, 3)), (x_coord, ())])
    return cube


def get_hybrid_pressure_cube_list():
    """Return list of cubes including hybrid pressure coordinate."""
    cube_0 = get_hybrid_pressure_cube()
    cube_1 = get_hybrid_pressure_cube()
    cube_0.add_dim_coord(get_time_coord(0), 0)
    cube_1.add_dim_coord(get_time_coord(1), 0)
    cubes = CubeList([cube_0, cube_1])
    for cube in cubes:
        aux_factory = HybridPressureFactory(
            delta=cube.coord(var_name='ap'),
            sigma=cube.coord(var_name='b'),
            surface_air_pressure=cube.coord(var_name='ps'),
        )
        cube.add_aux_factory(aux_factory)
    return cubes


def get_time_coord(time_point):
    """Time coordinate."""
    return DimCoord([time_point], var_name='time', standard_name='time',
                    units='days since 6453-2-1')


@pytest.fixture
def mock_empty_cube():
    """Return mocked cube with irrelevant coordinates."""
    cube = unittest.mock.create_autospec(Cube, spec_set=True, instance=True)
    a_coord = AuxCoord(0.0, var_name='a')
    b_coord = AuxCoord(0.0, var_name='b')
    cube.coords.return_value = [a_coord, b_coord]
    return cube


@pytest.fixture
def mock_atmosphere_sigma_cube():
    """Return mocked cube with atmosphere sigma coordinate."""
    cube = unittest.mock.create_autospec(Cube, spec_set=True, instance=True)
    ptop_coord = AuxCoord([1.0], var_name='ptop', units='Pa')
    lev_coord = AuxCoord([0.0], bounds=[[-0.5, 1.5]], var_name='lev',
                         units='1')
    ps_coord = AuxCoord([[[100000]]], var_name='ps', units='Pa')
    cube.coord.side_effect = [ptop_coord, lev_coord, ps_coord,
                              ptop_coord, lev_coord, ps_coord]
    cube.coords.return_value = [
        ptop_coord,
        lev_coord,
        ps_coord,
        AuxCoord(0.0, standard_name='atmosphere_sigma_coordinate'),
    ]
    aux_factory = AtmosphereSigmaFactory(
        pressure_at_top=ptop_coord,
        sigma=lev_coord,
        surface_air_pressure=ps_coord,
    )
    cube.aux_factories = ['dummy', aux_factory]
    return cube


@pytest.fixture
def mock_hybrid_height_cube():
    """Return mocked cube with hybrid height coordinate."""
    cube = unittest.mock.create_autospec(Cube, spec_set=True, instance=True)
    lev_coord = AuxCoord([1.0], bounds=[[0.0, 2.0]], var_name='lev', units='m')
    b_coord = AuxCoord([0.0], bounds=[[-0.5, 1.5]], var_name='b')
    orog_coord = AuxCoord([[[100000]]], var_name='orog', units='m')
    cube.coord.side_effect = [lev_coord, b_coord, orog_coord,
                              lev_coord, b_coord, orog_coord]
    cube.coords.return_value = [
        lev_coord,
        b_coord,
        orog_coord,
        AuxCoord(0.0, standard_name='atmosphere_hybrid_height_coordinate'),
    ]
    aux_factory = HybridHeightFactory(
        delta=lev_coord,
        sigma=b_coord,
        orography=orog_coord,
    )
    cube.aux_factories = ['dummy', aux_factory]
    return cube


@pytest.fixture
def mock_hybrid_pressure_cube():
    """Return mocked cube with hybrid pressure coordinate."""
    cube = unittest.mock.create_autospec(Cube, spec_set=True, instance=True)
    ap_coord = AuxCoord([1.0], bounds=[[0.0, 2.0]], var_name='ap', units='Pa')
    b_coord = AuxCoord([0.0], bounds=[[-0.5, 1.5]],
                       var_name='b', units=Unit('1'))
    ps_coord = AuxCoord([[[100000]]], var_name='ps', units='Pa')
    cube.coord.side_effect = [ap_coord, b_coord, ps_coord,
                              ap_coord, b_coord, ps_coord]
    cube.coords.return_value = [
        ap_coord,
        b_coord,
        ps_coord,
        AuxCoord(0.0,
                 standard_name='atmosphere_hybrid_sigma_pressure_coordinate'),
    ]
    aux_factory = HybridPressureFactory(
        delta=ap_coord,
        sigma=b_coord,
        surface_air_pressure=ps_coord,
    )
    cube.aux_factories = ['dummy', aux_factory]
    return cube


@pytest.fixture
def real_hybrid_pressure_cube():
    """Return real cube with hybrid pressure coordinate."""
    return get_hybrid_pressure_cube()


@pytest.fixture
def real_hybrid_pressure_cube_list():
    """Return real list of cubes with hybrid pressure coordinate."""
    return get_hybrid_pressure_cube_list()


def check_if_fix_aux_factories_is_necessary():
    """Check if _fix_aux_factories() is necessary (i.e. iris bug is fixed)."""
    cubes = get_hybrid_pressure_cube_list()
    cube = cubes.concatenate_cube()
    coords = [coord.name() for coord in cube.coords()]
    msg = ("Apparently concatenation of cubes that have a derived variable "
           "is now possible in iris (i.e. issue #2478 has been fixed). Thus, "
           "this test and ALL appearances of the function "
           "'_fix_aux_factories' can safely be removed!")
    if 'air_pressure' in coords:
        warnings.warn(msg)


def test_fix_aux_factories_empty_cube(mock_empty_cube):
    """Test fixing with empty cube."""
    check_if_fix_aux_factories_is_necessary()
    _io._fix_aux_factories(mock_empty_cube)
    assert mock_empty_cube.mock_calls == [call.coords()]


def test_fix_aux_factories_atmosphere_sigma(mock_atmosphere_sigma_cube):
    """Test fixing of atmosphere sigma coordinate."""
    check_if_fix_aux_factories_is_necessary()

    # Test with aux_factory object
    _io._fix_aux_factories(mock_atmosphere_sigma_cube)
    mock_atmosphere_sigma_cube.coords.assert_called_once_with()
    mock_atmosphere_sigma_cube.coord.assert_has_calls([call(var_name='ptop'),
                                                       call(var_name='lev'),
                                                       call(var_name='ps')])
    mock_atmosphere_sigma_cube.add_aux_factory.assert_not_called()

    # Test without aux_factory object
    mock_atmosphere_sigma_cube.reset_mock()
    mock_atmosphere_sigma_cube.aux_factories = ['dummy']
    _io._fix_aux_factories(mock_atmosphere_sigma_cube)
    mock_atmosphere_sigma_cube.coords.assert_called_once_with()
    mock_atmosphere_sigma_cube.coord.assert_has_calls([call(var_name='ptop'),
                                                       call(var_name='lev'),
                                                       call(var_name='ps')])
    mock_atmosphere_sigma_cube.add_aux_factory.assert_called_once()


def test_fix_aux_factories_hybrid_height(mock_hybrid_height_cube):
    """Test fixing of hybrid height coordinate."""
    check_if_fix_aux_factories_is_necessary()

    # Test with aux_factory object
    _io._fix_aux_factories(mock_hybrid_height_cube)
    mock_hybrid_height_cube.coords.assert_called_once_with()
    mock_hybrid_height_cube.coord.assert_has_calls([call(var_name='lev'),
                                                    call(var_name='b'),
                                                    call(var_name='orog')])
    mock_hybrid_height_cube.add_aux_factory.assert_not_called()

    # Test without aux_factory object
    mock_hybrid_height_cube.reset_mock()
    mock_hybrid_height_cube.aux_factories = ['dummy']
    _io._fix_aux_factories(mock_hybrid_height_cube)
    mock_hybrid_height_cube.coords.assert_called_once_with()
    mock_hybrid_height_cube.coord.assert_has_calls([call(var_name='lev'),
                                                    call(var_name='b'),
                                                    call(var_name='orog')])
    mock_hybrid_height_cube.add_aux_factory.assert_called_once()


def test_fix_aux_factories_hybrid_pressure(mock_hybrid_pressure_cube):
    """Test fixing of hybrid pressure coordinate."""
    check_if_fix_aux_factories_is_necessary()

    # Test with aux_factory object
    _io._fix_aux_factories(mock_hybrid_pressure_cube)
    mock_hybrid_pressure_cube.coords.assert_called_once_with()
    mock_hybrid_pressure_cube.coord.assert_has_calls([call(var_name='ap'),
                                                      call(var_name='b'),
                                                      call(var_name='ps')])
    mock_hybrid_pressure_cube.add_aux_factory.assert_not_called()

    # Test without aux_factory object
    mock_hybrid_pressure_cube.reset_mock()
    mock_hybrid_pressure_cube.aux_factories = ['dummy']
    _io._fix_aux_factories(mock_hybrid_pressure_cube)
    mock_hybrid_pressure_cube.coords.assert_called_once_with()
    mock_hybrid_pressure_cube.coord.assert_has_calls([call(var_name='ap'),
                                                      call(var_name='b'),
                                                      call(var_name='ps')])
    mock_hybrid_pressure_cube.add_aux_factory.assert_called_once()


def test_fix_aux_factories_real_cube(real_hybrid_pressure_cube):
    """Test fixing of hybrid pressure coordinate on real cube."""
    check_if_fix_aux_factories_is_necessary()
    assert not real_hybrid_pressure_cube.coords('air_pressure')
    _io._fix_aux_factories(real_hybrid_pressure_cube)
    air_pressure_coord = real_hybrid_pressure_cube.coord('air_pressure')
    expected_coord = AuxCoord([[[[1.0]]]], bounds=[[[[[-50000., 150002.]]]]],
                              standard_name='air_pressure', units='Pa')
    assert air_pressure_coord == expected_coord


def test_concatenation_with_aux_factory(real_hybrid_pressure_cube_list):
    """Test actual concatenation of a cube with a derived coordinate."""
    concatenated = _io.concatenate(real_hybrid_pressure_cube_list)
    air_pressure_coord = concatenated.coord('air_pressure')
    expected_coord = AuxCoord(
        [[[[1.0]]], [[[1.0]]]],
        bounds=[[[[[-50000.0, 150002.0]]]], [[[[-50000.0, 150002.0]]]]],
        standard_name='air_pressure',
        units='Pa',
    )
    assert air_pressure_coord == expected_coord


class TestConcatenate(unittest.TestCase):
    """Tests for :func:`esmvalcore.preprocessor._io.concatenate`."""

    def setUp(self):
        """Start tests."""
        self._model_coord = DimCoord([1., 2.],
                                     var_name='time',
                                     standard_name='time',
                                     units='days since 1950-01-01')
        self.raw_cubes = []
        self._add_cube([1., 2.], [1., 2.])
        self._add_cube([3., 4.], [3., 4.])
        self._add_cube([5., 6.], [5., 6.])

    def _add_cube(self, data, coord):
        self.raw_cubes.append(
            Cube(data,
                 var_name='sample',
                 dim_coords_and_dims=((self._model_coord.copy(coord), 0), )))

    def test_concatenate(self):
        """Test concatenation of two cubes."""
        concatenated = _io.concatenate(self.raw_cubes)
        np.testing.assert_array_equal(
            concatenated.coord('time').points, np.array([1, 2, 3, 4, 5, 6]))

    def test_concatenate_empty_cubes(self):
        """Test concatenation with empty :class:`iris.cube.CubeList`."""
        empty_cubes = CubeList([])
        result = _io.concatenate(empty_cubes)
        assert result is empty_cubes

    def test_concatenate_noop(self):
        """Test concatenation of a single cube."""
        concatenated = _io.concatenate([self.raw_cubes[0]])
        np.testing.assert_array_equal(
            concatenated.coord('time').points, np.array([1, 2]))

    def test_concatenate_with_overlap(self):
        """Test concatenation of time overalapping cubes."""
        self._add_cube([6.5, 7.5], [6., 7.])
        concatenated = _io.concatenate(self.raw_cubes)
        np.testing.assert_array_equal(
            concatenated.coord('time').points,
            np.array([1., 2., 3., 4., 5., 6., 7.]))
        np.testing.assert_array_equal(concatenated.data,
                                      np.array([1., 2., 3., 4., 5., 6.5, 7.5]))

    def test_concatenate_with_overlap_2(self):
        """Test a more generic case."""
        self._add_cube([65., 75., 100.], [9., 10., 11.])
        self._add_cube([65., 75., 100.], [7., 8., 9.])
        concatenated = _io.concatenate(self.raw_cubes)
        np.testing.assert_array_equal(
            concatenated.coord('time').points,
            np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]))

    def test_concatenate_with_overlap_3(self):
        """Test a more generic case."""
        self._add_cube([65., 75., 100.], [9., 10., 11.])
        self._add_cube([65., 75., 100., 100., 100., 112.],
                       [7., 8., 9., 10., 11., 12.])
        concatenated = _io.concatenate(self.raw_cubes)
        np.testing.assert_array_equal(
            concatenated.coord('time').points,
            np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]))

    def test_concatenate_with_overlap_same_start(self):
        """Test a more generic case."""
        cube1 = self.raw_cubes[0]
        raw_cubes = [
            cube1,
        ]
        time_coord = DimCoord([1., 7.],
                              var_name='time',
                              standard_name='time',
                              units='days since 1950-01-01')
        raw_cubes.append(
            Cube([33., 55.],
                 var_name='sample',
                 dim_coords_and_dims=((time_coord, 0), )))
        concatenated = _io.concatenate(raw_cubes)
        np.testing.assert_array_equal(
            concatenated.coord('time').points, np.array([1., 7.]))
        raw_cubes.reverse()
        concatenated = _io.concatenate(raw_cubes)
        np.testing.assert_array_equal(
            concatenated.coord('time').points, np.array([1., 7.]))

    def test_concatenate_with_iris_exception(self):
        """Test a more generic case."""
        time_coord_1 = DimCoord([1.5, 5., 7.],
                                var_name='time',
                                standard_name='time',
                                units='days since 1950-01-01')
        cube1 = Cube([33., 55., 77.],
                     var_name='sample',
                     dim_coords_and_dims=((time_coord_1, 0), ))
        time_coord_2 = DimCoord([1., 5., 7.],
                                var_name='time',
                                standard_name='time',
                                units='days since 1950-01-01')
        cube2 = Cube([33., 55., 77.],
                     var_name='sample',
                     dim_coords_and_dims=((time_coord_2, 0), ))
        cubes_single_ovlp = [cube2, cube1]
        cubess = _io.concatenate(cubes_single_ovlp)
        # this tests the scalar to vector cube conversion too
        time_points = cubess.coord("time").core_points()
        np.testing.assert_array_equal(time_points, [1., 1.5, 5., 7.])

    def test_concatenate_no_time_coords(self):
        """Test a more generic case."""
        time_coord_1 = DimCoord([1.5, 5., 7.],
                                var_name='time',
                                standard_name='time',
                                units='days since 1950-01-01')
        cube1 = Cube([33., 55., 77.],
                     var_name='sample',
                     dim_coords_and_dims=((time_coord_1, 0), ))
        ap_coord_2 = DimCoord([1., 5., 7.],
                              var_name='air_pressure',
                              standard_name='air_pressure',
                              units='m',
                              attributes={'positive': 'down'})
        cube2 = Cube([33., 55., 77.],
                     var_name='sample',
                     dim_coords_and_dims=((ap_coord_2, 0), ))
        with self.assertRaises(ValueError):
            _io.concatenate([cube1, cube2])

    def test_concatenate_with_order(self):
        """Test a more generic case."""
        time_coord_1 = DimCoord([1.5, 2., 5., 7.],
                                var_name='time',
                                standard_name='time',
                                units='days since 1950-01-01')
        cube1 = Cube([33., 44., 55., 77.],
                     var_name='sample',
                     dim_coords_and_dims=((time_coord_1, 0), ))
        time_coord_2 = DimCoord([1., 2., 5., 7., 100.],
                                var_name='time',
                                standard_name='time',
                                units='days since 1950-01-01')
        cube2 = Cube([33., 44., 55., 77., 1000.],
                     var_name='sample',
                     dim_coords_and_dims=((time_coord_2, 0), ))
        cubes_ordered = [cube2, cube1]
        concatenated = _io.concatenate(cubes_ordered)
        np.testing.assert_array_equal(
            concatenated.coord('time').points, np.array([1., 2., 5., 7.,
                                                         100.]))
        cubes_reverse = [cube1, cube2]
        concatenated = _io.concatenate(cubes_reverse)
        np.testing.assert_array_equal(
            concatenated.coord('time').points, np.array([1., 2., 5., 7.,
                                                         100.]))

    def test_fail_on_calendar_concatenate_with_overlap(self):
        """Test fail of concatenation with overlap."""
        time_coord = DimCoord([3., 7000.],
                              var_name='time',
                              standard_name='time',
                              units=Unit('days since 1950-01-01',
                                         calendar='360_day'))
        self.raw_cubes.append(
            Cube([33., 55.],
                 var_name='sample',
                 dim_coords_and_dims=((time_coord, 0), )))
        with self.assertRaises((TypeError, ValueError)):
            _io.concatenate(self.raw_cubes)

    def test_fail_on_units_concatenate_with_overlap(self):
        """Test fail of concatenation with overlap."""
        time_coord_1 = DimCoord([3., 7000.],
                                var_name='time',
                                standard_name='time',
                                units=Unit('days since 1950-01-01',
                                           calendar='360_day'))
        time_coord_2 = DimCoord([3., 9000.],
                                var_name='time',
                                standard_name='time',
                                units=Unit('days since 1950-01-01',
                                           calendar='360_day'))
        time_coord_3 = DimCoord([3., 9000.],
                                var_name='time',
                                standard_name='time',
                                units=Unit('days since 1850-01-01',
                                           calendar='360_day'))
        raw_cubes = []
        raw_cubes.append(
            Cube([33., 55.],
                 var_name='sample',
                 dim_coords_and_dims=((time_coord_1, 0), )))
        raw_cubes.append(
            Cube([33., 55.],
                 var_name='sample',
                 dim_coords_and_dims=((time_coord_2, 0), )))
        raw_cubes.append(
            Cube([33., 55.],
                 var_name='sample',
                 dim_coords_and_dims=((time_coord_3, 0), )))
        with self.assertRaises(ValueError):
            _io.concatenate(raw_cubes)

    def test_fail_metadata_differs(self):
        """Test exception raised if two cubes have different metadata."""
        self.raw_cubes[0].units = 'm'
        self.raw_cubes[1].units = 'K'
        with self.assertRaises(ValueError):
            _io.concatenate(self.raw_cubes)

    def test_fix_attributes(self):
        """Test fixing attributes for concatenation."""
        identical_attrs = {
            'int': 42,
            'float': 3.1415,
            'bool': True,
            'str': 'Hello, world',
            'list': [1, 1, 2, 3, 5, 8, 13],
            'tuple': (1, 2, 3, 4, 5),
            'dict': {
                1: 'one',
                2: 'two',
                3: 'three'
            },
            'nparray': np.arange(42),
        }
        differing_attrs = [
            {
                'new_int': 0,
                'new_str': 'hello',
                'new_nparray': np.arange(3),
                'mix': np.arange(2),
            },
            {
                'new_int': 1,
                'new_str': 'world',
                'new_list': [1, 1, 2],
                'new_tuple': (0, 1),
                'new_dict': {
                    0: 'zero',
                },
                'mix': {
                    1: 'one',
                },
            },
            {
                'new_str': '!',
                'new_list': [1, 1, 2, 3],
                'new_tuple': (1, 2, 3),
                'new_dict': {
                    0: 'zeroo',
                    1: 'one',
                },
                'new_nparray': np.arange(2),
                'mix': False,
            },
        ]
        resulting_attrs = {
            'new_int': '0;1',
            'new_str': 'hello;world;!',
            'new_nparray': '[0 1 2];[0 1]',
            'new_list': '[1, 1, 2];[1, 1, 2, 3]',
            'new_tuple': '(0, 1);(1, 2, 3)',
            'new_dict': "{0: 'zero'};{0: 'zeroo', 1: 'one'}",
            'mix': "[0 1];{1: 'one'};False",
        }
        resulting_attrs.update(identical_attrs)

        for idx in range(3):
            self.raw_cubes[idx].attributes = identical_attrs
            self.raw_cubes[idx].attributes.update(differing_attrs[idx])
        _io._fix_cube_attributes(self.raw_cubes)  # noqa
        for cube in self.raw_cubes:
            self.assertEqual(cube.attributes, resulting_attrs)
