"""Integration tests for :func:`esmvalcore.preprocessor.concatenate`."""

import unittest

import numpy as np
import pytest
from cf_units import Unit
from iris.aux_factory import HybridPressureFactory
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.cmor.check import CheckLevels
from esmvalcore.preprocessor import concatenate
from esmvalcore.preprocessor._concatenate import _remove_time_overlaps
from tests import assert_array_equal


def get_hybrid_pressure_cube():
    """Return cube with hybrid pressure coordinate."""
    ap_coord = AuxCoord([1.0], bounds=[[0.0, 2.0]], var_name="ap", units="Pa")
    b_coord = AuxCoord(
        [0.0],
        bounds=[[-0.5, 1.5]],
        var_name="b",
        units=Unit("1"),
    )
    ps_coord = AuxCoord([[[100000]]], var_name="ps", units="Pa")
    x_coord = AuxCoord(
        0.0,
        var_name="x",
        standard_name="atmosphere_hybrid_sigma_pressure_coordinate",
    )
    return Cube(
        [[[[0.0]]]],
        var_name="x",
        aux_coords_and_dims=[
            (ap_coord, 1),
            (b_coord, 1),
            (ps_coord, (0, 2, 3)),
            (x_coord, ()),
        ],
    )


def get_hybrid_pressure_cube_list():
    """Return list of cubes including hybrid pressure coordinate."""
    cube_0 = get_hybrid_pressure_cube()
    cube_1 = get_hybrid_pressure_cube()
    cube_0.add_dim_coord(get_time_coord(0), 0)
    cube_1.add_dim_coord(get_time_coord(1), 0)
    cubes = CubeList([cube_0, cube_1])
    for cube in cubes:
        aux_factory = HybridPressureFactory(
            delta=cube.coord(var_name="ap"),
            sigma=cube.coord(var_name="b"),
            surface_air_pressure=cube.coord(var_name="ps"),
        )
        cube.add_aux_factory(aux_factory)
    return cubes


def get_time_coord(time_point):
    """Time coordinate."""
    return DimCoord(
        [time_point],
        var_name="time",
        standard_name="time",
        units="days since 6453-2-1",
    )


@pytest.fixture
def real_hybrid_pressure_cube():
    """Return real cube with hybrid pressure coordinate."""
    return get_hybrid_pressure_cube()


@pytest.fixture
def real_hybrid_pressure_cube_list():
    """Return real list of cubes with hybrid pressure coordinate."""
    return get_hybrid_pressure_cube_list()


def test_concatenation_with_aux_factory(real_hybrid_pressure_cube_list):
    """Test actual concatenation of a cube with a derived coordinate."""
    concatenated = concatenate(real_hybrid_pressure_cube_list)
    air_pressure_coord = concatenated.coord("air_pressure")
    expected_coord = AuxCoord(
        [[[[1.0]]], [[[1.0]]]],
        bounds=[[[[[-50000.0, 150002.0]]]], [[[[-50000.0, 150002.0]]]]],
        standard_name="air_pressure",
        units="Pa",
    )
    assert air_pressure_coord == expected_coord


@pytest.mark.parametrize(
    "check_level",
    [CheckLevels.RELAXED, CheckLevels.IGNORE],
)
def test_relax_concatenation(check_level, caplog):
    caplog.set_level("DEBUG")
    cubes = get_hybrid_pressure_cube_list()
    concatenate(cubes, check_level)
    msg = (
        "Concatenation will be performed without checking "
        "auxiliary coordinates, cell measures, ancillaries "
        "and derived coordinates present in the cubes."
    )
    assert msg in caplog.text


class TestConcatenate(unittest.TestCase):
    """Tests for :func:`esmvalcore.preprocessor.concatenate`."""

    def setUp(self):
        """Start tests."""
        self._model_coord = DimCoord(
            [1.0, 2.0],
            var_name="time",
            standard_name="time",
            units="days since 1950-01-01",
        )
        self.raw_cubes = []
        self._add_cube([1.0, 2.0], [1.0, 2.0])
        self._add_cube([3.0, 4.0], [3.0, 4.0])
        self._add_cube([5.0, 6.0], [5.0, 6.0])

    def _add_cube(self, data, coord):
        self.raw_cubes.append(
            Cube(
                data,
                var_name="sample",
                dim_coords_and_dims=((self._model_coord.copy(coord), 0),),
            ),
        )

    def test_concatenate(self):
        """Test concatenation of two cubes."""
        concatenated = concatenate(self.raw_cubes)
        np.testing.assert_array_equal(
            concatenated.coord("time").points,
            np.array([1, 2, 3, 4, 5, 6]),
        )

    def test_concatenate_empty_cubes(self):
        """Test concatenation with empty :class:`iris.cube.CubeList`."""
        empty_cubes = CubeList([])
        result = concatenate(empty_cubes)
        assert result is empty_cubes

    def test_concatenate_noop(self):
        """Test concatenation of a single cube."""
        concatenated = concatenate([self.raw_cubes[0]])
        np.testing.assert_array_equal(
            concatenated.coord("time").points,
            np.array([1, 2]),
        )

    def test_concatenate_with_overlap(
        self,
    ):
        """Test concatenation of time overalapping cubes."""
        self._add_cube([6.5, 7.5], [6.0, 7.0])
        concatenated = concatenate(self.raw_cubes)
        np.testing.assert_array_equal(
            concatenated.coord("time").points,
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        )
        np.testing.assert_array_equal(
            concatenated.data,
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 7.5]),
        )

    def test_remove_time_overlap_noop(self):
        """Test time handling of a single cube."""
        cubes = [self.raw_cubes[0]]
        result = _remove_time_overlaps(cubes)
        assert result is cubes

    def test_concatenate_with_overlap_2(self):
        """Test a more generic case."""
        self._add_cube([65.0, 75.0, 100.0], [9.0, 10.0, 11.0])
        self._add_cube([65.0, 75.0, 100.0], [7.0, 8.0, 9.0])
        concatenated = concatenate(self.raw_cubes)
        np.testing.assert_array_equal(
            concatenated.coord("time").points,
            np.array(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            ),
        )

    def test_concatenate_with_overlap_3(self):
        """Test a more generic case."""
        self._add_cube([65.0, 75.0, 100.0], [9.0, 10.0, 11.0])
        self._add_cube(
            [65.0, 75.0, 100.0, 100.0, 100.0, 112.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        concatenated = concatenate(self.raw_cubes)
        np.testing.assert_array_equal(
            concatenated.coord("time").points,
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                ],
            ),
        )

    def test_concatenate_with_overlap_same_start(self):
        """Test a more generic case."""
        cube1 = self.raw_cubes[0]
        raw_cubes = [
            cube1,
        ]
        time_coord = DimCoord(
            [1.0, 7.0],
            var_name="time",
            standard_name="time",
            units="days since 1950-01-01",
        )
        raw_cubes.append(
            Cube(
                [33.0, 55.0],
                var_name="sample",
                dim_coords_and_dims=((time_coord, 0),),
            ),
        )
        concatenated = concatenate(raw_cubes)
        np.testing.assert_array_equal(
            concatenated.coord("time").points,
            np.array([1.0, 7.0]),
        )
        raw_cubes.reverse()
        concatenated = concatenate(raw_cubes)
        np.testing.assert_array_equal(
            concatenated.coord("time").points,
            np.array([1.0, 7.0]),
        )

    def test_concatenate_with_iris_exception(self):
        """Test a more generic case."""
        time_coord_1 = DimCoord(
            [1.5, 5.0, 7.0],
            var_name="time",
            standard_name="time",
            units="days since 1950-01-01",
        )
        cube1 = Cube(
            [33.0, 55.0, 77.0],
            var_name="sample",
            dim_coords_and_dims=((time_coord_1, 0),),
        )
        time_coord_2 = DimCoord(
            [1.0, 5.0, 7.0],
            var_name="time",
            standard_name="time",
            units="days since 1950-01-01",
        )
        cube2 = Cube(
            [33.0, 55.0, 77.0],
            var_name="sample",
            dim_coords_and_dims=((time_coord_2, 0),),
        )
        cubes_single_ovlp = [cube2, cube1]
        cubess = concatenate(cubes_single_ovlp)
        # this tests the scalar to vector cube conversion too
        time_points = cubess.coord("time").core_points()
        np.testing.assert_array_equal(time_points, [1.0, 1.5, 5.0, 7.0])

    def test_concatenate_no_time_coords(self):
        """Test a more generic case."""
        time_coord_1 = DimCoord(
            [1.5, 5.0, 7.0],
            var_name="time",
            standard_name="time",
            units="days since 1950-01-01",
        )
        cube1 = Cube(
            [33.0, 55.0, 77.0],
            var_name="sample",
            dim_coords_and_dims=((time_coord_1, 0),),
        )
        ap_coord_2 = DimCoord(
            [1.0, 5.0, 7.0],
            var_name="air_pressure",
            standard_name="air_pressure",
            units="m",
            attributes={"positive": "down"},
        )
        cube2 = Cube(
            [33.0, 55.0, 77.0],
            var_name="sample",
            dim_coords_and_dims=((ap_coord_2, 0),),
        )
        with self.assertRaises(ValueError):
            concatenate([cube1, cube2])

    def test_concatenate_with_order(self):
        """Test a more generic case."""
        time_coord_1 = DimCoord(
            [1.5, 2.0, 5.0, 7.0],
            var_name="time",
            standard_name="time",
            units="days since 1950-01-01",
        )
        cube1 = Cube(
            [33.0, 44.0, 55.0, 77.0],
            var_name="sample",
            dim_coords_and_dims=((time_coord_1, 0),),
        )
        time_coord_2 = DimCoord(
            [1.0, 2.0, 5.0, 7.0, 100.0],
            var_name="time",
            standard_name="time",
            units="days since 1950-01-01",
        )
        cube2 = Cube(
            [33.0, 44.0, 55.0, 77.0, 1000.0],
            var_name="sample",
            dim_coords_and_dims=((time_coord_2, 0),),
        )
        cubes_ordered = [cube2, cube1]
        concatenated = concatenate(cubes_ordered)
        np.testing.assert_array_equal(
            concatenated.coord("time").points,
            np.array([1.0, 2.0, 5.0, 7.0, 100.0]),
        )
        cubes_reverse = [cube1, cube2]
        concatenated = concatenate(cubes_reverse)
        np.testing.assert_array_equal(
            concatenated.coord("time").points,
            np.array([1.0, 2.0, 5.0, 7.0, 100.0]),
        )

    def test_concatenate_by_experiment_first(self):
        """Test that data from experiments does not get mixed."""
        historical_1 = Cube(
            np.zeros(2),
            dim_coords_and_dims=(
                [
                    DimCoord(
                        np.arange(2),
                        var_name="time",
                        standard_name="time",
                        units="days since 1950-01-01",
                    ),
                    0,
                ],
            ),
            attributes={"experiment_id": "historical"},
        )
        historical_2 = historical_1.copy()
        historical_2.coord("time").points = np.arange(2, 4)
        historical_3 = historical_1.copy()
        historical_3.coord("time").points = np.arange(4, 6)
        ssp585_1 = historical_1.copy(np.ones(2))
        ssp585_1.coord("time").points = np.arange(3, 5)
        ssp585_1.attributes["experiment_id"] = "ssp585"
        ssp585_2 = ssp585_1.copy()
        ssp585_2.coord("time").points = np.arange(5, 7)
        result = concatenate(
            [historical_1, historical_2, historical_3, ssp585_1, ssp585_2],
        )
        assert_array_equal(result.coord("time").points, np.arange(7))
        assert_array_equal(result.data, np.array([0, 0, 0, 1, 1, 1, 1]))

    def test_concatenate_remove_unwanted_attributes(self):
        """Test concatenate removes unwanted attributes."""
        attributes = ("history", "creation_date", "tracking_id", "comment")
        for i, cube in enumerate(self.raw_cubes):
            for attr in attributes:
                cube.attributes[attr] = f"{attr}-{i}"
        concatenated = concatenate(self.raw_cubes)
        assert not set(attributes) & set(concatenated.attributes)

    def test_concatenate_remove_unwanted_attributes_from_coords(self):
        """Test concatenate removes unwanted attributes from coords."""
        attributes = ("history",)
        for i, cube in enumerate(self.raw_cubes):
            for coord in cube.coords():
                for attr in attributes:
                    coord.attributes[attr] = f"{attr}-{i}"
        concatenated = concatenate(self.raw_cubes)
        for coord in concatenated.coords():
            assert not set(attributes) & set(coord.attributes)

    def test_concatenate_differing_attributes(self):
        """Test concatenation of cubes with different attributes."""
        cubes = CubeList(self.raw_cubes)
        for idx, cube in enumerate(cubes):
            cube.attributes = {
                "equal_attr": 1,
                "different_attr": 3 - idx,
            }
        concatenated = concatenate(cubes)
        np.testing.assert_array_equal(
            concatenated.coord("time").points,
            np.array([1, 2, 3, 4, 5, 6]),
        )
        self.assertEqual(
            concatenated.attributes,
            {"equal_attr": 1, "different_attr": "1 2 3"},
        )

    def test_convert_calendar_concatenate_with_overlap(self):
        """Test compatible calendars get converted."""
        time_coord = DimCoord(
            [4.0, 5.0],
            var_name="time",
            standard_name="time",
            units=Unit(
                "days since 1950-01-01",
                calendar="proleptic_gregorian",
            ),
        )
        self.raw_cubes.append(
            Cube(
                [33.0, 55.0],
                var_name="sample",
                dim_coords_and_dims=((time_coord, 0),),
            ),
        )
        concatenated = concatenate(self.raw_cubes)
        assert concatenated.coord("time").units.calendar == "standard"

    def test_fail_on_calendar_concatenate_with_overlap(self):
        """Test fail of concatenation with overlap."""
        time_coord = DimCoord(
            [3.0, 7000.0],
            var_name="time",
            standard_name="time",
            units=Unit("days since 1950-01-01", calendar="360_day"),
        )
        self.raw_cubes.append(
            Cube(
                [33.0, 55.0],
                var_name="sample",
                dim_coords_and_dims=((time_coord, 0),),
            ),
        )
        with self.assertRaises(TypeError):
            concatenate(self.raw_cubes)

    def test_fail_metadata_differs(self):
        """Test exception raised if two cubes have different metadata."""
        self.raw_cubes[0].units = "m"
        self.raw_cubes[1].units = "K"
        with self.assertRaises(ValueError):
            concatenate(self.raw_cubes)
