"""Unit test for the :func:`esmvalcore.preprocessor._mask` function."""

import unittest

import iris
import iris.fileformats
import numpy as np
from cf_units import Unit

import tests
from esmvalcore.preprocessor._mask import (
    _get_fx_mask,
    count_spells,
    mask_above_threshold,
    mask_below_threshold,
    mask_glaciated,
    mask_inside_range,
    mask_outside_range,
)


class Test(tests.Test):
    """Test class for _mask."""

    def setUp(self):
        """Prepare tests."""
        coord_sys = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
        self.data2 = np.array([[0.0, 1.0], [2.0, 3.0]])
        # Two points near the south pole and two points in the southern ocean
        lons2 = iris.coords.DimCoord(
            [1.5, 2.5],
            standard_name="longitude",
            bounds=[[1.0, 2.0], [2.0, 3.0]],
            units="degrees_east",
            coord_system=coord_sys,
        )
        lats2 = iris.coords.DimCoord(
            [-89.5, -70],
            standard_name="latitude",
            bounds=[[-90.0, -89.0], [-70.5, -69.5]],
            units="degrees_north",
            coord_system=coord_sys,
        )
        coords_spec3 = [(lats2, 0), (lons2, 1)]
        self.arr = iris.cube.Cube(self.data2, dim_coords_and_dims=coords_spec3)
        self.time_cube = iris.cube.Cube(
            np.arange(1, 25),
            var_name="co2",
            units="J",
        )
        self.time_cube.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(15.0, 720.0, 30.0),
                standard_name="time",
                units=Unit(
                    "days since 1950-01-01 00:00:00",
                    calendar="gregorian",
                ),
            ),
            0,
        )
        self.fx_data = np.array([20.0, 60.0, 50.0])

    def test_count_spells(self):
        """Test count_spells func."""
        ref_spells = count_spells(self.time_cube.data, -1000.0, 0, 1)
        np.testing.assert_equal(24, ref_spells)
        ref_spells = count_spells(self.time_cube.data, -1000.0, 0, 2)
        np.testing.assert_equal(12, ref_spells)

    def test_get_fx_mask(self):
        """Test _get_fx_mask func."""
        # Test getting land and sea mask from sftlf
        computed = _get_fx_mask(self.fx_data, "land", "sftlf")
        expected = np.array([False, True, False])
        self.assert_array_equal(expected, computed)
        computed = _get_fx_mask(self.fx_data, "sea", "sftlf")
        expected = np.array([True, False, True])
        self.assert_array_equal(expected, computed)
        # Test getting land and sea mask from sftof
        computed = _get_fx_mask(self.fx_data, "land", "sftof")
        expected = np.array([True, False, False])
        self.assert_array_equal(expected, computed)
        computed = _get_fx_mask(self.fx_data, "sea", "sftof")
        expected = np.array([False, True, True])
        self.assert_array_equal(expected, computed)
        # Test getting ice and landsea mask from sftlf
        computed = _get_fx_mask(self.fx_data, "ice", "sftgif")
        expected = np.array([False, True, False])
        self.assert_array_equal(expected, computed)
        computed = _get_fx_mask(self.fx_data, "landsea", "sftgif")
        expected = np.array([True, False, True])
        self.assert_array_equal(expected, computed)

    def test_mask_glaciated(self):
        """Test to mask glaciated (NE mask)."""
        result = mask_glaciated(self.arr, mask_out="glaciated")
        expected = np.ma.masked_array(
            self.data2,
            mask=np.array([[True, True], [False, False]]),
        )
        self.assert_array_equal(result.data, expected)

    def test_mask_above_threshold(self):
        """Test to mask above a threshold."""
        result = mask_above_threshold(self.arr, 1.5)
        expected = np.ma.array(self.data2, mask=[[False, False], [True, True]])
        self.assert_array_equal(result.data, expected)

    def test_mask_below_threshold(self):
        """Test to mask below a threshold."""
        result = mask_below_threshold(self.arr, 1.5)
        expected = np.ma.array(self.data2, mask=[[True, True], [False, False]])
        self.assert_array_equal(result.data, expected)

    def test_mask_inside_range(self):
        """Test to mask inside a range."""
        result = mask_inside_range(self.arr, 0.5, 2.5)
        expected = np.ma.array(self.data2, mask=[[False, True], [True, False]])
        self.assert_array_equal(result.data, expected)

    def test_mask_outside_range(self):
        """Test to mask outside a range."""
        result = mask_outside_range(self.arr, 0.5, 2.5)
        expected = np.ma.array(self.data2, mask=[[True, False], [False, True]])
        self.assert_array_equal(result.data, expected)


if __name__ == "__main__":
    unittest.main()
