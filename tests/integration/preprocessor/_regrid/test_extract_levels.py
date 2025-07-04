"""Tests for :func:`esmvalcore.preprocessor.regrid.extract_levels` function."""

import unittest

import dask.array as da
import iris
import numpy as np

import tests
from esmvalcore.preprocessor._regrid import _MDI, extract_levels
from tests.unit.preprocessor._regrid import _make_cube, _make_vcoord


class Test(tests.Test):
    def setUp(self):
        """Prepare tests."""
        shape = (3, 2, 2)
        self.z = shape[0]
        data = np.arange(np.prod(shape)).reshape(shape)
        cubes = iris.cube.CubeList()
        # Create first realization cube.
        cube = _make_cube(data)
        coord = iris.coords.DimCoord(0, standard_name="realization")
        cube.add_aux_coord(coord)
        cubes.append(cube)
        # Create second realization cube.
        cube = _make_cube(data + np.prod(shape))
        coord = iris.coords.DimCoord(1, standard_name="realization")
        cube.add_aux_coord(coord)
        cubes.append(cube)
        # Create a 4d synthetic test cube.
        self.cube = cubes.merge_cube()
        coord = self.cube.coord(axis="z", dim_coords=True)
        self.shape = list(self.cube.shape)
        [self.z_dim] = self.cube.coord_dims(coord)

    def test_nop__levels_match(self):
        vcoord = _make_vcoord(self.z)
        self.assertEqual(self.cube.coord(axis="z", dim_coords=True), vcoord)
        levels = vcoord.points
        result = extract_levels(self.cube, levels, "linear")
        self.assertEqual(result, self.cube)
        self.assertEqual(id(result), id(self.cube))

    def test_levels_almost_match(self):
        vcoord = self.cube.coord(axis="z", dim_coords=True)
        levels = np.array(vcoord.points, dtype=float)
        vcoord.points = vcoord.points + 1.0e-7
        result = extract_levels(self.cube, levels, "linear")
        self.assert_array_equal(vcoord.points, levels)
        self.assertTrue(result is self.cube)

    def test_interpolation__linear(self):
        levels = [0.5, 1.5]
        scheme = "linear"
        result = extract_levels(self.cube, levels, scheme)
        expected = np.ma.array(
            [
                [
                    [[2.0, 3.0], [4.0, 5.0]],
                    [[6.0, 7.0], [8.0, 9.0]],
                ],
                [
                    [[14.0, 15.0], [16.0, 17.0]],
                    [[18.0, 19.0], [20.0, 21.0]],
                ],
            ],
        )
        self.assert_array_equal(result.data, expected)
        self.shape[self.z_dim] = len(levels)
        self.assertEqual(result.shape, tuple(self.shape))

    def test_interpolation__linear_lazy(self):
        levels = [0.5, 1.5]
        scheme = "linear"
        cube = self.cube.copy(self.cube.lazy_data())
        coord_name = "multidimensional_vertical_coord"
        coord_points = cube.coord("air_pressure").core_points().reshape(
            3,
            1,
            1,
        ) * np.ones((3, 2, 2))
        cube.add_aux_coord(
            iris.coords.AuxCoord(
                da.asarray(coord_points),
                long_name=coord_name,
            ),
            [1, 2, 3],
        )
        result = extract_levels(cube, levels, scheme, coordinate=coord_name)
        self.assertTrue(result.has_lazy_data())
        self.assertTrue(cube.coord(coord_name).has_lazy_points())
        expected = np.ma.array(
            [
                [
                    [[2.0, 3.0], [4.0, 5.0]],
                    [[6.0, 7.0], [8.0, 9.0]],
                ],
                [
                    [[14.0, 15.0], [16.0, 17.0]],
                    [[18.0, 19.0], [20.0, 21.0]],
                ],
            ],
        )
        self.assert_array_equal(result.data, expected)

    def test_interpolation__nearest(self):
        levels = [0.49, 1.51]
        scheme = "nearest"
        result = extract_levels(self.cube, levels, scheme)
        expected = np.ma.array(
            [
                [
                    [[0.0, 1.0], [2.0, 3.0]],
                    [[8.0, 9.0], [10.0, 11.0]],
                ],
                [
                    [[12.0, 13.0], [14.0, 15.0]],
                    [[20.0, 21.0], [22.0, 23.0]],
                ],
            ],
        )
        self.assert_array_equal(result.data, expected)
        self.shape[self.z_dim] = len(levels)
        self.assertEqual(result.shape, tuple(self.shape))

    def test_interpolation__extrapolated_nan_filling(self):
        levels = [-10, 1, 2, 10]
        scheme = "nearest"
        result = extract_levels(self.cube, levels, scheme)
        expected = np.array(
            [
                [
                    [[_MDI, _MDI], [_MDI, _MDI]],
                    [[4.0, 5.0], [6.0, 7.0]],
                    [[8.0, 9.0], [10.0, 11.0]],
                    [[_MDI, _MDI], [_MDI, _MDI]],
                ],
                [
                    [[_MDI, _MDI], [_MDI, _MDI]],
                    [[16.0, 17.0], [18.0, 19.0]],
                    [[20.0, 21.0], [22.0, 23.0]],
                    [[_MDI, _MDI], [_MDI, _MDI]],
                ],
            ],
        )
        expected_mask = np.array(
            [
                [
                    [[True, True], [True, True]],
                    [[False, False], [False, False]],
                    [[False, False], [False, False]],
                    [[True, True], [True, True]],
                ],
                [
                    [[True, True], [True, True]],
                    [[False, False], [False, False]],
                    [[False, False], [False, False]],
                    [[True, True], [True, True]],
                ],
            ],
        )
        expected = np.ma.array(expected, mask=expected_mask)
        self.assert_array_equal(result.data, expected)
        self.shape[self.z_dim] = len(levels)
        self.assertEqual(result.shape, tuple(self.shape))

    def test_interpolation__scalar_collapse(self):
        level = 1
        scheme = "nearest"
        result = extract_levels(self.cube, level, scheme)
        expected = np.array(
            [[[4.0, 5.0], [6.0, 7.0]], [[16.0, 17.0], [18.0, 19.0]]],
        )
        self.assert_array_equal(result.data, expected)
        del self.shape[self.z_dim]
        self.assertEqual(result.shape, tuple(self.shape))

    def test_add_alt_coord(self):
        assert self.cube.coords("air_pressure")
        assert not self.cube.coords("altitude")
        result = extract_levels(
            self.cube,
            [1, 2],
            "linear_extrapolate",
            coordinate="altitude",
        )
        assert not result.coords("air_pressure")
        assert result.coords("altitude")
        assert result.shape == (2, 2, 2, 2)
        np.testing.assert_allclose(result.coord("altitude").points, [1.0, 2.0])

    def test_add_plev_coord(self):
        self.cube.coord("air_pressure").standard_name = "altitude"
        self.cube.coord("altitude").var_name = "alt"
        self.cube.coord("altitude").long_name = "altitude"
        self.cube.coord("altitude").units = "m"
        assert not self.cube.coords("air_pressure")
        assert self.cube.coords("altitude")
        result = extract_levels(
            self.cube,
            [1, 2],
            "linear_extrapolate",
            coordinate="air_pressure",
        )
        assert result.coords("air_pressure")
        assert not result.coords("altitude")
        assert result.shape == (2, 2, 2, 2)
        np.testing.assert_allclose(
            result.coord("air_pressure").points,
            [1.0, 2.0],
        )


if __name__ == "__main__":
    unittest.main()
