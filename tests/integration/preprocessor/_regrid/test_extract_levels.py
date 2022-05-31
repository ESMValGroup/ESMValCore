"""
Integration tests for the :func:`esmvalcore.preprocessor.regrid.extract_levels`
function.

"""

import unittest

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
        coord = iris.coords.DimCoord(0, standard_name='realization')
        cube.add_aux_coord(coord)
        cubes.append(cube)
        # Create second realization cube.
        cube = _make_cube(data + np.prod(shape))
        coord = iris.coords.DimCoord(1, standard_name='realization')
        cube.add_aux_coord(coord)
        cubes.append(cube)
        # Create a 4d synthetic test cube.
        self.cube = cubes.merge_cube()
        coord = self.cube.coord(axis='z', dim_coords=True)
        self.shape = list(self.cube.shape)
        [self.z_dim] = self.cube.coord_dims(coord)

    def test_nop__levels_match(self):
        vcoord = _make_vcoord(self.z)
        self.assertEqual(self.cube.coord(axis='z', dim_coords=True), vcoord)
        levels = vcoord.points
        result = extract_levels(self.cube, levels, 'linear')
        self.assertEqual(result, self.cube)
        self.assertEqual(id(result), id(self.cube))

    def test_levels_almost_match(self):
        vcoord = self.cube.coord(axis='z', dim_coords=True)
        levels = np.array(vcoord.points, dtype=float)
        vcoord.points = vcoord.points + 1.e-7
        result = extract_levels(self.cube, levels, 'linear')
        self.assert_array_equal(vcoord.points, levels)
        self.assertTrue(result is self.cube)

    def test_interpolation__linear(self):
        levels = [0.5, 1.5]
        scheme = 'linear'
        result = extract_levels(self.cube, levels, scheme)
        expected = np.array([[[[2., 3.], [4., 5.]], [[6., 7.], [8., 9.]]],
                             [[[14., 15.], [16., 17.]], [[18., 19.],
                                                         [20., 21.]]]])
        self.assert_array_equal(result.data, expected)
        self.shape[self.z_dim] = len(levels)
        self.assertEqual(result.shape, tuple(self.shape))

    def test_interpolation__nearest(self):
        levels = [0.49, 1.51]
        scheme = 'nearest'
        result = extract_levels(self.cube, levels, scheme)
        expected = np.array([[[[0., 1.], [2., 3.]], [[8., 9.], [10., 11.]]],
                             [[[12., 13.], [14., 15.]], [[20., 21.],
                                                         [22., 23.]]]])
        self.assert_array_equal(result.data, expected)
        self.shape[self.z_dim] = len(levels)
        self.assertEqual(result.shape, tuple(self.shape))

    def test_interpolation__extrapolated_nan_filling(self):
        levels = [-10, 1, 2, 10]
        scheme = 'nearest'
        result = extract_levels(self.cube, levels, scheme)
        expected = np.array([[[[_MDI, _MDI], [_MDI, _MDI]], [[4., 5.],
                                                             [6., 7.]],
                              [[8., 9.], [10., 11.]],
                              [[_MDI, _MDI], [_MDI, _MDI]]],
                             [[[_MDI, _MDI], [_MDI, _MDI]],
                              [[16., 17.], [18., 19.]], [[20., 21.],
                                                         [22., 23.]],
                              [[_MDI, _MDI], [_MDI, _MDI]]]])
        expected_mask = np.array([[[[True, True], [True, True]],
                                   [[False, False], [False, False]],
                                   [[False, False], [False, False]],
                                   [[True, True], [True, True]]],
                                  [[[True, True], [True, True]],
                                   [[False, False], [False, False]],
                                   [[False, False], [False, False]],
                                   [[True, True], [True, True]]]])
        expected = np.ma.array(expected, mask=expected_mask)
        self.assert_array_equal(result.data, expected)
        self.shape[self.z_dim] = len(levels)
        self.assertEqual(result.shape, tuple(self.shape))

    def test_interpolation__scalar_collapse(self):
        level = 1
        scheme = 'nearest'
        result = extract_levels(self.cube, level, scheme)
        expected = np.array([[[4., 5.], [6., 7.]], [[16., 17.], [18., 19.]]])
        self.assert_array_equal(result.data, expected)
        del self.shape[self.z_dim]
        self.assertEqual(result.shape, tuple(self.shape))

    def test_add_alt_coord(self):
        assert self.cube.coords('air_pressure')
        assert not self.cube.coords('altitude')
        result = extract_levels(self.cube, [1, 2],
                                'linear_extrapolate',
                                coordinate='altitude')
        assert not result.coords('air_pressure')
        assert result.coords('altitude')
        assert result.shape == (2, 2, 2, 2)
        np.testing.assert_allclose(result.coord('altitude').points,
                                   [1.0, 2.0])

    def test_add_plev_coord(self):
        self.cube.coord('air_pressure').standard_name = 'altitude'
        self.cube.coord('altitude').var_name = 'alt'
        self.cube.coord('altitude').long_name = 'altitude'
        self.cube.coord('altitude').units = 'm'
        assert not self.cube.coords('air_pressure')
        assert self.cube.coords('altitude')
        result = extract_levels(self.cube, [1, 2],
                                'linear_extrapolate',
                                coordinate='air_pressure')
        assert result.coords('air_pressure')
        assert not result.coords('altitude')
        assert result.shape == (2, 2, 2, 2)
        np.testing.assert_allclose(result.coord('air_pressure').points,
                                   [1.0, 2.0])


if __name__ == '__main__':
    unittest.main()
