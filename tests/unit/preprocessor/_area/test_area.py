"""Unit tests for the :func:`esmvalcore.preprocessor._area` module."""

import unittest

import iris
import numpy as np
import pytest
from cf_units import Unit

import tests
from esmvalcore.preprocessor._area import (
    area_statistics,
    extract_named_regions,
    extract_region,
)


class Test(tests.Test):
    """Test class for the :func:`esmvalcore.preprocessor._area_pp` module."""
    def setUp(self):
        """Prepare tests."""
        self.coord_sys = iris.coord_systems.GeogCS(
            iris.fileformats.pp.EARTH_RADIUS)
        data = np.ones((5, 5))
        lons = iris.coords.DimCoord(
            [i + .5 for i in range(5)],
            standard_name='longitude',
            bounds=[[i, i + 1.] for i in range(5)],  # [0,1] to [4,5]
            units='degrees_east',
            coord_system=self.coord_sys)
        lats = iris.coords.DimCoord(
            [i + .5 for i in range(5)],
            standard_name='latitude',
            bounds=[[i, i + 1.] for i in range(5)],
            units='degrees_north',
            coord_system=self.coord_sys,
        )
        coords_spec = [(lats, 0), (lons, 1)]
        self.grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)

        ndata = np.ones((6, 6))
        nlons = iris.coords.DimCoord(
            [i - 2.5 for i in range(6)],
            standard_name='longitude',
            bounds=[[i - 3., i - 2.] for i in range(6)],  # [3,2] to [4,5]
            units='degrees_east',
            coord_system=self.coord_sys)
        nlats = iris.coords.DimCoord(
            [i - 2.5 for i in range(6)],
            standard_name='latitude',
            bounds=[[i - 3., i - 2.] for i in range(6)],
            units='degrees_north',
            coord_system=self.coord_sys,
        )
        coords_spec = [(nlats, 0), (nlons, 1)]
        self.negative_grid = iris.cube.Cube(ndata,
                                            dim_coords_and_dims=coords_spec)

    def test_area_statistics_mean(self):
        """Test for area average of a 2D field."""
        result = area_statistics(self.grid, 'mean')
        expected = np.array([1.])
        self.assertArrayEqual(result.data, expected)

    def test_area_statistics_min(self):
        """Test for area average of a 2D field."""
        result = area_statistics(self.grid, 'min')
        expected = np.array([1.])
        self.assertArrayEqual(result.data, expected)

    def test_area_statistics_max(self):
        """Test for area average of a 2D field."""
        result = area_statistics(self.grid, 'max')
        expected = np.array([1.])
        self.assertArrayEqual(result.data, expected)

    def test_area_statistics_median(self):
        """Test for area average of a 2D field."""
        result = area_statistics(self.grid, 'median')
        expected = np.array([1.])
        self.assertArrayEqual(result.data, expected)

    def test_area_statistics_std_dev(self):
        """Test for area average of a 2D field."""
        result = area_statistics(self.grid, 'std_dev')
        expected = np.array([0.])
        self.assertArrayEqual(result.data, expected)

    def test_area_statistics_variance(self):
        """Test for area average of a 2D field."""
        result = area_statistics(self.grid, 'variance')
        expected = np.array([0.])
        self.assertArrayEqual(result.data, expected)

    def test_area_statistics_neg_lon(self):
        """Test for area average of a 2D field."""
        result = area_statistics(self.negative_grid, 'mean')
        expected = np.array([1.])
        self.assertArrayEqual(result.data, expected)

    def test_extract_region(self):
        """Test for extracting a region from a 2D field."""
        result = extract_region(self.grid, 1.5, 2.5, 1.5, 2.5)
        # expected outcome
        expected = np.ones((2, 2))
        self.assertArrayEqual(result.data, expected)

    def test_extract_region_neg_lon(self):
        """Test for extracting a region with a negative longitude field."""
        result = extract_region(self.negative_grid, -0.5, 0.5, -0.5, 0.5)
        expected = np.ones((2, 2))
        self.assertArrayEqual(result.data, expected)

    def test_extract_named_region(self):
        """Test for extracting a named region."""
        # tests:
        # Create a cube with regions
        times = np.array([15., 45., 75.])
        bounds = np.array([[0., 30.], [30., 60.], [60., 90.]])
        time = iris.coords.DimCoord(
            times,
            bounds=bounds,
            standard_name='time',
            units=Unit('days since 1950-01-01', calendar='gregorian'),
        )

        regions = ['region1', 'region2', 'region3']
        region = iris.coords.AuxCoord(
            regions,
            standard_name='region',
            units='1',
        )

        data = np.ones((3, 3))
        region_cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[(time, 0)],
            aux_coords_and_dims=[(region, 1)],
        )

        # test string region
        result1 = extract_named_regions(region_cube, 'region1')
        expected = np.ones((3, ))
        self.assertArrayEqual(result1.data, expected)

        # test list of regions
        result2 = extract_named_regions(region_cube, ['region1', 'region2'])
        expected = np.ones((3, 2))
        self.assertArrayEqual(result2.data, expected)

        # test for expected failures:
        with self.assertRaises(ValueError):
            extract_named_regions(region_cube, 'reg_A')
            extract_named_regions(region_cube, ['region1', 'reg_A'])


@pytest.fixture
def irregular_grid_cube():
    """Create test cube on irregular grid."""
    # Define grid and data
    data = np.arange(18, dtype=np.float32).reshape((2, 3, 3))
    lats = np.array(
        [
            [0.0, 0.0, 0.1],
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.0],
        ],
        dtype=np.float64,
    )
    lons = np.array(
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ],
        dtype=np.float64,
    )

    times = iris.coords.DimCoord(np.array([10, 20], dtype=np.float64),
                                 standard_name='time',
                                 units=Unit('days since 1950-01-01',
                                            calendar='gregorian'))

    # Construct cube
    nlat = iris.coords.DimCoord(range(data.shape[1]), var_name='nlat')
    nlon = iris.coords.DimCoord(range(data.shape[2]), var_name='nlon')
    lat = iris.coords.AuxCoord(lats,
                               var_name='lat',
                               standard_name='latitude',
                               units='degrees')
    lon = iris.coords.AuxCoord(lons,
                               var_name='lon',
                               standard_name='longitude',
                               units='degrees')
    dim_coord_spec = [
        (times, 0),
        (nlat, 1),
        (nlon, 2),
    ]
    aux_coord_spec = [
        (lat, [1, 2]),
        (lon, [1, 2]),
    ]
    cube = iris.cube.Cube(
        data,
        var_name='tos',
        units='K',
        dim_coords_and_dims=dim_coord_spec,
        aux_coords_and_dims=aux_coord_spec,
    )
    return cube


IRREGULAR_TEST_CASES = [
    {
        'region': (0.5, 1.5, 0.5, 3),
        'mask': np.array(
            [
                [False],
                [False],
            ],
            dtype=bool,
        ),
        'data': np.arange(18, dtype=np.float32).reshape((2, 3, 3))[:, 1:3, 1:2]
    },
    {
        'region': (1, 2, 1, 2),
        'mask': np.array(
            [
                [False, False],
                [True, False],
            ],
            dtype=bool,
        ),
        'data': np.arange(18, dtype=np.float32).reshape((2, 3, 3))[:, 1:3, 1:3]
    },
    {
        'region': (-0.5, 4, 0, 0.5),
        'mask': np.array(
            [
                [False, False, False],
            ],
            dtype=bool,
        ),
        'data': np.arange(18, dtype=np.float32).reshape((2, 3, 3))[:, :1, :]
    },
]


@pytest.mark.parametrize('case', IRREGULAR_TEST_CASES)
def test_extract_region_irregular(irregular_grid_cube, case):
    """Test `extract_region` with data on an irregular grid."""
    start_lon, end_lon, start_lat, end_lat = case['region']
    cube = extract_region(
        irregular_grid_cube,
        start_longitude=start_lon,
        end_longitude=end_lon,
        start_latitude=start_lat,
        end_latitude=end_lat,
    )

    for i in range(2):
        np.testing.assert_array_equal(cube.data[i].mask, case['mask'])
    np.testing.assert_array_equal(cube.data.data, case['data'])


if __name__ == '__main__':
    unittest.main()
