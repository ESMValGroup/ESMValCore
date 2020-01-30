"""Unit tests for the :func:`esmvalcore.preprocessor._area` module."""

import unittest

import fiona
import iris
import numpy as np
import pytest
from cf_units import Unit
from shapely.geometry import Polygon, mapping

import tests
from esmvalcore.preprocessor._area import (_crop_cube, area_statistics,
                                           extract_named_regions,
                                           extract_region, extract_shape)


def calc_grid_area():
    """
    Creates a regular grid with the same size as the fx file below.
    """
    coord_sys = iris.coord_systems.GeogCS(
        iris.fileformats.pp.EARTH_RADIUS)
    data = np.ones((2, 2))
    lons = iris.coords.DimCoord(
        [-5., 5.],
        standard_name='longitude',
        bounds=[[-10., 0.], [0., 10.]],
        units='degrees_east',
        coord_system=coord_sys)
    lats = iris.coords.DimCoord(
        [-5., 5.],
        standard_name='latitude',
        bounds=[[-10., 0.], [0., 10.]],
        units='degrees_north',
        coord_system=coord_sys)
    coords_spec = [(lats, 0), (lons, 1)]
    grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)
    return iris.analysis.cartography.area_weights(grid)


def make_fx_file():
    """
    Create and save an FX file
    """
    i_axis = iris.coords.DimCoord(
        [i for i in range(2)],
        standard_name=None,
        var_name='i',
        long_name='cell index along first dimension'
        )
    j_axis = iris.coords.DimCoord(
        [i for i in range(2)],
        standard_name=None,
        var_name='j',
        long_name='cell index along second dimension'
        )

    lat2d = np.array([[-5., -5.], [5., 5.]])
    lon2d = np.array([[-5.,  5.], [-5., 5.]])

    lat2d_bounds = np.array([[[-10., -10., 0., 0.],
                              [-10., -10., 0., 0.]],
                             [[0., 0., 10., 10.],
                              [0., 0., 10., 10.]]])
    lon2d_bounds = np.array([[[-10., 0., 0., -10.],
                              [0., 10., 10., 0.]],
                             [[-10., 0., 0., -10.],
                             [0., 10., 10., 0.]]])

    coords_spec = [(i_axis, 0), (j_axis, 1)]
    data = calc_grid_area()

    gridir = iris.cube.Cube(data, var_name='areacello', units=Unit('m^2'),
                            dim_coords_and_dims=coords_spec)

    lat_coord = iris.coords.AuxCoord(lat2d, standard_name='latitude',
                                     long_name='latitude',
                                     var_name='latitude',
                                     units=Unit('degrees'),
                                     bounds=lat2d_bounds)
    lon_coord = iris.coords.AuxCoord(lon2d, standard_name='longitude',
                                     long_name='longitude',
                                     var_name='longitude',
                                     units=Unit('degrees'),
                                     bounds=lon2d_bounds)

    gridir.add_aux_coord(lat_coord, data_dims=(0, 1))
    gridir.add_aux_coord(lon_coord, data_dims=(0, 1))
    fn = 'fx_test_areacello.nc'
    iris.save(gridir, fn)
    return fn


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

        # Create Irregular grid.
        fx_file = make_fx_file()
        self.gridir = iris.load_cube(fx_file)
        self.gridir.data = np.array([[0., 1.], [2., 3.]])
        self.fx_files = {'areacello': fx_file}

    # Regular grid area_statistics tests:
    def test_area_statistics_mean(self):
        """Test for area average of a 2D field."""
        with self.assertRaises(ValueError):
            area_statistics(self.grid, 'mean', calculate_grid=False)

        result = area_statistics(self.grid, 'mean',  calculate_grid=True)
        expected = np.array([1.])
        self.assertArrayEqual(result.data, expected)

    def test_area_statistics_mean_calgrid(self):
        """Test for area average of a 2D field."""
        result = area_statistics(self.grid, 'mean', calculate_grid=True)
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

    def test_area_statistics_sum(self):
        """Test for sum of a 2D field."""
        result = area_statistics(self.grid, 'sum', calculate_grid=True)
        grid_areas = iris.analysis.cartography.area_weights(self.grid)
        expected = np.sum(grid_areas)
        self.assertArrayEqual(result.data, expected)

    def test_area_statistics_weightless_sum(self):
        """Test for weightless sum of a 2D field."""
        result = area_statistics(self.grid, 'weightless_sum')
        expected = np.sum(self.grid.data)
        self.assertArrayEqual(result.data, expected)

    def test_area_statistics_weightless_mean(self):
        """Test for weightless mean of a 2D field."""
        result = area_statistics(self.grid, 'weightless_mean')
        expected = np.mean(self.grid.data)
        self.assertArrayEqual(result.data, expected)

    def test_area_statistics_variance(self):
        """Test for area average of a 2D field."""
        result = area_statistics(self.grid, 'variance')
        expected = np.array([0.])
        self.assertArrayEqual(result.data, expected)

    # Negative grids:
    def test_area_statistics_neg_lon(self):
        """Test for area average of a 2D field."""
        result = area_statistics(self.negative_grid, 'mean',
                                 calculate_grid=True)
        expected = np.array([1.])
        self.assertArrayEqual(result.data, expected)

    # Irregular grid area_statistics tests:
    def test_area_statistics_mean_ir(self):
        """Test for area average of a 2D field."""
        # This should fail:
        with self.assertRaises(ValueError):
            area_statistics(self.gridir, 'mean', calculate_grid=False)

        # This should work:
        result = area_statistics(self.gridir, 'mean', fx_files=self.fx_files)
        expected = np.array([1.5])
        self.assertArrayEqual(result.data, expected)

    def test_area_statistics_sum_ir(self):
        """Test for area average of a 2D field."""
        # This should fail:
        self.assertRaises(iris.exceptions.CoordinateMultiDimError,
                          area_statistics, self.gridir, 'sum')
        # This should work:
        result = area_statistics(self.gridir, 'sum', fx_files=self.fx_files)
        expected = np.array(7381511118575.528)
        self.assertArrayAlmostEqual(result.data, expected, decimal=1)

    # extract region
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


def create_irregular_grid_cube(data, lons, lats):
    """Create test cube on irregular grid."""
    nlat = iris.coords.DimCoord(range(data.shape[0]), var_name='nlat')
    nlon = iris.coords.DimCoord(range(data.shape[1]), var_name='nlon')
    lat = iris.coords.AuxCoord(lats,
                               var_name='lat',
                               standard_name='latitude',
                               units='degrees')
    lon = iris.coords.AuxCoord(lons,
                               var_name='lon',
                               standard_name='longitude',
                               units='degrees')
    dim_coord_spec = [
        (nlat, 0),
        (nlon, 1),
    ]
    aux_coord_spec = [
        (lat, [0, 1]),
        (lon, [0, 1]),
    ]
    cube = iris.cube.Cube(
        data,
        var_name='tos',
        units='K',
        dim_coords_and_dims=dim_coord_spec,
        aux_coords_and_dims=aux_coord_spec,
    )
    return cube


IRREGULAR_EXTRACT_REGION_TESTS = [
    {
        'region': (10, 360, 0, 90),
        'mask':
        np.array(
            [
                [True, True, True],
                [True, True, False],
                [True, False, False],
            ],
            dtype=bool,
        ),
    },
    {
        'region': (200, 10, -90, -60),
        'mask':
        np.array(
            [
                [False, True, False],
                [True, True, True],
                [True, True, True],
            ],
            dtype=bool,
        ),
    },
    {
        'region': (-150, 50, 50, -50),
        'mask':
        np.array(
            [
                [False, True, False],
                [True, True, True],
                [False, True, False],
            ],
            dtype=bool,
        ),
    },
    {
        'region': (0, 0, -100, 0),
        'raises': "Invalid start_latitude: -100."
    },
    {
        'region': (0, 0, 0, 100),
        'raises': "Invalid end_latitude: -100."
    },
]


@pytest.fixture
def irregular_extract_region_cube():
    """Create a test cube on an irregular grid to test `extract_region`."""
    data = np.arange(9, dtype=np.float32).reshape((3, 3))
    lons = np.array(
        [
            [0, 120, 240],
            [0, 120, 240],
            [0, 120, 240],
        ],
        dtype=np.float64,
    )
    lats = np.array(
        [
            [-60, -61., -60],
            [0, -1, 0],
            [60, 60, 60],
        ],
        dtype=np.float64,
    )
    cube = create_irregular_grid_cube(data, lons, lats)
    return cube


@pytest.mark.parametrize('case', IRREGULAR_EXTRACT_REGION_TESTS)
def test_extract_region_irregular(irregular_extract_region_cube, case):
    """Test `extract_region` with data on an irregular grid."""
    start_lon, end_lon, start_lat, end_lat = case['region']
    if 'raises' not in case:
        cube = extract_region(
            irregular_extract_region_cube,
            start_longitude=start_lon,
            end_longitude=end_lon,
            start_latitude=start_lat,
            end_latitude=end_lat,
        )

        data = np.arange(9, dtype=np.float32).reshape((3, 3))
        np.testing.assert_array_equal(cube.data.mask, case['mask'])
        np.testing.assert_array_equal(cube.data.data, data)
    else:
        with pytest.raises(ValueError) as exc:
            extract_region(
                irregular_extract_region_cube,
                start_longitude=start_lon,
                end_longitude=end_lon,
                start_latitude=start_lat,
                end_latitude=end_lat,
            )
            assert exc.value == case['raises']


@pytest.fixture
def make_testcube():
    """Create a test cube on a Cartesian grid."""
    coord_sys = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
    data = np.ones((5, 5))
    lons = iris.coords.DimCoord(
        [i + .5 for i in range(5)],
        standard_name='longitude',
        bounds=[[i, i + 1.] for i in range(5)],  # [0,1] to [4,5]
        units='degrees_east',
        coord_system=coord_sys)
    lats = iris.coords.DimCoord([i + .5 for i in range(5)],
                                standard_name='latitude',
                                bounds=[[i, i + 1.] for i in range(5)],
                                units='degrees_north',
                                coord_system=coord_sys)
    coords_spec = [(lats, 0), (lons, 1)]
    return iris.cube.Cube(data, dim_coords_and_dims=coords_spec)


def write_shapefile(shape, path):
    """Write (a) shape(s) to a shapefile."""
    # Define a polygon feature geometry with one attribute
    schema = {
        'geometry': 'Polygon',
        'properties': {
            'id': 'int'
        },
    }
    if not isinstance(shape, list):
        shape = [shape]

    # Write a new Shapefile
    with fiona.open(path, 'w', 'ESRI Shapefile', schema) as file:
        for id_, s in enumerate(shape):
            file.write({
                'geometry': mapping(s),
                'properties': {
                    'id': id_
                },
            })


@pytest.fixture(params=[(2, 2), (1, 3), (9, 2)])
def square_shape(request, tmp_path):
    # Define polygons to test extract_shape
    slat = request.param[0]
    slon = request.param[1]
    polyg = Polygon([
        (1.0, 1.0 + slat),
        (1.0, 1.0),
        (1.0 + slon, 1.0),
        (1.0 + slon, 1.0 + slat),
    ])
    write_shapefile(polyg, tmp_path / 'test_shape.shp')

    # Make corresponding expected masked array
    (slat, slon) = np.ceil([slat, slon]).astype(int)
    vals = np.ones((min(slat + 2, 5), min(slon + 2, 5)))
    mask = vals.copy()
    mask[1:1 + slat, 1:1 + slon] = 0
    return np.ma.masked_array(vals, mask)


@pytest.fixture(params=[(2, 2, 1), (2, 2, 2), (1, 2, 3)])
def square_composite_shape(request, tmp_path):
    # Define polygons to test extract_shape
    slat = request.param[0]
    slon = request.param[1]
    nshape = request.param[2]
    polyg = []
    for n in range(nshape):
        polyg.append(
            Polygon([(1.0 + n, 1.0 + slat), (1.0 + n, 1.0),
                     (1.0 + n + slon, 1.0), (1.0 + n + slon, 1.0 + slat)]))
    write_shapefile(polyg, tmp_path / 'test_shape.shp')

    # Make corresponding expected masked array
    (slat, slon) = np.ceil([slat, slon]).astype(int)
    vals = np.ones((nshape, min(slat + 2, 5), min(slon + 1 + nshape, 5)))
    mask = vals.copy()
    for n in range(nshape):
        mask[n, 1:1 + slat, 1 + n:1 + n + slon] = 0
    return np.ma.masked_array(vals, mask)


def test_crop_cube(make_testcube, square_shape, tmp_path):
    """Test for cropping a cube by shape bounds."""
    with fiona.open(tmp_path / 'test_shape.shp') as geometries:
        result = _crop_cube(make_testcube, *geometries.bounds)
        expected = square_shape.data
        np.testing.assert_array_equal(result.data, expected)


@pytest.mark.parametrize('crop', [True, False])
def test_extract_shape(make_testcube, square_shape, tmp_path, crop):
    """Test for extracting a region with shapefile"""
    expected = square_shape
    if not crop:
        # If cropping is not used, embed expected in the original test array
        original = np.ma.ones((5, 5))
        original.mask = np.ones_like(original, dtype=bool)
        original[:expected.shape[0], :expected.shape[1]] = expected
        expected = original
    result = extract_shape(make_testcube,
                           tmp_path / 'test_shape.shp',
                           crop=crop)
    np.testing.assert_array_equal(result.data.data, expected.data)
    np.testing.assert_array_equal(result.data.mask, expected.mask)


@pytest.mark.parametrize('crop', [True, False])
@pytest.mark.parametrize('decomposed', [True, False])
def test_extract_composite_shape(make_testcube, square_composite_shape,
                                 tmp_path, crop, decomposed):
    """Test for extracting a region with shapefile"""
    expected = square_composite_shape
    if not crop:
        # If cropping is not used, embed expected in the original test array
        original = np.ma.ones((expected.shape[0], 5, 5))
        original.mask = np.ones_like(original, dtype=bool)
        original[:, :expected.shape[1], :expected.shape[2]] = expected
        expected = original

    if not decomposed or expected.shape[0] == 1:
        # this detour is necessary, otherwise the data will not agree
        data = expected.data.max(axis=0)
        mask = expected.max(axis=0).mask
        expected = np.ma.masked_array(data=data, mask=mask)

    result = extract_shape(make_testcube,
                           tmp_path / 'test_shape.shp',
                           crop=crop,
                           decomposed=decomposed)
    np.testing.assert_array_equal(result.data.data, expected.data)
    np.testing.assert_array_equal(result.data.mask, expected.mask)


@pytest.fixture
def irreg_extract_shape_cube():
    """Create a test cube on an irregular grid to test `extract_shape`."""
    data = np.arange(9, dtype=np.float32).reshape((3, 3))
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
    cube = create_irregular_grid_cube(data, lons, lats)
    return cube


@pytest.mark.parametrize('method', ['contains', 'representative'])
def test_extract_shape_irregular(irreg_extract_shape_cube, tmp_path, method):
    """Test `extract_shape` with a cube on an irregular grid."""
    # Points are (lon, lat)
    shape = Polygon([
        (0.5, 0.5),
        (0.5, 3.0),
        (1.5, 3.0),
        (1.5, 0.5),
    ])

    shapefile = tmp_path / 'shapefile.shp'
    write_shapefile(shape, shapefile)

    cube = extract_shape(irreg_extract_shape_cube, shapefile, method)

    data = np.arange(9, dtype=np.float32).reshape((3, 3))
    mask = np.array(
        [
            [True, True, True],
            [True, False, True],
            [True, False, True],
        ],
        dtype=bool,
    )
    if method == 'representative':
        mask[1, 1] = True
    np.testing.assert_array_equal(cube.data, data)
    np.testing.assert_array_equal(cube.data.mask, mask)


def test_extract_shape_wrong_method_raises():
    with pytest.raises(ValueError) as exc:
        extract_shape(iris.cube.Cube([]), 'test.shp', method='wrong')
        assert exc.value == ("Invalid value for `method`. Choose from "
                             "'contains', 'representative'.")


if __name__ == '__main__':
    unittest.main()
