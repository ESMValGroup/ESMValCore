"""Unit tests for the :func:`esmvalcore.preprocessor._area` module."""
import unittest
from pathlib import Path

import fiona
import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.cube import Cube
from numpy.testing._private.utils import assert_raises
from shapely.geometry import Polygon, mapping

import esmvalcore.preprocessor
import tests
from esmvalcore.preprocessor._area import (
    _crop_cube,
    area_statistics,
    extract_named_regions,
    extract_region,
    extract_shape,
)
from esmvalcore.preprocessor._shared import guess_bounds


class Test(tests.Test):
    """Test class for the :func:`esmvalcore.preprocessor._area_pp` module."""
    def setUp(self):
        """Prepare tests."""
        self.coord_sys = iris.coord_systems.GeogCS(
            iris.fileformats.pp.EARTH_RADIUS)
        data = np.ones((5, 5), dtype=np.float32)
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
        expected = np.array([1.], dtype=np.float32)
        self.assert_array_equal(result.data, expected)

    def test_area_statistics_cell_measure_mean(self):
        """Test for area average of a 2D field.

        The area measure is pre-loaded in the cube
        """
        cube = guess_bounds(self.grid, ['longitude', 'latitude'])
        grid_areas = iris.analysis.cartography.area_weights(cube)
        measure = iris.coords.CellMeasure(
            grid_areas,
            standard_name='cell_area',
            units='m2',
            measure='area')
        self.grid.add_cell_measure(measure, range(0, measure.ndim))
        result = area_statistics(self.grid, 'mean')
        expected = np.array([1.], dtype=np.float32)
        self.assert_array_equal(result.data, expected)

    def test_area_statistics_min(self):
        """Test for area average of a 2D field."""
        result = area_statistics(self.grid, 'min')
        expected = np.array([1.], dtype=np.float32)
        self.assert_array_equal(result.data, expected)

    def test_area_statistics_max(self):
        """Test for area average of a 2D field."""
        result = area_statistics(self.grid, 'max')
        expected = np.array([1.], dtype=np.float32)
        self.assert_array_equal(result.data, expected)

    def test_area_statistics_median(self):
        """Test for area average of a 2D field."""
        result = area_statistics(self.grid, 'median')
        expected = np.array([1.], dtype=np.float32)
        self.assert_array_equal(result.data, expected)

    def test_area_statistics_std_dev(self):
        """Test for area average of a 2D field."""
        result = area_statistics(self.grid, 'std_dev')
        expected = np.array([0.], dtype=np.float32)
        self.assert_array_equal(result.data, expected)

    def test_area_statistics_sum(self):
        """Test for sum of a 2D field."""
        result = area_statistics(self.grid, 'sum')
        grid_areas = iris.analysis.cartography.area_weights(self.grid)
        expected = np.sum(grid_areas).astype(np.float32)
        self.assert_array_equal(result.data, expected)

    def test_area_statistics_variance(self):
        """Test for area average of a 2D field."""
        result = area_statistics(self.grid, 'variance')
        expected = np.array([0.], dtype=np.float32)
        self.assert_array_equal(result.data, expected)

    def test_area_statistics_neg_lon(self):
        """Test for area average of a 2D field."""
        result = area_statistics(self.negative_grid, 'mean')
        expected = np.array([1.], dtype=np.float32)
        self.assert_array_equal(result.data, expected)

    def test_area_statistics_rms(self):
        """Test for area rms of a 2D field."""
        result = area_statistics(self.grid, 'rms')
        expected = np.array([1.], dtype=np.float32)
        self.assert_array_equal(result.data, expected)

    def test_extract_region(self):
        """Test for extracting a region from a 2D field."""
        result = extract_region(self.grid, 1.5, 2.5, 1.5, 2.5)
        # expected outcome
        expected = np.ones((2, 2))
        self.assert_array_equal(result.data, expected)

    def test_extract_region_mean(self):
        """Test for extracting a region and performing the area mean of a 2D
        field."""
        cube = guess_bounds(self.grid, ['longitude', 'latitude'])
        grid_areas = iris.analysis.cartography.area_weights(cube)
        measure = iris.coords.CellMeasure(
            grid_areas,
            standard_name='cell_area',
            units='m2',
            measure='area')
        self.grid.add_cell_measure(measure, range(0, measure.ndim))
        region = extract_region(self.grid, 1.5, 2.5, 1.5, 2.5)
        # expected outcome
        expected = np.ones((2, 2))
        self.assert_array_equal(region.data, expected)
        result = area_statistics(region, 'mean')
        expected_mean = np.array([1.])
        self.assert_array_equal(result.data, expected_mean)

    def test_extract_region_neg_lon(self):
        """Test for extracting a region with a negative longitude field."""
        result = extract_region(self.negative_grid, -0.5, 0.5, -0.5, 0.5)
        expected = np.ones((2, 2))
        self.assert_array_equal(result.data, expected)

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
        self.assert_array_equal(result1.data, expected)

        # test list of regions
        result2 = extract_named_regions(region_cube, ['region1', 'region2'])
        expected = np.ones((3, 2))
        self.assert_array_equal(result2.data, expected)

        # test for expected failures:
        with self.assertRaises(ValueError):
            extract_named_regions(region_cube, 'reg_A')
            extract_named_regions(region_cube, ['region1', 'reg_A'])


def create_irregular_grid_cube(data, lons, lats):
    """Create test cube on irregular grid."""
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


IRREGULAR_EXTRACT_REGION_TESTS = [
    {
        'region': (100, 140, -10, 90),
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
        'region': (100, 360, -60, 0),
        'mask': np.array(
            [
                [True, False],
                [False, False],
            ],
            dtype=bool,
        ),
        'data': np.arange(18, dtype=np.float32).reshape((2, 3, 3))[:, 0:2, 1:3]
    },
    {
        'region': (10, 360, 0, 90),
        'mask': np.array(
            [
                [True, False],
                [False, False],
            ],
            dtype=bool,
        ),
        'data': np.arange(18, dtype=np.float32).reshape((2, 3, 3))[:, 1:, 1:]
    },
    {
        'region': (0, 360, -90, -30),
        'mask': np.array(
            [
                [False, False, False],
            ],
            dtype=bool,
        ),
        'data': np.arange(18, dtype=np.float32).reshape((2, 3, 3))[:, :1, :]
    },
    {
        'region': (200, 10, -90, -60),
        'mask': np.array(
            [
                [False, True, False],
            ],
            dtype=bool,
        ),
        'data': np.arange(18, dtype=np.float32).reshape((2, 3, 3))[:, :1, :]
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
        'data':
        np.arange(18, dtype=np.float32).reshape((2, 3, 3))
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
    data = np.arange(18, dtype=np.float32).reshape((2, 3, 3))
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

        for i in range(2):
            np.testing.assert_array_equal(cube.data[i].mask, case['mask'])
        np.testing.assert_array_equal(cube.data.data, case['data'])
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


def create_rotated_grid_cube(data):
    """Create test cube on rotated grid."""
    # CORDEX EUR-44 example
    grid_north_pole_latitude = 39.25
    grid_north_pole_longitude = -162.0
    grid_lons = np.array(
        [-10, 0, 10],
        dtype=np.float64,
    )
    grid_lats = np.array(
        [-10, 0, 10],
        dtype=np.float64,
    )

    coord_sys_rotated = iris.coord_systems.RotatedGeogCS(
        grid_north_pole_latitude, grid_north_pole_longitude)
    grid_lat = iris.coords.DimCoord(grid_lats,
                                    var_name='rlon',
                                    standard_name='grid_latitude',
                                    units='degrees',
                                    coord_system=coord_sys_rotated)
    grid_lon = iris.coords.DimCoord(grid_lons,
                                    var_name='rlon',
                                    standard_name='grid_longitude',
                                    units='degrees',
                                    coord_system=coord_sys_rotated)

    coord_sys = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
    glon, glat = np.meshgrid(grid_lons, grid_lats)
    lons, lats = iris.analysis.cartography.unrotate_pole(
        np.deg2rad(glon), np.deg2rad(glat), grid_north_pole_longitude,
        grid_north_pole_latitude)

    lat = iris.coords.AuxCoord(lats,
                               var_name='lat',
                               standard_name='latitude',
                               units='degrees',
                               coord_system=coord_sys)
    lon = iris.coords.AuxCoord(lons,
                               var_name='lon',
                               standard_name='longitude',
                               units='degrees',
                               coord_system=coord_sys)
    dim_coord_spec = [
        (grid_lat, 0),
        (grid_lon, 1),
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


ROTATED_AREA_STATISTICS_TEST = [
    {
        'operator': 'mean',
        'data': np.ones(9, dtype=np.float32).reshape((3, 3)),
        'expected': np.array([1.]),
    },
    {
        'operator': 'median',
        'data': np.ones(9, dtype=np.float32).reshape((3, 3)),
        'expected': np.array([1.]),
    },
    {
        'operator': 'std_dev',
        'data': np.ones(9, dtype=np.float32).reshape((3, 3)),
        'expected': np.array([0.]),
    },
    {
        'operator': 'sum',
        'data': np.ones(9, dtype=np.float32).reshape((3, 3)),
    },
    {
        'operator': 'variance',
        'data': np.ones(9, dtype=np.float32).reshape((3, 3)),
        'expected': np.array([0.]),
    },
    {
        'operator': 'min',
        'data': np.arange(9, dtype=np.float32).reshape((3, 3)),
        'expected': np.array([0.]),
    },
    {
        'operator': 'max',
        'data': np.arange(9, dtype=np.float32).reshape((3, 3)),
        'expected': np.array([8.]),
    },
]


@pytest.mark.parametrize('case', ROTATED_AREA_STATISTICS_TEST)
def test_area_statistics_rotated(case):
    """Test `area_statistics` with data on an rotated grid."""
    rotated_cube = create_rotated_grid_cube(case['data'])
    operator = case['operator']
    cube = area_statistics(
        rotated_cube,
        operator,
    )
    if operator != 'sum':
        np.testing.assert_array_equal(cube.data, case['expected'])
    else:
        cube_tmp = rotated_cube.copy()
        cube_tmp.remove_coord('latitude')
        cube_tmp.coord('grid_latitude').rename('latitude')
        cube_tmp.remove_coord('longitude')
        cube_tmp.coord('grid_longitude').rename('longitude')
        grid_areas = iris.analysis.cartography.area_weights(cube_tmp)
        expected = np.sum(grid_areas).astype(np.float32)
        np.testing.assert_array_equal(cube.data, expected)


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


def write_shapefile(shape, path, negative_bounds=False):
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
            if not negative_bounds:
                file.write({
                    'geometry': mapping(s),
                    'properties': {
                        'id': id_
                    },
                })
            else:
                file.write({
                    'geometry': mapping(s),
                    'properties': {
                        'id': id_
                    },
                    'bounds': [-180, 180, -90, 90],
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
    write_shapefile(polyg,
                    tmp_path / 'test_shape_negative_bounds.shp',
                    negative_bounds=True)

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
    write_shapefile(polyg,
                    tmp_path / 'test_shape_negative_bounds.shp',
                    negative_bounds=True)

    # Make corresponding expected masked array
    (slat, slon) = np.ceil([slat, slon]).astype(int)
    vals = np.ones((nshape, min(slat + 2, 5), min(slon + 1 + nshape, 5)))
    mask = vals.copy()
    for n in range(nshape):
        mask[n, 1:1 + slat, 1 + n:1 + n + slon] = 0
    return np.ma.masked_array(vals, mask)


def _create_sample_full_cube():
    cube = Cube(np.zeros((4, 180, 360)), var_name='co2', units='J')
    cube.add_dim_coord(
        iris.coords.DimCoord(
            np.array([10., 40., 70., 110.]),
            standard_name='time',
            units=Unit('days since 1950-01-01 00:00:00', calendar='gregorian'),
        ),
        0,
    )
    cube.add_dim_coord(
        iris.coords.DimCoord(
            np.arange(-90., 90., 1.),
            standard_name='latitude',
            units='degrees',
        ),
        1,
    )
    cube.add_dim_coord(
        iris.coords.DimCoord(
            np.arange(0., 360., 1.),
            standard_name='longitude',
            units='degrees',
        ),
        2,
    )

    cube.coord("time").guess_bounds()
    cube.coord("longitude").guess_bounds()
    cube.coord("latitude").guess_bounds()

    return cube


def test_crop_cube(make_testcube, square_shape, tmp_path):
    """Test for cropping a cube by shape bounds."""
    with fiona.open(tmp_path / 'test_shape.shp') as geometries:
        result = _crop_cube(make_testcube, *geometries.bounds)
        expected = square_shape.data
        np.testing.assert_array_equal(result.data, expected)


def test_crop_cube_with_ne_file_imitation():
    """Test for cropping a cube by shape bounds."""
    cube = _create_sample_full_cube()
    bounds = [-10., -99., 370., 100.]
    result = _crop_cube(cube, *tuple(bounds))
    result = (result.coord("latitude").points[-1],
              result.coord("longitude").points[-1])
    expected = (89., 359.)
    np.testing.assert_allclose(result, expected)


@pytest.fixture
def ne_ocean_shapefile():
    """Path to natural earth 50m ocean shapefile."""
    preproc_path = Path(esmvalcore.preprocessor.__file__).parent
    shapefile = preproc_path / "ne_masks" / "ne_50m_ocean.shp"
    return str(shapefile)


def test_crop_cube_with_ne_file(ne_ocean_shapefile):
    """Test for cropping a cube by shape bounds."""
    with fiona.open(ne_ocean_shapefile) as geometries:
        cube = _create_sample_full_cube()
        result = _crop_cube(cube, *geometries.bounds, cmor_coords=False)
        result = (result.coord("latitude").points[-1],
                  result.coord("longitude").points[-1])
        expected = (89., 359.)
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize('crop', [True, False])
@pytest.mark.parametrize('ids', [None, [
    0,
]])
def test_extract_shape(make_testcube, square_shape, tmp_path, crop, ids):
    """Test for extracting a region with shapefile."""
    expected = square_shape
    if not crop:
        # If cropping is not used, embed expected in the original test array
        original = np.ma.ones((5, 5))
        original.mask = np.ones_like(original, dtype=bool)
        original[:expected.shape[0], :expected.shape[1]] = expected
        expected = original
    result = extract_shape(make_testcube,
                           tmp_path / 'test_shape.shp',
                           crop=crop,
                           ids=ids)
    np.testing.assert_array_equal(result.data.data, expected.data)
    np.testing.assert_array_equal(result.data.mask, expected.mask)


def test_extract_shape_natural_earth(make_testcube, ne_ocean_shapefile):
    """Test for extracting a shape from NE file."""
    expected = np.ones((5, 5))
    preproc_path = Path(esmvalcore.preprocessor.__file__).parent
    shp_file = preproc_path / "ne_masks" / "ne_50m_ocean.shp"
    result = extract_shape(
        make_testcube,
        shp_file,
        crop=False,
    )
    np.testing.assert_array_equal(result.data.data, expected)


def test_extract_shape_fx(make_testcube, ne_ocean_shapefile):
    """Test for extracting a shape from NE file."""
    expected = np.ones((5, 5))
    preproc_path = Path(esmvalcore.preprocessor.__file__).parent
    shp_file = preproc_path / "ne_masks" / "ne_50m_ocean.shp"
    cube = make_testcube
    measure = iris.coords.CellMeasure(cube.data,
                                      standard_name='cell_area',
                                      var_name='areacello',
                                      units='m2',
                                      measure='area')
    ancillary_var = iris.coords.AncillaryVariable(
        cube.data,
        standard_name='land_ice_area_fraction',
        var_name='sftgif',
        units='%')
    cube.add_cell_measure(measure, (0, 1))
    cube.add_ancillary_variable(ancillary_var, (0, 1))
    result = extract_shape(
        cube,
        shp_file,
        crop=False,
    )
    np.testing.assert_array_equal(result.data.data, expected)

    assert result.cell_measures()
    result_measure = result.cell_measure('cell_area').data
    np.testing.assert_array_equal(measure.data, result_measure)

    assert result.ancillary_variables()
    result_ancillary_var = result.ancillary_variable(
        'land_ice_area_fraction').data
    np.testing.assert_array_equal(ancillary_var.data, result_ancillary_var)


def test_extract_shape_ne_check_nans(ne_ocean_shapefile):
    """Test shape from NE file with check for boundary NaN's."""
    cube = _create_sample_full_cube()
    result = extract_shape(cube, ne_ocean_shapefile, crop=False)
    assert not result[:, 90, 180].data.mask.all()


@pytest.mark.parametrize('crop', [True, False])
def test_extract_shape_negative_bounds(make_testcube, square_shape, tmp_path,
                                       crop):
    """Test for extr a reg with shapefile w/neg ie bound ie (-180, 180)."""
    expected = square_shape
    if not crop:
        # If cropping is not used, embed expected in the original test array
        original = np.ma.ones((5, 5))
        original.mask = np.ones_like(original, dtype=bool)
        original[:expected.shape[0], :expected.shape[1]] = expected
        expected = original
    negative_bounds_shapefile = tmp_path / 'test_shape_negative_bounds.shp'
    result = extract_shape(make_testcube, negative_bounds_shapefile, crop=crop)
    np.testing.assert_array_equal(result.data.data, expected.data)
    np.testing.assert_array_equal(result.data.mask, expected.mask)


def test_extract_shape_neg_lon(make_testcube, tmp_path, crop=False):
    """Test for extr a reg with shapefile w/negative lon."""
    (slat, slon) = (2, -2)
    polyg = Polygon([
        (1.0, 1.0 + slat),
        (1.0, 1.0),
        (1.0 + slon, 1.0),
        (1.0 + slon, 1.0 + slat),
    ])
    write_shapefile(polyg,
                    tmp_path / 'test_shape_negative_lon.shp',
                    negative_bounds=True)

    expected_data = np.ones((5, 5))
    expected_mask = np.ones((5, 5))
    expected_mask[1, 0] = False
    expected_mask[2, 0] = False
    expected = np.ma.array(expected_data, mask=expected_mask)
    negative_bounds_shapefile = tmp_path / 'test_shape_negative_lon.shp'
    result = extract_shape(make_testcube, negative_bounds_shapefile, crop=crop)
    np.testing.assert_array_equal(result.data.data, expected.data)
    np.testing.assert_array_equal(result.data.mask, expected.mask)


@pytest.mark.parametrize('crop', [True, False])
@pytest.mark.parametrize('decomposed', [True, False])
def test_extract_composite_shape(make_testcube, square_composite_shape,
                                 tmp_path, crop, decomposed):
    """Test for extracting a region with shapefile."""
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


@pytest.mark.parametrize('ids', [[0], [1], [2], [1, 2]])
def test_extract_specific_shape(make_testcube, tmp_path, ids):
    """Test for extracting a region with shapefile."""
    slat = 2.
    slon = 2.
    nshape = 3
    polyg = []
    for n in range(nshape):
        polyg.append(
            Polygon([
                (1.0 + n, 1.0 + slat),
                (1.0 + n, 1.0),
                (1.0 + n + slon, 1.0),
                (1.0 + n + slon, 1.0 + slat),
            ])
        )
    write_shapefile(polyg, tmp_path / 'test_shape.shp')

    result = extract_shape(make_testcube,
                           tmp_path / 'test_shape.shp',
                           crop=True,
                           decomposed=False,
                           ids=ids)

    expected_bounds = np.vstack([polyg[i].bounds for i in ids])

    lon_min = expected_bounds[:, 0]
    lat_min = expected_bounds[:, 1]
    lon_max = expected_bounds[:, 2]
    lat_max = expected_bounds[:, 3]

    # results from `extract_shape` are padded with masked values
    lats = result.coord('latitude')[1:-1]
    lons = result.coord('longitude')[1:-1]

    assert np.all((lats.points >= lat_min) & (lats.points <= lat_max))
    assert np.all((lons.points >= lon_min) & (lons.points <= lon_max))


def test_extract_specific_shape_raises_if_not_present(make_testcube, tmp_path):
    """Test for extracting a region with shapefile."""
    slat = 2.
    slon = 2.
    nshape = 3
    polyg = []
    for n in range(nshape):
        polyg.append(
            Polygon([(1.0 + n, 1.0 + slat), (1.0 + n, 1.0),
                     (1.0 + n + slon, 1.0), (1.0 + n + slon, 1.0 + slat)]))
    write_shapefile(polyg, tmp_path / 'test_shape.shp')

    with assert_raises(ValueError):
        extract_shape(make_testcube,
                      tmp_path / 'test_shape.shp',
                      crop=True,
                      decomposed=False,
                      ids=[1, 2, 3])


@pytest.mark.parametrize('crop', [True, False])
@pytest.mark.parametrize('decomposed', [True, False])
def test_extract_composite_shape_negative_bounds(make_testcube,
                                                 square_composite_shape,
                                                 tmp_path, crop, decomposed):
    """Test for extr a reg with shapefile w/neg bounds ie (-180, 180)."""
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

    negative_bounds_shapefile = tmp_path / 'test_shape_negative_bounds.shp'
    result = extract_shape(make_testcube,
                           negative_bounds_shapefile,
                           crop=crop,
                           decomposed=decomposed)
    np.testing.assert_array_equal(result.data.data, expected.data)
    np.testing.assert_array_equal(result.data.mask, expected.mask)


@pytest.fixture
def irreg_extract_shape_cube():
    """Create a test cube on an irregular grid to test `extract_shape`."""
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

    data = np.arange(18, dtype=np.float32).reshape((2, 3, 3))
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
    for i in range(2):
        np.testing.assert_array_equal(cube.data[i].mask, mask)


def test_extract_shape_wrong_method_raises():
    with pytest.raises(ValueError) as exc:
        extract_shape(iris.cube.Cube([]), 'test.shp', method='wrong')
        assert exc.value == ("Invalid value for `method`. Choose from "
                             "'contains', 'representative'.")


if __name__ == '__main__':
    unittest.main()
