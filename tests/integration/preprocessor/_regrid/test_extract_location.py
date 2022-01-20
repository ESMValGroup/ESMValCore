"""Integration tests for :func:`esmvalcore.preprocessor.extract_location."""

import iris
import iris.fileformats
import numpy as np
from iris.coords import CellMethod, DimCoord

import tests
from esmvalcore.preprocessor import extract_location


class Test(tests.Test):
    def setUp(self):
        """Prepare tests."""
        shape = (3, 45, 36)
        data = np.arange(np.prod(shape)).reshape(shape)
        self.cube = self._make_cube(data)

    @staticmethod
    def _make_cube(data):
        """Create a 3d synthetic test cube."""
        z, y, x = data.shape

        # Create the cube.
        cm = CellMethod(method='mean',
                        coords='time',
                        intervals='20 minutes',
                        comments=None)
        kwargs = dict(standard_name='air_temperature',
                      long_name='Air Temperature',
                      var_name='ta',
                      units='K',
                      attributes=dict(cube='attribute'),
                      cell_methods=(cm, ))
        cube = iris.cube.Cube(data, **kwargs)

        # Create a synthetic test latitude coordinate.
        data = np.linspace(-90, 90, y)
        cs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
        kwargs = dict(standard_name='latitude',
                      long_name='Latitude',
                      var_name='lat',
                      units='degrees_north',
                      attributes=dict(latitude='attribute'),
                      coord_system=cs)
        ycoord = DimCoord(data, **kwargs)
        ycoord.guess_bounds()
        cube.add_dim_coord(ycoord, 1)

        # Create a synthetic test longitude coordinate.
        data = np.linspace(0, 360, x)
        kwargs = dict(standard_name='longitude',
                      long_name='Longitude',
                      var_name='lon',
                      units='degrees_east',
                      attributes=dict(longitude='attribute'),
                      coord_system=cs)
        xcoord = DimCoord(data, **kwargs)
        xcoord.guess_bounds()
        cube.add_dim_coord(xcoord, 2)
        return cube

    def test_extract_only_town_name(self):
        """Test only town name."""
        point = extract_location(self.cube,
                                 scheme='nearest',
                                 location='Peñacaballera')
        self.assertEqual(point.shape, (3, ))
        np.testing.assert_equal(point.data, [1186, 2806, 4426])

    def test_extract_town_name_and_region(self):
        """Test town plus region."""
        point = extract_location(self.cube,
                                 scheme='nearest',
                                 location='Salamanca,Castilla y León')
        self.assertEqual(point.shape, (3, ))
        np.testing.assert_equal(point.data, [1186, 2806, 4426])

    def test_extract_town_and_country(self):
        """Test town plus country."""
        point = extract_location(self.cube,
                                 scheme='nearest',
                                 location='Salamanca,USA')
        self.assertEqual(point.shape, (3, ))
        np.testing.assert_equal(point.data, [1179, 2799, 4419])

    def test_extract_all_params(self):
        """Test town plus region plus country."""
        point = extract_location(self.cube,
                                 scheme='nearest',
                                 location='Salamanca,Castilla y León,Spain')
        self.assertEqual(point.shape, (3, ))
        print(point.data)
        np.testing.assert_equal(point.data, [1186, 2806, 4426])

    def test_extract_mountain(self):
        """Test town plus region plus country."""
        point = extract_location(self.cube,
                                 scheme='nearest',
                                 location='Calvitero,Candelario')
        self.assertEqual(point.shape, (3, ))
        print(point.data)
        np.testing.assert_equal(point.data, [1186, 2806, 4426])

    def test_non_existing_location(self):
        """Test town plus region plus country."""
        with self.assertRaises(ValueError):
            extract_location(self.cube,
                             scheme='nearest',
                             location='Minas Tirith,Gondor')

    def test_no_location_parameter(self):
        """Test if no location supplied."""
        with self.assertRaises(ValueError):
            extract_location(self.cube,
                             scheme='nearest',
                             location=None)

    def test_no_scheme_parameter(self):
        """Test if no scheme supplied."""
        with self.assertRaises(ValueError):
            extract_location(self.cube,
                             scheme=None,
                             location='Calvitero,Candelario')
