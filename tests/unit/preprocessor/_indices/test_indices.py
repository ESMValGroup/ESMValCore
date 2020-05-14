"""Unit test for :func:`esmvalcore.preprocessor._indices`."""
import iris
import numpy as np
from cf_units import Unit

import tests
from esmvalcore.preprocessor._indices import _extract_u850, _get_jets


class Test(tests.Test):
    """Test class for preprocessor/_multimodel.py."""

    def setUp(self):
        """Prepare tests."""
        coord_sys = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
        data = np.ma.ones((3, 3, 3, 3))
        data[1, 1, 1, 1] = 22.
        data[0, 2, 1, 1] = 33.

        time = iris.coords.DimCoord([15, 45, 75],
                                    standard_name='time',
                                    bounds=[[1., 30.],
                                            [30., 60.],
                                            [60., 90.]],
                                    units=Unit(
                                        'days since 1950-01-01',
                                        calendar='gregorian'))
        zcoord = iris.coords.DimCoord([70000., 85000., 100000.],
                                      standard_name='air_pressure',
                                      long_name='air_pressure',
                                      units='m',
                                      attributes={'positive': 'down'})
        lons = iris.coords.DimCoord([1.5, 2.5, 3.5],
                                    standard_name='longitude',
                                    long_name='longitude',
                                    bounds=[[1., 2.], [2., 3.], [3., 4.]],
                                    units='degrees_east',
                                    coord_system=coord_sys)
        lats = iris.coords.DimCoord([15., 25., 35.],
                                    standard_name='latitude',
                                    long_name='latitude',
                                    bounds=[[10., 20.],
                                            [20., 30.],
                                            [30., 40.]],
                                    units='degrees_north',
                                    coord_system=coord_sys)

        coords_spec = [(time, 0), (zcoord, 1), (lats, 2), (lons, 3)]
        self.cube = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)

    def test_extract_u850(self):
        """Test extract_u850 func."""
        self.cube.coord("air_pressure").guess_bounds()
        result = _extract_u850(self.cube)
        expected_djf = np.ma.array([[1., 1., 1.], [1., 8., 1.]])
        expected_mam = np.ma.array([1., 1., 1.])
        np.testing.assert_equal(result["DJF"].data, expected_djf)
        np.testing.assert_equal(result["MAM"].data, expected_mam)
        np.testing.assert_equal(result["JJA"], None)
        np.testing.assert_equal(result["SON"], None)

    def test_get_jets(self):
        """Test _get_jets func."""
        self.cube.coord("air_pressure").guess_bounds()
        extracted = _extract_u850(self.cube)
        jets, lats = _get_jets(extracted)
        np.testing.assert_equal(jets["DJF"].data, np.ma.array([1., 8.]))
        np.testing.assert_equal(lats["DJF"].data, np.array([25., 4.]))
