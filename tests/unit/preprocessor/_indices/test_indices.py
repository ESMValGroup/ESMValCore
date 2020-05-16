"""Unit test for :func:`esmvalcore.preprocessor._indices`."""
import os
import tempfile

import iris
import numpy as np
import pytest
from cf_units import Unit

import tests
from esmvalcore.preprocessor._indices import (acsis_indices,
                                              _add_attribute,
                                              _djf_greenland_iceland,
                                              _extract_u850, _get_jets,
                                              _load_cube, _moc_vn)


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
        self.cube = iris.cube.Cube(data, var_name='sample',
                                   dim_coords_and_dims=coords_spec)

    def _create_sample_full_cube(self):
        """Create full map cube to select desired regions."""
        cube = iris.cube.Cube(np.zeros((4, 180, 360)),
                              var_name='co2', units='J')
        cube.data[1] = 33.
        cube.add_dim_coord(
            iris.coords.DimCoord(
                np.array([10., 40., 70., 110.]),
                standard_name='time',
                units=Unit('days since 1950-01-01 00:00:00',
                           calendar='gregorian'),
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

    def _create_sample_full_3d_cube(self):
        """Create a cube with z-lan-lat coords."""
        cube = iris.cube.Cube(np.zeros((4, 3, 180, 360)),
                              var_name='co2', units='J')
        base_cube = self._create_sample_full_cube()
        cube.add_dim_coord(base_cube.coord("time"), 0)
        cube.add_dim_coord(base_cube.coord("latitude"), 2)
        cube.add_dim_coord(base_cube.coord("longitude"), 3)
        cube.add_dim_coord(iris.coords.DimCoord([70000., 85000., 100000.],
                                      standard_name='air_pressure',
                                      long_name='air_pressure',
                                      units='m',
                                      attributes={'positive': 'down'}), 1)
        return cube

    def _save_cube(self, cube):
        descriptor, temp_file = tempfile.mkstemp('.nc')
        os.close(descriptor)
        iris.save(cube, temp_file)
        return temp_file

    def test_load_cube(self):
        """Test _load_cube func."""
        cube = self.cube
        temp_file = self._save_cube(cube)
        result_cube = _load_cube(temp_file, 'sample')
        expected_cube = cube
        np.testing.assert_equal(result_cube.data, expected_cube.data)
        np.testing.assert_equal(result_cube.var_name, expected_cube.var_name)

    def test_load_cube_fail(self):
        """Test fail of _load_cube func."""
        cube = self.cube
        temp_file = self._save_cube(cube)
        msg = "No variable cow found in file {}".format(temp_file)
        with pytest.raises(ValueError) as err_exp:
            _load_cube(temp_file, 'cow')
        assert str(err_exp.value) == msg

    def test_add_attribute(self):
        """Test _add_attribute func."""
        cube = self.cube
        result = _add_attribute(cube, [22, 22], 'cow')
        assert 'cow' in cube.attributes
        assert cube.attributes['cow'] == [22, 22]

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
        np.testing.assert_equal(jets["DJF"], np.ma.array([1., 8.]))
        np.testing.assert_equal(lats["DJF"], np.array([15., 25.]))

    def test_djf_greenland_iceland(self):
        """Test _djf_greenland_iceland func."""
        cube = self._create_sample_full_cube()
        temp_file = self._save_cube(cube)
        greenland, iceland, gre_ice = _djf_greenland_iceland(temp_file,
                                                             "co2",
                                                             "DJF")
        np.testing.assert_equal(greenland.data, np.array([0., 33.]))
        np.testing.assert_equal(iceland.data, np.array([0., 33.]))
        np.testing.assert_equal(gre_ice.data, np.array([0., 0.]))

    def test_moc_vn(self):
        """Test _moc_vn func."""
        moc_cube = self._create_sample_full_cube()
        moc_cube.var_name = "moc"
        vn_cube = self._create_sample_full_cube()
        vn_cube.var_name = "vn"
        moc_file = self._save_cube(moc_cube)
        vn_file = self._save_cube(vn_cube)
        (annual_moc,
         annual_vn,
         greenland_djf,
         iceland_djf,
         season_geo_diff) = _moc_vn(moc_file, vn_file, "moc", "vn")
        np.testing.assert_equal(annual_moc.data[0], 8.25)
        np.testing.assert_equal(annual_moc.data.shape, (1, 180, 360))
        np.testing.assert_equal(annual_vn.data[0], 8.25)
        np.testing.assert_equal(annual_vn.data.shape, (1, 180, 360))
        np.testing.assert_equal(greenland_djf.data, np.array([0., 33.]))
        np.testing.assert_equal(iceland_djf.data, np.array([0., 33.]))
        np.testing.assert_equal(season_geo_diff.data, np.ma.array([0., 0.]))

    def test_acsis_indices(self):
        """Test acsis mainfunc."""
        cube = self._create_sample_full_3d_cube()
        moc_cube = self._create_sample_full_3d_cube()
        moc_cube.var_name = "moc"
        vn_cube = self._create_sample_full_3d_cube()
        vn_cube.var_name = "vn"
        moc_file = self._save_cube(moc_cube)
        vn_file = self._save_cube(vn_cube)
        result = acsis_indices(cube, moc_file, vn_file, "moc", "vn")
        assert 'max_annual_moc' in cube.attributes
        assert 'max_annual_vn' in cube.attributes
