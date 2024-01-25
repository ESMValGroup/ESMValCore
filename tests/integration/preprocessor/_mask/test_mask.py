"""Test mask.

Integration tests for the :func:`esmvalcore.preprocessor._mask` module.
"""
from pathlib import Path

import iris
import iris.fileformats
import numpy as np
import pytest

from esmvalcore.preprocessor import (
    PreprocessorFile,
    add_supplementary_variables,
    mask_fillvalues,
    mask_landsea,
    mask_landseaice,
)
from tests import assert_array_equal


class Test:
    """Test class."""

    @pytest.fixture(autouse=True)
    def setUp(self):
        """Assemble a stock cube."""
        fx_data = np.empty((3, 3))
        fx_data[:] = 60.
        fx_data[1, 2] = 30.
        self.new_cube_data = np.empty((2, 3, 3))
        self.new_cube_data[:] = 200.
        crd_sys = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
        self.lons = iris.coords.DimCoord([0, 1.5, 3],
                                         standard_name='longitude',
                                         bounds=[[0, 1], [1, 2], [2, 3]],
                                         units='degrees_east',
                                         coord_system=crd_sys)
        self.lats = iris.coords.DimCoord([0, 1.5, 3],
                                         standard_name='latitude',
                                         bounds=[[0, 1], [1, 2], [2, 3]],
                                         units='degrees_north',
                                         coord_system=crd_sys)
        self.zcoord = iris.coords.DimCoord([0.5, 5.],
                                           long_name='zcoord',
                                           bounds=[[0., 2.5], [2.5, 25.]],
                                           units='m',
                                           attributes={'positive': 'down'})
        self.times = iris.coords.DimCoord([0, 1.5, 2.5, 3.5],
                                          standard_name='time',
                                          bounds=[[0, 1], [1, 2], [2, 3],
                                                  [3, 4]],
                                          units='hours')
        self.time2 = iris.coords.DimCoord([0, 1.5, 2.5],
                                          standard_name='time',
                                          bounds=[[0, 1], [1, 2], [2, 3]],
                                          units='hours')
        self.fx_coords_spec = [(self.lats, 0), (self.lons, 1)]
        self.cube_coords_spec = [(self.zcoord, 0),
                                 (self.lats, 1), (self.lons, 2)]
        self.fx_mask = iris.cube.Cube(fx_data,
                                      dim_coords_and_dims=self.fx_coords_spec,
                                      units='%')
        self.mock_data = np.ma.empty((4, 3, 3))
        self.mock_data[:] = 10.

    def test_components_fx_var(self):
        """Test compatibility of ancillary variables."""
        self.fx_mask.var_name = 'sftlf'
        self.fx_mask.standard_name = 'land_area_fraction'
        new_cube_land = iris.cube.Cube(
            self.new_cube_data,
            dim_coords_and_dims=self.cube_coords_spec
        )
        new_cube_land = add_supplementary_variables(
            new_cube_land,
            [self.fx_mask],
        )
        result_land = mask_landsea(
            new_cube_land,
            'land',
        )
        assert isinstance(result_land, iris.cube.Cube)

        self.fx_mask.var_name = 'sftgif'
        self.fx_mask.standard_name = 'land_ice_area_fraction'
        new_cube_ice = iris.cube.Cube(
            self.new_cube_data,
            dim_coords_and_dims=self.cube_coords_spec
        )
        new_cube_ice = add_supplementary_variables(
            new_cube_ice,
            [self.fx_mask],
        )
        result_ice = mask_landseaice(
            new_cube_ice,
            'ice',
        )
        assert isinstance(result_ice, iris.cube.Cube)

    def test_mask_landsea(self):
        """Test mask_landsea func."""
        self.fx_mask.var_name = 'sftlf'
        self.fx_mask.standard_name = 'land_area_fraction'
        new_cube_land = iris.cube.Cube(
            self.new_cube_data,
            dim_coords_and_dims=self.cube_coords_spec
        )
        new_cube_land = add_supplementary_variables(
            new_cube_land,
            [self.fx_mask],
        )
        new_cube_sea = iris.cube.Cube(
            self.new_cube_data,
            dim_coords_and_dims=self.cube_coords_spec
        )
        new_cube_sea = add_supplementary_variables(
            new_cube_sea,
            [self.fx_mask],
        )

        # mask with fx files
        result_land = mask_landsea(
            new_cube_land,
            'land',
        )
        result_sea = mask_landsea(
            new_cube_sea,
            'sea',
        )
        expected = np.ma.empty((2, 3, 3))
        expected.data[:] = 200.
        expected.mask = np.ones((2, 3, 3), bool)
        expected.mask[:, 1, 2] = False
        # set fillvalues so we are sure they are equal
        np.ma.set_fill_value(result_land.data, 1e+20)
        np.ma.set_fill_value(result_sea.data, 1e+20)
        np.ma.set_fill_value(expected, 1e+20)
        assert_array_equal(result_land.data, expected)
        expected.mask = np.zeros((2, 3, 3), bool)
        expected.mask[:, 1, 2] = True
        assert_array_equal(result_sea.data, expected)

        # mask with shp files
        new_cube_land = iris.cube.Cube(
            self.new_cube_data,
            dim_coords_and_dims=self.cube_coords_spec
        )
        new_cube_sea = iris.cube.Cube(
            self.new_cube_data,
            dim_coords_and_dims=self.cube_coords_spec
        )
        result_land = mask_landsea(new_cube_land, 'land')
        result_sea = mask_landsea(new_cube_sea, 'sea')

        # bear in mind all points are in the ocean
        np.ma.set_fill_value(result_land.data, 1e+20)
        np.ma.set_fill_value(result_sea.data, 1e+20)
        expected.mask = np.zeros((3, 3), bool)
        assert_array_equal(result_land.data, expected)
        expected.mask = np.ones((3, 3), bool)
        assert_array_equal(result_sea.data, expected)

    def test_mask_landseaice(self):
        """Test mask_landseaice func."""
        self.fx_mask.var_name = 'sftgif'
        self.fx_mask.standard_name = 'land_ice_area_fraction'
        new_cube_ice = iris.cube.Cube(
            self.new_cube_data,
            dim_coords_and_dims=self.cube_coords_spec
        )
        new_cube_ice = add_supplementary_variables(
            new_cube_ice,
            [self.fx_mask],
        )
        result_ice = mask_landseaice(new_cube_ice, 'ice')
        expected = np.ma.empty((2, 3, 3))
        expected.data[:] = 200.
        expected.mask = np.ones((2, 3, 3), bool)
        expected.mask[:, 1, 2] = False
        np.ma.set_fill_value(result_ice.data, 1e+20)
        np.ma.set_fill_value(expected, 1e+20)
        assert_array_equal(result_ice.data, expected)

    def test_mask_fillvalues(self, mocker):
        """Test the fillvalues mask: func mask_fillvalues."""
        data_1 = data_2 = self.mock_data
        data_2.mask = np.ones((4, 3, 3), bool)
        coords_spec = [(self.times, 0), (self.lats, 1), (self.lons, 2)]
        cube_1 = iris.cube.Cube(data_1, dim_coords_and_dims=coords_spec)
        cube_2 = iris.cube.Cube(data_2, dim_coords_and_dims=coords_spec)
        filename_1 = 'file1.nc'
        filename_2 = 'file2.nc'
        product_1 = mocker.create_autospec(
            PreprocessorFile,
            spec_set=True,
            instance=True,
        )
        product_1.filename = filename_1
        product_1.cubes = [cube_1]
        product_2 = mocker.create_autospec(
            PreprocessorFile,
            spec_set=True,
            instance=True,
        )
        product_2.filename = filename_2
        product_2.cubes = [cube_2]
        results = mask_fillvalues({product_1, product_2},
                                  0.95,
                                  min_value=-1.e10,
                                  time_window=1)
        result_1, result_2 = None, None
        for product in results:
            if product.filename == filename_1:
                result_1 = product.cubes[0]
            if product.filename == filename_2:
                result_2 = product.cubes[0]
        assert_array_equal(result_2.data.mask, data_2.mask)
        assert_array_equal(result_1.data, data_1)

    def test_mask_fillvalues_zero_threshold(self, mocker):
        """Test the fillvalues mask: func mask_fillvalues for 0-threshold."""
        data_1 = self.mock_data
        data_2 = self.mock_data[0:3]
        data_1.mask = np.ones((4, 3, 3), bool)
        data_1.mask[0] = False
        data_1.mask[2] = False
        data_2.mask = np.ones((3, 3, 3), bool)
        data_2.mask[0] = False
        data_2.mask[1] = False
        coords_spec = [(self.times, 0), (self.lats, 1), (self.lons, 2)]
        coords_spec2 = [(self.time2, 0), (self.lats, 1), (self.lons, 2)]
        cube_1 = iris.cube.Cube(data_1, dim_coords_and_dims=coords_spec)
        cube_2 = iris.cube.Cube(data_2, dim_coords_and_dims=coords_spec2)
        filename_1 = Path('file1.nc')
        filename_2 = Path('file2.nc')
        product_1 = mocker.create_autospec(
            PreprocessorFile,
            spec_set=True,
            instance=True,
        )
        product_1.filename = filename_1
        product_1.cubes = [cube_1]
        product_2 = mocker.create_autospec(
            PreprocessorFile,
            spec_set=True,
            instance=True,
        )
        product_2.filename = filename_2
        product_2.cubes = [cube_2]
        results = mask_fillvalues({product_1, product_2}, 0., min_value=-1.e20)
        result_1, result_2 = None, None
        for product in results:
            if product.filename == filename_1:
                result_1 = product.cubes[0]
            if product.filename == filename_2:
                result_2 = product.cubes[0]
        # identical masks
        assert_array_equal(
            result_2.data[0, ...].mask,
            result_1.data[0, ...].mask,
        )
        # identical masks with cumulative
        cumulative_mask = cube_1[1:2].data.mask | cube_2[1:2].data.mask
        assert_array_equal(result_1[1:2].data.mask, cumulative_mask)
        assert_array_equal(result_2[2:3].data.mask, cumulative_mask)

    def test_mask_fillvalues_min_value_none(self, mocker):
        """Test ``mask_fillvalues`` for min_value=None."""
        # We use non-masked data here and explicitly set some values to 0 here
        # since this caused problems in the past, see
        # github.com/ESMValGroup/ESMValCore/issues/487#issuecomment-677732121
        data_1 = self.mock_data
        data_2 = np.ones((3, 3, 3))
        data_2[:, 0, :] = 0.0

        coords_spec = [(self.times, 0), (self.lats, 1), (self.lons, 2)]
        coords_spec2 = [(self.time2, 0), (self.lats, 1), (self.lons, 2)]
        cube_1 = iris.cube.Cube(data_1, dim_coords_and_dims=coords_spec)
        cube_2 = iris.cube.Cube(data_2, dim_coords_and_dims=coords_spec2)
        filename_1 = Path('file1.nc')
        filename_2 = Path('file2.nc')

        # Mock PreprocessorFile to avoid provenance errors
        product_1 = mocker.create_autospec(
            PreprocessorFile,
            spec_set=True,
            instance=True,
        )
        product_1.filename = filename_1
        product_1.cubes = [cube_1]
        product_2 = mocker.create_autospec(
            PreprocessorFile,
            spec_set=True,
            instance=True,
        )
        product_2.filename = filename_2
        product_2.cubes = [cube_2]

        results = mask_fillvalues(
            {product_1, product_2},
            threshold_fraction=1.0,
            min_value=None,
        )

        assert len(results) == 2
        for product in results:
            if product.filename in (filename_1, filename_2):
                assert len(product.cubes) == 1
                assert not np.ma.is_masked(product.cubes[0].data)
            else:
                assert False, f"Invalid filename: {product.filename}"
