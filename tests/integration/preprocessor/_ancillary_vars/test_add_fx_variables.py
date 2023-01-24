"""Test add_fx_variables.

Integration tests for the
:func:`esmvalcore.preprocessor._ancillary_vars` module.
"""
import logging

import iris
import numpy as np
import pytest

from esmvalcore.cmor.check import CheckLevels
from esmvalcore.preprocessor._ancillary_vars import (
    _is_fx_broadcastable,
    add_ancillary_variable,
    add_cell_measure,
    add_fx_variables,
    remove_fx_variables,
)
from esmvalcore.preprocessor._time import clip_timerange

logger = logging.getLogger(__name__)

SHAPES_TO_BROADCAST = [
    ((), (1, ), True),
    ((), (10, 10), True),
    ((1, ), (10, ), True),
    ((1, ), (10, 10), True),
    ((2, ), (10, ), False),
    ((10, ), (), False),
    ((10, ), (1, ), False),
    ((10, ), (10, ), True),
    ((10, ), (10, 10), True),
    ((10, ), (7, 1), False),
    ((10, ), (10, 7), False),
    ((10, ), (7, 1, 10), True),
    ((10, ), (7, 1, 1), False),
    ((10, ), (7, 1, 7), False),
    ((10, ), (7, 10, 7), False),
    ((10, 1), (1, 1), False),
    ((10, 1), (1, 100), False),
    ((10, 1), (10, 7), True),
    ((10, 12), (10, 1), False),
    ((10, 1), (10, 12), True),
    ((10, 12), (), False),
    ((), (10, 12), True),
    ((10, 12), (1, ), False),
    ((1, ), (10, 12), True),
    ((10, 12), (12, ), False),
    ((10, 12), (1, 1), False),
    ((1, 1), (10, 12), True),
    ((10, 12), (1, 12), False),
    ((1, 12), (10, 12), True),
    ((10, 12), (10, 10, 1), False),
    ((10, 12), (10, 12, 1), False),
    ((10, 12), (10, 12, 12), False),
    ((10, 12), (10, 10, 12), True)]


@pytest.mark.parametrize('shape_1,shape_2,out', SHAPES_TO_BROADCAST)
def test_shape_is_broadcastable(shape_1, shape_2, out):
    """Test check if two shapes are broadcastable."""
    fx_cube = iris.cube.Cube(np.ones(shape_1))
    cube = iris.cube.Cube(np.ones(shape_2))
    is_broadcastable = _is_fx_broadcastable(fx_cube, cube)
    assert is_broadcastable == out


class Test:
    """Test class."""
    @pytest.fixture(autouse=True)
    def setUp(self):
        """Assemble a stock cube."""
        fx_area_data = np.ones((3, 3))
        fx_volume_data = np.ones((3, 3, 3))
        self.new_cube_data = np.empty((3, 3))
        self.new_cube_data[:] = 200.
        self.new_cube_3D_data = np.empty((3, 3, 3))
        self.new_cube_3D_data[:] = 200.
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
        self.depth = iris.coords.DimCoord([0, 1.5, 3],
                                          standard_name='depth',
                                          bounds=[[0, 1], [1, 2], [2, 3]],
                                          units='m',
                                          long_name='ocean depth coordinate')
        self.monthly_times = iris.coords.DimCoord(
            [15.5, 45, 74.5, 105, 135.5, 166,
             196.5, 227.5, 258, 288.5, 319, 349.5],
            standard_name='time',
            var_name='time',
            bounds=[[0, 31], [31, 59], [59, 90],
                    [90, 120], [120, 151], [151, 181],
                    [181, 212], [212, 243], [243, 273],
                    [273, 304], [304, 334], [334, 365]],
            units='days since 1950-01-01 00:00:00')
        self.yearly_times = iris.coords.DimCoord(
            [182.5, 547.5],
            standard_name='time',
            bounds=[[0, 365], [365, 730]],
            units='days since 1950-01-01 00:00')
        self.coords_spec = [(self.lats, 0), (self.lons, 1)]
        self.fx_area = iris.cube.Cube(fx_area_data,
                                      dim_coords_and_dims=self.coords_spec)
        self.fx_volume = iris.cube.Cube(fx_volume_data,
                                        dim_coords_and_dims=[
                                            (self.depth, 0),
                                            (self.lats, 1),
                                            (self.lons, 2)
                                            ])
        self.monthly_volume = iris.cube.Cube(np.ones((12, 3, 3, 3)),
                                             dim_coords_and_dims=[
                                             (self.monthly_times, 0),
                                             (self.depth, 1),
                                             (self.lats, 2),
                                             (self.lons, 3)
                                             ])

    def test_add_cell_measure_area(self, tmp_path):
        """Test add area fx variables as cell measures."""
        fx_vars = {
            'areacella': {
                'short_name': 'areacella',
                'project': 'CMIP6',
                'dataset': 'EC-Earth3',
                'mip': 'fx',
                'frequency': 'fx'},
            'areacello': {
                'short_name': 'areacello',
                'project': 'CMIP6',
                'dataset': 'EC-Earth3',
                'mip': 'Ofx',
                'frequency': 'fx'
            }
            }
        for fx_var in fx_vars:
            self.fx_area.var_name = fx_var
            self.fx_area.standard_name = 'cell_area'
            self.fx_area.units = 'm2'
            fx_file = str(tmp_path / f'{fx_var}.nc')
            fx_vars[fx_var].update({'filename': fx_file})
            iris.save(self.fx_area, fx_file)
            cube = iris.cube.Cube(self.new_cube_data,
                                  dim_coords_and_dims=self.coords_spec)
            cube = add_fx_variables(
                cube, {fx_var: fx_vars[fx_var]}, CheckLevels.IGNORE)
            assert cube.cell_measure(self.fx_area.standard_name) is not None

    def test_add_cell_measure_volume(self, tmp_path):
        """Test add volume as cell measure."""
        fx_vars = {
            'volcello': {
                'short_name': 'volcello',
                'project': 'CMIP6',
                'dataset': 'EC-Earth3',
                'mip': 'Ofx',
                'frequency': 'fx'}
            }
        self.fx_volume.var_name = 'volcello'
        self.fx_volume.standard_name = 'ocean_volume'
        self.fx_volume.units = 'm3'
        fx_file = str(tmp_path / 'volcello.nc')
        iris.save(self.fx_volume, fx_file)
        fx_vars['volcello'].update({'filename': fx_file})
        cube = iris.cube.Cube(self.new_cube_3D_data,
                              dim_coords_and_dims=[
                                  (self.depth, 0),
                                  (self.lats, 1),
                                  (self.lons, 2)])
        cube = add_fx_variables(cube, fx_vars, CheckLevels.IGNORE)
        assert cube.cell_measure(self.fx_volume.standard_name) is not None

    def test_clip_volume_timerange(self, tmp_path):
        """Test timerange is clipped in time dependent measures."""
        cell_measures = {
            'volcello': {
                'short_name': 'volcello',
                'project': 'CMIP6',
                'dataset': 'EC-Earth3',
                'mip': 'Omon',
                'frequency': 'mon',
                'timerange': '195001/195003'}
            }
        self.monthly_volume.var_name = 'volcello'
        self.monthly_volume.standard_name = 'ocean_volume'
        self.monthly_volume.units = 'm3'
        cell_measure_file = str(tmp_path / 'volcello.nc')
        iris.save(self.monthly_volume, cell_measure_file)
        cell_measures['volcello'].update(
            {'filename': cell_measure_file})
        cube = iris.cube.Cube(np.ones((12, 3, 3, 3)),
                              dim_coords_and_dims=[
                                  (self.monthly_times, 0),
                                  (self.depth, 1),
                                  (self.lats, 2),
                                  (self.lons, 3)])
        cube = clip_timerange(cube, '195001/195003')
        cube = add_fx_variables(cube, cell_measures, CheckLevels.IGNORE)
        cell_measure = cube.cell_measure(self.fx_volume.standard_name)
        assert cell_measure is not None
        assert cell_measure.shape == (3, 3, 3, 3)

    def test_no_cell_measure(self):
        """Test no cell measure is added."""
        cube = iris.cube.Cube(self.new_cube_3D_data,
                              dim_coords_and_dims=[
                                  (self.depth, 0),
                                  (self.lats, 1),
                                  (self.lons, 2)])
        cube = add_fx_variables(cube, {'areacello': None}, CheckLevels.IGNORE)
        assert cube.cell_measures() == []

    def test_add_ancillary_vars(self, tmp_path):
        """Test invalid variable is not added as cell measure."""
        self.fx_area.var_name = 'sftlf'
        self.fx_area.standard_name = "land_area_fraction"
        self.fx_area.units = '%'
        fx_file = str(tmp_path / f'{self.fx_area.var_name}.nc')
        iris.save(self.fx_area, fx_file)
        fx_vars = {
            'sftlf': {
                'short_name': 'sftlf',
                'project': 'CMIP6',
                'dataset': 'EC-Earth3',
                'mip': 'fx',
                'frequency': 'fx',
                'filename': fx_file}
        }
        cube = iris.cube.Cube(self.new_cube_data,
                              dim_coords_and_dims=self.coords_spec)
        cube = add_fx_variables(cube, fx_vars, CheckLevels.IGNORE)
        assert cube.ancillary_variable(self.fx_area.standard_name) is not None

    def test_wrong_shape(self, tmp_path):
        """Test fx_variable is not added if it's not broadcastable to cube."""
        volume_data = np.ones((2, 3, 3, 3))
        volume_cube = iris.cube.Cube(
            volume_data,
            dim_coords_and_dims=[(self.yearly_times, 0),
                                 (self.depth, 1),
                                 (self.lats, 2),
                                 (self.lons, 3)])
        volume_cube.standard_name = 'ocean_volume'
        volume_cube.var_name = 'volcello'
        volume_cube.units = 'm3'
        fx_file = str(tmp_path / f'{volume_cube.var_name}.nc')
        iris.save(volume_cube, fx_file)
        fx_vars = {
            'volcello': {
                'short_name': 'volcello',
                'project': 'CMIP6',
                'dataset': 'EC-Earth3',
                'mip': 'Oyr',
                'frequency': 'yr',
                'filename': fx_file,
                'timerange': '1950/1951'}
            }
        data = np.ones((12, 3, 3, 3))
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[(self.monthly_times, 0),
                                 (self.depth, 1),
                                 (self.lats, 2),
                                 (self.lons, 3)])
        cube.var_name = 'thetao'
        cube = add_fx_variables(cube, fx_vars, CheckLevels.IGNORE)
        assert cube.cell_measures() == []

    def test_remove_fx_vars(self):
        """Test fx_variables are removed from cube."""
        cube = iris.cube.Cube(self.new_cube_3D_data,
                              dim_coords_and_dims=[(self.depth, 0),
                                                   (self.lats, 1),
                                                   (self.lons, 2)])
        self.fx_area.var_name = 'areacella'
        self.fx_area.standard_name = 'cell_area'
        self.fx_area.units = 'm2'
        add_cell_measure(cube, self.fx_area, measure='area')
        assert cube.cell_measure(self.fx_area.standard_name) is not None
        self.fx_area.var_name = 'sftlf'
        self.fx_area.standard_name = "land_area_fraction"
        self.fx_area.units = '%'
        add_ancillary_variable(cube, self.fx_area)
        assert cube.ancillary_variable(self.fx_area.standard_name) is not None
        cube = remove_fx_variables(cube)
        assert cube.cell_measures() == []
        assert cube.ancillary_variables() == []
