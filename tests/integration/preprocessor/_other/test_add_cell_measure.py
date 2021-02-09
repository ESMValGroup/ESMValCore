"""
Test add_cell_measure.

Integration tests for the
:func:`esmvalcore.preprocessor._other.add_cell_measure`
function.

"""

import iris
import numpy as np
import pytest

from esmvalcore.cmor.check import CheckLevels
from esmvalcore.preprocessor._other import add_cell_measure


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
        self.times = iris.coords.DimCoord([0, 1.5, 2.5, 3.5],
                                          standard_name='time',
                                          bounds=[[0, 1], [1, 2], [2, 3],
                                                  [3, 4]],
                                          units='hours')
        self.time2 = iris.coords.DimCoord([0, 1.5, 2.5],
                                          standard_name='time',
                                          bounds=[[0, 1], [1, 2], [2, 3]],
                                          units='hours')
        self.coords_spec = [(self.lats, 0), (self.lons, 1)]
        self.fx_area = iris.cube.Cube(fx_area_data,
                                      dim_coords_and_dims=self.coords_spec)
        self.fx_volume = iris.cube.Cube(fx_volume_data,
                                        dim_coords_and_dims=[
                                            (self.depth, 0),
                                            (self.lats, 1),
                                            (self.lons, 2)
                                            ])

    def test_add_cell_measure_area(self, tmp_path):
        """Test mask_landsea func."""
        fx_vars = {
            'areacella': {'table_id': 'fx', 'frequency': 'fx'},
            'areacello': {'table_id': 'Ofx', 'frequency': 'fx'}
            }
        for fx_var in fx_vars:
            self.fx_area.var_name = fx_var
            self.fx_area.standard_name = 'cell_area'
            self.fx_area.units = 'm2'
            self.fx_area.attributes['table_id'] = fx_vars[fx_var]['table_id']
            self.fx_area.attributes['frequency'] = fx_vars[fx_var]['frequency']
            fx_file = str(tmp_path / f'{fx_var}.nc')
            iris.save(self.fx_area, fx_file)
            cube = iris.cube.Cube(self.new_cube_data,
                                  dim_coords_and_dims=self.coords_spec)
            cube = add_cell_measure(
                cube, {fx_var: fx_file}, 'CMIP6',
                'EC-Earth3', CheckLevels.IGNORE)
            assert cube.cell_measure(self.fx_area.standard_name) is not None

    def test_add_cell_measure_volume(self, tmp_path):
        """Test mask_landsea func."""
        fx_vars = {
            'volcello': {'table_id': 'Ofx', 'frequency': 'fx'}
            }
        for fx_var in fx_vars:
            self.fx_volume.var_name = fx_var
            self.fx_volume.standard_name = 'ocean_volume'
            self.fx_volume.units = 'm3'
            self.fx_volume.attributes['table_id'] = (
                fx_vars[fx_var]['table_id'])
            self.fx_volume.attributes['frequency'] = (
                fx_vars[fx_var]['frequency'])
            fx_file = str(tmp_path / f'{fx_var}.nc')
            iris.save(self.fx_volume, fx_file)
            cube = iris.cube.Cube(self.new_cube_3D_data,
                                  dim_coords_and_dims=[
                                      (self.depth, 0),
                                      (self.lats, 1),
                                      (self.lons, 2)])
            cube = add_cell_measure(
                cube, {fx_var: fx_file}, 'CMIP6',
                'EC-Earth3', CheckLevels.IGNORE)
            assert cube.cell_measure(self.fx_volume.standard_name) is not None 

    def test_no_cell_measure(self):
        cube = iris.cube.Cube(self.new_cube_3D_data,
                              dim_coords_and_dims=[
                                  (self.depth, 0),
                                  (self.lats, 1),
                                  (self.lons, 2)])
        cube = add_cell_measure(cube, {'areacello': None}, 'CMIP6',
                                'EC-Earth3', CheckLevels.IGNORE)
        assert cube.cell_measures() == []
