"""
Integration tests for the :func:
`esmvalcore.preprocessor.regrid.get_cmor_levels`
function.

"""

import os
import tempfile
import unittest

import iris
import iris.coords
import iris.cube
import numpy as np

from esmvalcore.preprocessor import _regrid


class TestGetFileLevels(unittest.TestCase):
    def setUp(self):
        """Prepare the sample file for the test"""
        self.cube = iris.cube.Cube(np.ones([2, 2, 2]), var_name='var')
        self.cube.add_dim_coord(
            iris.coords.DimCoord(np.arange(0, 2), var_name='coord'), 0)

        self.cube.coord('coord').attributes['positive'] = 'up'
        iris.util.guess_coord_axis(self.cube.coord('coord'))
        descriptor, self.path = tempfile.mkstemp('.nc')
        os.close(descriptor)
        print(self.cube)
        iris.save(self.cube, self.path)

    def tearDown(self):
        """Remove the sample file for the test"""
        os.remove(self.path)

    def test_get_coord(self):
        fix_file = unittest.mock.create_autospec(_regrid.fix_file)
        fix_file.side_effect = lambda file, **_: file
        fix_metadata = unittest.mock.create_autospec(_regrid.fix_metadata)
        fix_metadata.side_effect = lambda cubes, **_: cubes
        with unittest.mock.patch('esmvalcore.preprocessor._regrid.fix_file',
                                 fix_file):
            with unittest.mock.patch(
                    'esmvalcore.preprocessor._regrid.fix_metadata',
                    fix_metadata):
                reference_levels = _regrid.get_reference_levels(
                    filename=self.path,
                    project='CMIP6',
                    dataset='dataset',
                    short_name='short_name',
                    mip='mip',
                    frequency='mon',
                    fix_dir='output_dir',
                )
        self.assertListEqual(reference_levels, [0., 1])
