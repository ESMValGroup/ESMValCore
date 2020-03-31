import os
import shutil
import tempfile
import unittest

import pytest
from iris.cube import Cube

from esmvalcore.cmor.fix import Fix


class TestFix(unittest.TestCase):
    def setUp(self):
        """Set up temp folder"""
        self.temp_folder = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temp folder"""
        shutil.rmtree(self.temp_folder)

    def test_get_fix(self):
        from esmvalcore.cmor._fixes.cmip5.canesm2 import FgCo2
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'CanESM2', 'Amon', 'fgco2'), [FgCo2(None)])

    def test_get_fix_case_insensitive(self):
        from esmvalcore.cmor._fixes.cmip5.canesm2 import FgCo2
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'CanESM2', 'Amon', 'fgCo2'), [FgCo2(None)])

    def test_get_fixes_with_replace(self):
        from esmvalcore.cmor._fixes.cmip5.bnu_esm import Ch4
        self.assertListEqual(Fix.get_fixes('CMIP5', 'BNU-ESM', 'Amon', 'ch4'),
                             [Ch4(None)])

    def test_get_fixes_with_generic(self):
        from esmvalcore.cmor._fixes.cmip5.cesm1_bgc import Gpp
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'CESM1-BGC', 'Amon', 'gpp'), [Gpp(None)])

    def test_get_fix_no_project(self):
        with pytest.raises(KeyError):
            Fix.get_fixes('BAD_PROJECT', 'BNU-ESM', 'Amon', 'ch4')

    def test_get_fix_no_model(self):
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'BAD_MODEL', 'Amon', 'ch4'), [])

    def test_get_fix_no_var(self):
        self.assertListEqual(
            Fix.get_fixes('CMIP5', 'BNU-ESM', 'Amon', 'BAD_VAR'), [])

    def test_fix_metadata(self):
        cube = Cube([0])
        reference = Cube([0])

        self.assertEqual(Fix(None).fix_metadata(cube), reference)

    def test_fix_data(self):
        cube = Cube([0])
        reference = Cube([0])

        self.assertEqual(Fix(None).fix_data(cube), reference)

    def test_fix_file(self):
        filepath = 'sample_filepath'
        self.assertEqual(Fix(None).fix_file(filepath, 'preproc'), filepath)

    def test_fixed_filenam(self):
        filepath = os.path.join(self.temp_folder, 'file.nc')
        output_dir = os.path.join(self.temp_folder, 'fixed')
        os.makedirs(output_dir)
        fixed_filepath = Fix(None).get_fixed_filepath(output_dir, filepath)
        self.assertTrue(fixed_filepath, os.path.join(output_dir, 'file.nc'))
