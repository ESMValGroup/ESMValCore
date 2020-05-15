"""Unit tests for the variable_info module."""

from unittest import TestCase
from unittest.mock import Mock, patch

from esmvalcore.cmor.check import CheckLevels
from esmvalcore.cmor.fix import Fix, fix_data, fix_file, fix_metadata


class TestFixFile(TestCase):
    """Fix file tests."""
    def setUp(self):
        """Prepare for testing."""
        self.filename = 'filename'
        self.mock_fix = Mock()
        self.mock_fix.fix_file.return_value = 'new_filename'

    def test_fix(self):
        """Check that the returned fix is applied."""
        with patch('esmvalcore.cmor._fixes.fix.Fix.get_fixes',
                   return_value=[self.mock_fix]):
            file_returned = fix_file(
                file='filename',
                short_name='short_name',
                project='project',
                dataset='model',
                mip='mip',
                output_dir='output_dir',
            )
            self.assertNotEqual(file_returned, self.filename)
            self.assertEqual(file_returned, 'new_filename')

    def test_nofix(self):
        """Check that the same file is returned if no fix is available."""
        with patch('esmvalcore.cmor._fixes.fix.Fix.get_fixes',
                   return_value=[]):
            file_returned = fix_file(
                file='filename',
                short_name='short_name',
                project='project',
                dataset='model',
                mip='mip',
                output_dir='output_dir',
            )
            self.assertEqual(file_returned, self.filename)


class TestGetCube(TestCase):
    """Test get cube by var_name method."""
    def setUp(self):
        """Prepare for testing."""
        self.cube_1 = Mock()
        self.cube_1.var_name = 'cube1'
        self.cube_2 = Mock()
        self.cube_2.var_name = 'cube2'
        self.cubes = [self.cube_1, self.cube_2]
        vardef = Mock()
        vardef.short_name = 'fix'
        self.fix = Fix(vardef)

    def test_get_first_cube(self):
        """Test selecting first cube."""
        self.assertIs(self.cube_1,
                      self.fix.get_cube_from_list(self.cubes, "cube1"))

    def test_get_second_cube(self):
        """Test selecting second cube."""
        self.assertIs(self.cube_2,
                      self.fix.get_cube_from_list(self.cubes, "cube2"))

    def test_get_default_raises(self):
        """Check that the default raises (Fix is not a cube)."""
        with self.assertRaises(Exception):
            self.fix.get_cube_from_list(self.cubes)

    def test_get_default(self):
        """Check that the default return the cube (fix is a cube)."""
        self.cube_1.var_name = 'fix'
        self.assertIs(self.cube_1, self.fix.get_cube_from_list(self.cubes))


class TestFixMetadata(TestCase):
    """Fix metadata tests."""
    def setUp(self):
        """Prepare for testing."""

        self.cube = self._create_mock_cube()
        self.intermediate_cube = self._create_mock_cube()
        self.fixed_cube = self._create_mock_cube()
        self.mock_fix = Mock()
        self.mock_fix.fix_metadata.return_value = [self.intermediate_cube]
        self.checker = Mock()
        self.check_metadata = self.checker.return_value.check_metadata

    @staticmethod
    def _create_mock_cube(var_name='short_name'):
        cube = Mock()
        cube.var_name = var_name
        cube.attributes = {'source_file': 'source_file'}
        return cube

    def test_fix(self):
        """Check that the returned fix is applied."""
        self.check_metadata.side_effect = lambda: self.fixed_cube
        with patch('esmvalcore.cmor._fixes.fix.Fix.get_fixes',
                   return_value=[self.mock_fix]):
            with patch('esmvalcore.cmor.fix._get_cmor_checker',
                       return_value=self.checker):
                cube_returned = fix_metadata(
                    cubes=[self.cube],
                    short_name='short_name',
                    project='project',
                    dataset='model',
                    mip='mip',
                )[0]
                self.checker.assert_called_once_with(self.intermediate_cube)
                self.check_metadata.assert_called_once_with()
                assert cube_returned is not self.cube
                assert cube_returned is not self.intermediate_cube
                assert cube_returned is self.fixed_cube

    def test_nofix(self):
        """Check that the same cube is returned if no fix is available."""
        self.check_metadata.side_effect = lambda: self.cube
        with patch('esmvalcore.cmor._fixes.fix.Fix.get_fixes',
                   return_value=[]):
            with patch('esmvalcore.cmor.fix._get_cmor_checker',
                       return_value=self.checker):
                cube_returned = fix_metadata(
                    cubes=[self.cube],
                    short_name='short_name',
                    project='project',
                    dataset='model',
                    mip='mip',
                )[0]
                self.checker.assert_called_once_with(self.cube)
                self.check_metadata.assert_called_once_with()
                assert cube_returned is self.cube
                assert cube_returned is not self.intermediate_cube
                assert cube_returned is not self.fixed_cube

    def test_select_var(self):
        """Check that the same cube is returned if no fix is available."""
        self.check_metadata.side_effect = lambda: self.cube
        with patch('esmvalcore.cmor._fixes.fix.Fix.get_fixes',
                   return_value=[]):
            with patch('esmvalcore.cmor.fix._get_cmor_checker',
                       return_value=self.checker):
                cube_returned = fix_metadata(
                    cubes=[self.cube,
                           self._create_mock_cube('extra')],
                    short_name='short_name',
                    project='CMIP6',
                    dataset='model',
                    mip='mip',
                )[0]
                self.checker.assert_called_once_with(self.cube)
                self.check_metadata.assert_called_once_with()
                assert cube_returned is self.cube

    def test_select_var_failed_if_bad_var_name(self):
        """Check that the same cube is returned if no fix is available."""
        with patch('esmvalcore.cmor._fixes.fix.Fix.get_fixes',
                   return_value=[]):
            with self.assertRaises(ValueError):
                fix_metadata(
                    cubes=[
                        self._create_mock_cube('not_me'),
                        self._create_mock_cube('me_neither')
                    ],
                    short_name='short_name',
                    project='CMIP6',
                    dataset='model',
                    mip='mip',
                )

    def test_cmor_checker_called(self):
        """Check that the cmor check is done."""
        checker = Mock()
        checker.return_value = Mock()
        with patch('esmvalcore.cmor._fixes.fix.Fix.get_fixes',
                   return_value=[]):
            with patch('esmvalcore.cmor.fix._get_cmor_checker',
                       return_value=checker) as get_mock:
                fix_metadata(
                    cubes=[self.cube],
                    short_name='short_name',
                    project='CMIP6',
                    dataset='dataset',
                    mip='mip',
                    frequency='frequency',
                )
                get_mock.assert_called_once_with(
                    automatic_fixes=True,
                    fail_on_error=False,
                    frequency='frequency',
                    mip='mip',
                    short_name='short_name',
                    table='CMIP6',
                    check_level=CheckLevels.DEFAULT,)
                checker.assert_called_once_with(self.cube)
                checker.return_value.check_metadata.assert_called_once_with()


class TestFixData(TestCase):
    """Fix data tests."""
    def setUp(self):
        """Prepare for testing."""
        self.cube = Mock()
        self.intermediate_cube = Mock()
        self.fixed_cube = Mock()
        self.mock_fix = Mock()
        self.mock_fix.fix_data.return_value = self.intermediate_cube
        self.checker = Mock()
        self.check_data = self.checker.return_value.check_data

    def test_fix(self):
        """Check that the returned fix is applied."""
        self.check_data.side_effect = lambda: self.fixed_cube
        with patch('esmvalcore.cmor._fixes.fix.Fix.get_fixes',
                   return_value=[self.mock_fix]):
            with patch('esmvalcore.cmor.fix._get_cmor_checker',
                       return_value=self.checker):
                cube_returned = fix_data(
                    self.cube,
                    short_name='short_name',
                    project='project',
                    dataset='model',
                    mip='mip',
                )
                self.checker.assert_called_once_with(self.intermediate_cube)
                self.check_data.assert_called_once_with()
                assert cube_returned is not self.cube
                assert cube_returned is not self.intermediate_cube
                assert cube_returned is self.fixed_cube

    def test_nofix(self):
        """Check that the same cube is returned if no fix is available."""
        self.check_data.side_effect = lambda: self.cube
        with patch('esmvalcore.cmor._fixes.fix.Fix.get_fixes',
                   return_value=[]):
            with patch('esmvalcore.cmor.fix._get_cmor_checker',
                       return_value=self.checker):
                cube_returned = fix_data(
                    self.cube,
                    short_name='short_name',
                    project='CMIP6',
                    dataset='model',
                    mip='mip',
                )
                self.checker.assert_called_once_with(self.cube)
                self.check_data.assert_called_once_with()
                assert cube_returned is self.cube
                assert cube_returned is not self.intermediate_cube
                assert cube_returned is not self.fixed_cube

    def test_cmor_checker_called(self):
        """Check that the cmor check is done."""
        checker = Mock()
        checker.return_value = Mock()
        with patch('esmvalcore.cmor._fixes.fix.Fix.get_fixes',
                   return_value=[]):
            with patch('esmvalcore.cmor.fix._get_cmor_checker',
                       return_value=checker) as get_mock:
                fix_data(self.cube, 'short_name', 'CMIP6', 'model', 'mip',
                         'frequency')
                get_mock.assert_called_once_with(
                    table='CMIP6',
                    automatic_fixes=True,
                    check_level=CheckLevels.DEFAULT,
                    fail_on_error=False,
                    frequency='frequency',
                    mip='mip',
                    short_name='short_name',
                )
                checker.assert_called_once_with(self.cube)
                checker.return_value.check_data.assert_called_once_with()
