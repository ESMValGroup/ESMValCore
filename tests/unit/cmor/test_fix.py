"""Unit tests for :mod:`esmvalcore.cmor.fix`."""

from pathlib import Path
from unittest.mock import Mock, patch, sentinel

import pytest

from esmvalcore.cmor.fix import Fix, fix_data, fix_file, fix_metadata


class TestFixFile():
    """Fix file tests."""

    @pytest.fixture(autouse=True)
    def setUp(self):
        """Prepare for testing."""
        self.filename = 'filename'
        self.mock_fix = Mock()
        self.mock_fix.fix_file.return_value = 'new_filename'
        self.expected_get_fixes_call = {
            'project': 'project',
            'dataset': 'model',
            'mip': 'mip',
            'short_name': 'short_name',
            'extra_facets': {
                'project': 'project',
                'dataset': 'model',
                'mip': 'mip',
                'short_name': 'short_name',
                'frequency': 'frequency',
            },
            'session': sentinel.session,
            'frequency': 'frequency',
        }

    def test_fix(self):
        """Check that the returned fix is applied."""
        with patch('esmvalcore.cmor._fixes.fix.Fix.get_fixes',
                   return_value=[self.mock_fix]) as mock_get_fixes:
            file_returned = fix_file(
                file='filename',
                short_name='short_name',
                project='project',
                dataset='model',
                mip='mip',
                output_dir=Path('output_dir'),
                session=sentinel.session,
                frequency='frequency',
            )
            assert file_returned != self.filename
            assert file_returned == 'new_filename'
            mock_get_fixes.assert_called_once_with(
                **self.expected_get_fixes_call
            )

    def test_nofix(self):
        """Check that the same file is returned if no fix is available."""
        with patch('esmvalcore.cmor._fixes.fix.Fix.get_fixes',
                   return_value=[]) as mock_get_fixes:
            file_returned = fix_file(
                file='filename',
                short_name='short_name',
                project='project',
                dataset='model',
                mip='mip',
                output_dir=Path('output_dir'),
                session=sentinel.session,
                frequency='frequency',
            )
            assert file_returned == self.filename
            mock_get_fixes.assert_called_once_with(
                **self.expected_get_fixes_call
            )


class TestGetCube():
    """Test get cube by var_name method."""

    @pytest.fixture(autouse=True)
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
        assert self.cube_1 is self.fix.get_cube_from_list(self.cubes, "cube1")

    def test_get_second_cube(self):
        """Test selecting second cube."""
        assert self.cube_2 is self.fix.get_cube_from_list(self.cubes, "cube2")

    def test_get_default_raises(self):
        """Check that the default raises (Fix is not a cube)."""
        with pytest.raises(Exception):
            self.fix.get_cube_from_list(self.cubes)

    def test_get_default(self):
        """Check that the default return the cube (fix is a cube)."""
        self.cube_1.var_name = 'fix'
        assert self.cube_1 is self.fix.get_cube_from_list(self.cubes)


class TestFixMetadata():
    """Fix metadata tests."""

    @pytest.fixture(autouse=True)
    def setUp(self):
        """Prepare for testing."""
        self.cube = self._create_mock_cube()
        self.intermediate_cube = self._create_mock_cube()
        self.fixed_cube = self._create_mock_cube()
        self.mock_fix = Mock()
        self.mock_fix.fix_metadata.return_value = [self.intermediate_cube]
        self.checker = Mock()
        self.check_metadata = self.checker.return_value.check_metadata
        self.expected_get_fixes_call = {
            'project': 'project',
            'dataset': 'model',
            'mip': 'mip',
            'short_name': 'short_name',
            'extra_facets': {
                'project': 'project',
                'dataset': 'model',
                'mip': 'mip',
                'short_name': 'short_name',
                'frequency': 'frequency',
            },
            'session': sentinel.session,
            'frequency': 'frequency',
        }

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
                   return_value=[self.mock_fix]) as mock_get_fixes:
            with patch('esmvalcore.cmor.fix._get_cmor_checker',
                       return_value=self.checker):
                cube_returned = fix_metadata(
                    cubes=[self.cube],
                    short_name='short_name',
                    project='project',
                    dataset='model',
                    mip='mip',
                    frequency='frequency',
                    session=sentinel.session,
                )[0]
                self.checker.assert_called_once_with(
                    self.intermediate_cube
                )
                self.check_metadata.assert_called_once_with()
                assert cube_returned is not self.cube
                assert cube_returned is not self.intermediate_cube
                assert cube_returned is self.fixed_cube
                mock_get_fixes.assert_called_once_with(
                    **self.expected_get_fixes_call
                )

    def test_nofix(self):
        """Check that the same cube is returned if no fix is available."""
        self.check_metadata.side_effect = lambda: self.cube
        with patch('esmvalcore.cmor._fixes.fix.Fix.get_fixes',
                   return_value=[]) as mock_get_fixes:
            with patch('esmvalcore.cmor.fix._get_cmor_checker',
                       return_value=self.checker):
                cube_returned = fix_metadata(
                    cubes=[self.cube],
                    short_name='short_name',
                    project='project',
                    dataset='model',
                    mip='mip',
                    frequency='frequency',
                    session=sentinel.session,
                )[0]
                self.checker.assert_called_once_with(self.cube)
                self.check_metadata.assert_called_once_with()
                assert cube_returned is self.cube
                assert cube_returned is not self.intermediate_cube
                assert cube_returned is not self.fixed_cube
                mock_get_fixes.assert_called_once_with(
                    **self.expected_get_fixes_call
                )

    def test_select_var(self):
        """Check that the same cube is returned if no fix is available."""
        self.check_metadata.side_effect = lambda: self.cube
        with patch('esmvalcore.cmor._fixes.fix.Fix.get_fixes',
                   return_value=[]):
            with patch('esmvalcore.cmor.fix._get_cmor_checker',
                       return_value=self.checker):
                cube_returned = fix_metadata(
                    cubes=[self.cube, self._create_mock_cube('extra')],
                    short_name='short_name',
                    project='CMIP6',
                    dataset='model',
                    mip='mip',
                )[0]
                self.checker.assert_called_once_with(self.cube)
                self.check_metadata.assert_called_once_with()
                assert cube_returned is self.cube

    def test_select_var_failed_if_bad_var_name(self):
        """Check that an error is raised if short_names do not match."""
        msg = "More than one cube found for variable tas in CMIP6:model"
        with pytest.raises(ValueError, match=msg):
            fix_metadata(
                cubes=[
                    self._create_mock_cube('not_me'),
                    self._create_mock_cube('me_neither')
                ],
                short_name='tas',
                project='CMIP6',
                dataset='model',
                mip='Amon',
            )


class TestFixData():
    """Fix data tests."""

    @pytest.fixture(autouse=True)
    def setUp(self):
        """Prepare for testing."""
        self.cube = Mock()
        self.intermediate_cube = Mock()
        self.fixed_cube = Mock()
        self.mock_fix = Mock()
        self.mock_fix.fix_data.return_value = self.intermediate_cube
        self.checker = Mock()
        self.check_data = self.checker.return_value.check_data
        self.expected_get_fixes_call = {
            'project': 'project',
            'dataset': 'model',
            'mip': 'mip',
            'short_name': 'short_name',
            'extra_facets': {
                'project': 'project',
                'dataset': 'model',
                'mip': 'mip',
                'short_name': 'short_name',
                'frequency': 'frequency',
            },
            'session': sentinel.session,
            'frequency': 'frequency',
        }

    def test_fix(self):
        """Check that the returned fix is applied."""
        self.check_data.side_effect = lambda: self.fixed_cube
        with patch('esmvalcore.cmor._fixes.fix.Fix.get_fixes',
                   return_value=[self.mock_fix]) as mock_get_fixes:
            with patch('esmvalcore.cmor.fix._get_cmor_checker',
                       return_value=self.checker):
                cube_returned = fix_data(
                    self.cube,
                    short_name='short_name',
                    project='project',
                    dataset='model',
                    mip='mip',
                    frequency='frequency',
                    session=sentinel.session,
                )
                self.checker.assert_called_once_with(
                    self.intermediate_cube
                )
                self.check_data.assert_called_once_with()
                assert cube_returned is not self.cube
                assert cube_returned is not self.intermediate_cube
                assert cube_returned is self.fixed_cube
                mock_get_fixes.assert_called_once_with(
                    **self.expected_get_fixes_call
                )

    def test_nofix(self):
        """Check that the same cube is returned if no fix is available."""
        self.check_data.side_effect = lambda: self.cube
        with patch('esmvalcore.cmor._fixes.fix.Fix.get_fixes',
                   return_value=[]) as mock_get_fixes:
            with patch('esmvalcore.cmor.fix._get_cmor_checker',
                       return_value=self.checker):
                cube_returned = fix_data(
                    self.cube,
                    short_name='short_name',
                    project='project',
                    dataset='model',
                    mip='mip',
                    frequency='frequency',
                    session=sentinel.session,
                )
                self.checker.assert_called_once_with(self.cube)
                self.check_data.assert_called_once_with()
                assert cube_returned is self.cube
                assert cube_returned is not self.intermediate_cube
                assert cube_returned is not self.fixed_cube
                mock_get_fixes.assert_called_once_with(
                    **self.expected_get_fixes_call
                )
