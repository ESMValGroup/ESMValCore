"""Unit tests for :mod:`esmvalcore.cmor.fix`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch, sentinel

import iris
import iris.cube
import pytest

from esmvalcore.cmor.fix import Fix, fix_data, fix_file, fix_metadata
from esmvalcore.io.local import LocalFile
from esmvalcore.io.protocol import DataElement

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestFixFile:
    """Fix file tests."""

    @pytest.fixture(autouse=True)
    def setUp(self):
        """Prepare for testing."""
        self.filename = Path("filename")
        self.mock_fix = Mock()
        self.mock_fix.fix_file.return_value = Path("new_filename")
        self.expected_get_fixes_call = {
            "project": "project",
            "dataset": "model",
            "mip": "mip",
            "short_name": "short_name",
            "extra_facets": {
                "project": "project",
                "dataset": "model",
                "mip": "mip",
                "short_name": "short_name",
                "frequency": "frequency",
            },
            "session": sentinel.session,
            "frequency": "frequency",
        }

    def test_fix(self, mocker: MockerFixture) -> None:
        """Check that the returned fix is applied."""
        mock_get_fixes = mocker.patch(
            "esmvalcore.cmor._fixes.fix.Fix.get_fixes",
            return_value=[self.mock_fix],
        )
        file_returned = fix_file(
            file=Path("filename"),
            short_name="short_name",
            project="project",
            dataset="model",
            mip="mip",
            output_dir=Path("output_dir"),
            session=sentinel.session,
            frequency="frequency",
        )
        assert file_returned != self.filename
        assert file_returned == Path("new_filename")
        mock_get_fixes.assert_called_once_with(
            **self.expected_get_fixes_call,
        )

    def test_fix_returns_cubes(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Check that the returned fix is applied."""
        # Prepare some mock fixed data and save it to a file.
        fixed_file = LocalFile(tmp_path / "new_filename.nc")
        fixed_cube = iris.cube.Cube([0], var_name="tas")
        fixed_cube.attributes.globals = {"a": "b"}
        iris.save(fixed_cube, fixed_file)

        # Set up a mock fix to that returns this data.
        self.mock_fix.fix_file.return_value = fixed_file
        mock_get_fixes = mocker.patch(
            "esmvalcore.cmor._fixes.fix.Fix.get_fixes",
            return_value=[self.mock_fix],
        )

        mock_input_file = LocalFile(self.filename)
        result = fix_file(
            file=mock_input_file,
            short_name="short_name",
            project="project",
            dataset="model",
            mip="mip",
            output_dir=Path("output_dir"),
            session=sentinel.session,
            frequency="frequency",
        )
        # Check that a sequence of cubes is returned and that the attributes
        # of the input file have been updated with the global attributes of the
        # fixed cube for recording provenance.
        assert isinstance(result, Sequence)
        assert len(result) == 1
        assert isinstance(result[0], iris.cube.Cube)
        assert result[0].var_name == "tas"
        assert "a" in mock_input_file.attributes
        assert mock_input_file.attributes["a"] == "b"
        mock_get_fixes.assert_called_once_with(
            **self.expected_get_fixes_call,
        )

    def test_nofix(self, mocker: MockerFixture) -> None:
        """Check that the same file is returned if no fix is available."""
        mock_get_fixes = mocker.patch(
            "esmvalcore.cmor._fixes.fix.Fix.get_fixes",
            return_value=[],
        )
        file_returned = fix_file(
            file=Path("filename"),
            short_name="short_name",
            project="project",
            dataset="model",
            mip="mip",
            output_dir=Path("output_dir"),
            session=sentinel.session,
            frequency="frequency",
        )
        assert file_returned == self.filename
        mock_get_fixes.assert_called_once_with(
            **self.expected_get_fixes_call,
        )

    def test_nofix_if_not_path(self, mocker: MockerFixture) -> None:
        """Check that the same object is returned if the input is not a Path."""
        mock_data_element = mocker.create_autospec(DataElement, instance=True)
        file_returned = fix_file(
            file=mock_data_element,
            short_name="short_name",
            project="project",
            dataset="model",
            mip="mip",
            output_dir=Path("output_dir"),
            session=sentinel.session,
            frequency="frequency",
        )
        assert file_returned is mock_data_element


class TestGetCube:
    """Test get cube by var_name method."""

    @pytest.fixture(autouse=True)
    def setUp(self):
        """Prepare for testing."""
        self.cube_1 = Mock()
        self.cube_1.var_name = "cube1"
        self.cube_2 = Mock()
        self.cube_2.var_name = "cube2"
        self.cubes = [self.cube_1, self.cube_2]
        vardef = Mock()
        vardef.short_name = "fix"
        self.fix = Fix(vardef)

    def test_get_first_cube(self):
        """Test selecting first cube."""
        assert self.cube_1 is self.fix.get_cube_from_list(self.cubes, "cube1")

    def test_get_second_cube(self):
        """Test selecting second cube."""
        assert self.cube_2 is self.fix.get_cube_from_list(self.cubes, "cube2")

    def test_get_default_raises(self):
        """Check that the default raises (Fix is not a cube)."""
        with pytest.raises(ValueError):
            self.fix.get_cube_from_list(self.cubes)

    def test_get_default(self):
        """Check that the default return the cube (fix is a cube)."""
        self.cube_1.var_name = "fix"
        assert self.cube_1 is self.fix.get_cube_from_list(self.cubes)


class TestFixMetadata:
    """Fix metadata tests."""

    @pytest.fixture(autouse=True)
    def setUp(self):
        """Prepare for testing."""
        self.cube = self._create_mock_cube()
        self.fixed_cube = self._create_mock_cube()
        self.mock_fix = Mock()
        self.mock_fix.fix_metadata.return_value = [self.fixed_cube]
        self.expected_get_fixes_call = {
            "project": "project",
            "dataset": "model",
            "mip": "mip",
            "short_name": "short_name",
            "extra_facets": {
                "project": "project",
                "dataset": "model",
                "mip": "mip",
                "short_name": "short_name",
                "frequency": "frequency",
            },
            "session": sentinel.session,
            "frequency": "frequency",
        }

    @staticmethod
    def _create_mock_cube(var_name="short_name"):
        cube = Mock()
        cube.var_name = var_name
        cube.attributes = {"source_file": "source_file"}
        return cube

    def test_fix(self):
        """Check that the returned fix is applied."""
        with patch(
            "esmvalcore.cmor._fixes.fix.Fix.get_fixes",
            return_value=[self.mock_fix],
        ) as mock_get_fixes:
            cube_returned = fix_metadata(
                cubes=[self.cube],
                short_name="short_name",
                project="project",
                dataset="model",
                mip="mip",
                frequency="frequency",
                session=sentinel.session,
            )[0]
            assert cube_returned is not self.cube
            assert cube_returned is self.fixed_cube
            mock_get_fixes.assert_called_once_with(
                **self.expected_get_fixes_call,
            )

    def test_nofix(self):
        """Check that the same cube is returned if no fix is available."""
        with patch(
            "esmvalcore.cmor._fixes.fix.Fix.get_fixes",
            return_value=[],
        ) as mock_get_fixes:
            cube_returned = fix_metadata(
                cubes=[self.cube],
                short_name="short_name",
                project="project",
                dataset="model",
                mip="mip",
                frequency="frequency",
                session=sentinel.session,
            )[0]
            assert cube_returned is self.cube
            assert cube_returned is not self.fixed_cube
            mock_get_fixes.assert_called_once_with(
                **self.expected_get_fixes_call,
            )

    def test_select_var(self):
        """Check that the same cube is returned if no fix is available."""
        with patch(
            "esmvalcore.cmor._fixes.fix.Fix.get_fixes",
            return_value=[],
        ):
            cube_returned = fix_metadata(
                cubes=[self.cube, self._create_mock_cube("extra")],
                short_name="short_name",
                project="CMIP6",
                dataset="model",
                mip="mip",
            )[0]
            assert cube_returned is self.cube

    def test_select_var_failed_if_bad_var_name(self):
        """Check that an error is raised if short_names do not match."""
        msg = "More than one cube found for variable tas in CMIP6:model"
        with pytest.raises(ValueError, match=msg):
            fix_metadata(
                cubes=[
                    self._create_mock_cube("not_me"),
                    self._create_mock_cube("me_neither"),
                ],
                short_name="tas",
                project="CMIP6",
                dataset="model",
                mip="Amon",
            )


class TestFixData:
    """Fix data tests."""

    @pytest.fixture(autouse=True)
    def setUp(self):
        """Prepare for testing."""
        self.cube = Mock()
        self.fixed_cube = Mock()
        self.mock_fix = Mock()
        self.mock_fix.fix_data.return_value = self.fixed_cube
        self.expected_get_fixes_call = {
            "project": "project",
            "dataset": "model",
            "mip": "mip",
            "short_name": "short_name",
            "extra_facets": {
                "project": "project",
                "dataset": "model",
                "mip": "mip",
                "short_name": "short_name",
                "frequency": "frequency",
            },
            "session": sentinel.session,
            "frequency": "frequency",
        }

    def test_fix(self):
        """Check that the returned fix is applied."""
        with patch(
            "esmvalcore.cmor._fixes.fix.Fix.get_fixes",
            return_value=[self.mock_fix],
        ) as mock_get_fixes:
            cube_returned = fix_data(
                self.cube,
                short_name="short_name",
                project="project",
                dataset="model",
                mip="mip",
                frequency="frequency",
                session=sentinel.session,
            )
            assert cube_returned is not self.cube
            assert cube_returned is self.fixed_cube
            mock_get_fixes.assert_called_once_with(
                **self.expected_get_fixes_call,
            )

    def test_nofix(self):
        """Check that the same cube is returned if no fix is available."""
        with patch(
            "esmvalcore.cmor._fixes.fix.Fix.get_fixes",
            return_value=[],
        ) as mock_get_fixes:
            cube_returned = fix_data(
                self.cube,
                short_name="short_name",
                project="project",
                dataset="model",
                mip="mip",
                frequency="frequency",
                session=sentinel.session,
            )
            assert cube_returned is self.cube
            assert cube_returned is not self.fixed_cube
            mock_get_fixes.assert_called_once_with(
                **self.expected_get_fixes_call,
            )
