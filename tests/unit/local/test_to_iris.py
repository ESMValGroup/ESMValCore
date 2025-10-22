from pathlib import Path

import iris.cube
import pytest
from pytest_mock import MockerFixture

from esmvalcore.local import LocalFile, _get_attr_from_field_coord


@pytest.fixture
def local_file(tmp_path: Path) -> LocalFile:
    cube = iris.cube.Cube([0])
    cube.attributes.globals["attribute"] = "value"
    file = tmp_path / "test.nc"
    iris.save(cube, file)
    return LocalFile(file)


def test_to_iris(local_file: LocalFile) -> None:
    cubes = local_file.to_iris()
    assert len(cubes) == 1


def test_attributes(local_file: LocalFile) -> None:
    local_file.to_iris()  # Load the file to populate attributes
    attrs = local_file.attributes
    assert attrs["attribute"] == "value"


def test_attributes_without_loading(local_file: LocalFile) -> None:
    """Test that accessing attributes without loading the file first raises."""
    with pytest.raises(
        ValueError,
        match=r"Attributes have not been read yet.*",
    ):
        local_file.attributes  # noqa: B018


def test_get_attr_from_field_coord_none(mocker: MockerFixture) -> None:
    """Test ``_get_attr_from_field_coord``."""
    attr = _get_attr_from_field_coord(mocker.sentinel.ncfield, None, "attr")
    assert attr is None
