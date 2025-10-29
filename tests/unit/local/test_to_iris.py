import iris.cube
import pytest

from esmvalcore.local import LocalFile


@pytest.fixture
def local_file(tmp_path):
    cube = iris.cube.Cube([0])
    cube.attributes.globals["attribute"] = "value"
    file = tmp_path / "test.nc"
    iris.save(cube, file)
    return LocalFile(file)


def test_to_iris(local_file):
    cubes = local_file.to_iris()
    assert len(cubes) == 1


def test_attributes(local_file):
    local_file.to_iris()  # Load the file to populate attributes
    attrs = local_file.attributes
    assert attrs["attribute"] == "value"


def test_attributes_without_loading(local_file):
    """Test that accessing attributes without loading the file first raises."""
    with pytest.raises(
        ValueError,
        match=r"Attributes have not been read yet.*",
    ):
        local_file.attributes  # noqa: B018
