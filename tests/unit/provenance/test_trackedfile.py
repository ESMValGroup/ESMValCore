from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from prov.model import ProvDocument

from esmvalcore._provenance import ESMVALTOOL_URI_PREFIX, TrackedFile
from esmvalcore.io.protocol import DataElement
from esmvalcore.local import LocalFile

if TYPE_CHECKING:
    import iris.cube
    import prov.model


def test_set() -> None:
    assert {
        TrackedFile(Path("file1.nc"), attributes={}),
        TrackedFile(Path("file1.nc"), attributes={}),
        TrackedFile(Path("file2.nc"), attributes={}),
    } == {
        TrackedFile(Path("file1.nc"), attributes={}),
        TrackedFile(Path("file2.nc"), attributes={}),
    }


def test_sort() -> None:
    file1 = TrackedFile(Path("file1.nc"), attributes={})
    file2 = TrackedFile(Path("file2.nc"), attributes={})
    assert sorted([file2, file1]) == [file1, file2]


def test_equals() -> None:
    file = TrackedFile(Path("file.nc"), attributes={})
    assert file == TrackedFile(Path("file.nc"), attributes={})


@pytest.fixture
def tracked_input_file_nc() -> TrackedFile:
    input_file_nc = LocalFile("/path/to/file.nc")
    input_file_nc.attributes = {"a": "A"}
    return TrackedFile(filename=input_file_nc)


@pytest.fixture
def tracked_output_file_nc() -> TrackedFile:
    return TrackedFile(
        filename=Path("/path/to/file.nc"),
        attributes={"a": "A"},
    )


@pytest.fixture
def tracked_input_file_grb() -> TrackedFile:
    input_file_grb = LocalFile("/path/to/file.grb")
    input_file_grb.attributes = {"a": "A"}
    return TrackedFile(filename=input_file_grb)


def test_init_input_nc(tracked_input_file_nc: TrackedFile) -> None:
    """Test `esmvalcore._provenance.TrackedFile.__init__`."""
    assert tracked_input_file_nc.filename == LocalFile("/path/to/file.nc")
    with pytest.raises(
        ValueError,
        match=r"Call TrackedFile.initialize_provenance before accessing attributes",
    ):
        tracked_input_file_nc.attributes  # noqa: B018


def test_init_output_nc(tracked_output_file_nc: TrackedFile) -> None:
    """Test `esmvalcore._provenance.TrackedFile.__init__`."""
    assert tracked_output_file_nc.filename == Path("/path/to/file.nc")
    assert tracked_output_file_nc.attributes == {"a": "A"}


def test_init_grb(tracked_input_file_grb: TrackedFile) -> None:
    """Test `esmvalcore._provenance.TrackedFile.__init__`."""
    assert tracked_input_file_grb.filename == LocalFile("/path/to/file.grb")
    with pytest.raises(
        ValueError,
        match=r"Call TrackedFile.initialize_provenance before accessing attributes",
    ):
        tracked_input_file_grb.attributes  # noqa: B018


@pytest.fixture
def activity() -> prov.model.ProvActivity:
    provenance = ProvDocument()
    provenance.add_namespace("task", uri=ESMVALTOOL_URI_PREFIX + "task")
    return provenance.activity("task:test-task-name")


@pytest.mark.parametrize(
    "fixture_name",
    ["tracked_input_file_nc", "tracked_output_file_nc"],
)
def test_initialize_provenance_nc(
    fixture_name: str,
    request: pytest.FixtureRequest,
    activity: prov.model.ProvActivity,
) -> None:
    """Test `esmvalcore._provenance.TrackedFile.initialize_provenance`."""
    tracked_file_nc = request.getfixturevalue(fixture_name)
    tracked_file_nc.initialize_provenance(activity)
    assert isinstance(tracked_file_nc.provenance, ProvDocument)
    assert tracked_file_nc.activity == activity
    assert str(tracked_file_nc.entity.identifier) == "file:/path/to/file.nc"
    assert tracked_file_nc.attributes == {"a": "A"}


def test_initialize_provenance_grb(
    tracked_input_file_grb: TrackedFile,
    activity: prov.model.ProvActivity,
) -> None:
    """Test `esmvalcore._provenance.TrackedFile.initialize_provenance`."""
    tracked_input_file_grb.initialize_provenance(activity)
    assert isinstance(tracked_input_file_grb.provenance, ProvDocument)
    assert tracked_input_file_grb.activity == activity
    assert (
        str(tracked_input_file_grb.entity.identifier)  # type: ignore[attr-defined]
        == "file:/path/to/file.grb"
    )
    assert tracked_input_file_grb.attributes == {"a": "A"}


def test_initialize_provenance_twice_raises(
    tracked_output_file_nc: TrackedFile,
    activity: prov.model.ProvActivity,
) -> None:
    """Test `esmvalcore._provenance.TrackedFile.initialize_provenance` raises if called twice."""
    tracked_output_file_nc.initialize_provenance(activity)

    with pytest.raises(
        ValueError,
        match=r"Provenance of TrackedFile: /path/to/file.nc already initialized",
    ):
        tracked_output_file_nc.initialize_provenance(activity)


def test_initialize_provenance_no_attributes_raises(
    activity: prov.model.ProvActivity,
) -> None:
    """Test `esmvalcore._provenance.TrackedFile.initialize_provenance` with no attributes."""
    tracked_file = TrackedFile(filename=Path("/path/to/file.nc"))

    with pytest.raises(
        TypeError,
        match=r"Delayed reading of attributes is only supported for `DataElement`s",
    ):
        tracked_file.initialize_provenance(activity)


@pytest.mark.parametrize(
    "fixture_name",
    ["tracked_input_file_nc", "tracked_output_file_nc"],
)
def test_copy_provenance(
    fixture_name: str,
    request: pytest.FixtureRequest,
    activity: prov.model.ProvActivity,
) -> None:
    """Test `esmvalcore._provenance.TrackedFile.copy_provenance`."""
    tracked_file_nc = request.getfixturevalue(fixture_name)
    tracked_file_nc.initialize_provenance(activity)

    copied_file = tracked_file_nc.copy_provenance()
    assert copied_file.activity == tracked_file_nc.activity
    assert copied_file.entity == tracked_file_nc.entity
    assert copied_file.provenance == tracked_file_nc.provenance
    assert copied_file.provenance is not tracked_file_nc.provenance


def test_copy_provenance_not_initialized() -> None:
    """Test `esmvalcore._provenance.TrackedFile.copy_provenance` raises if provenance not initialized."""
    tracked_file = TrackedFile(filename=Path("/path/to/file.nc"))

    with pytest.raises(
        ValueError,
        match=r"Provenance of TrackedFile: /path/to/file.nc not initialized",
    ):
        tracked_file.copy_provenance()


def test_wasderivedfrom_not_initialized() -> None:
    """Test `esmvalcore._provenance.TrackedFile.wasderivedfrom` raises if provenance not initialized."""
    tracked_file = TrackedFile(filename=Path("/path/to/file.nc"))
    other_tracked_file = TrackedFile(filename=Path("/path/to/other_file.nc"))

    with pytest.raises(
        ValueError,
        match=r"Provenance of TrackedFile: /path/to/file.nc not initialized",
    ):
        tracked_file.wasderivedfrom(other_tracked_file)


@dataclass
class MockDataElement(DataElement):
    """Mock DataElement for testing purposes."""

    name: str
    facets: dict[str, Any]
    attributes: dict[str, Any]

    def prepare(self) -> None:
        pass

    def __hash__(self) -> int:
        return hash(self.name)

    def to_iris(self) -> iris.cube.CubeList:
        return []


def test_provenance_file_nonpath_notimplemented() -> None:
    """Test `esmvalcore._provenance.TrackedFile.provenance_file` with a DataElement."""
    input_file = MockDataElement(
        name="/path/to/input_file.nc",
        facets={},
        attributes={},
    )
    tracked_file = TrackedFile(filename=input_file)

    assert tracked_file.filename == input_file
    with pytest.raises(
        NotImplementedError,
        match=r"Saving provenance is only supported for pathlib.Path.*",
    ):
        _ = tracked_file.provenance_file


def test_save_provenance_notimplemented(
    activity: prov.model.ProvActivity,
) -> None:
    """Test `esmvalcore._provenance.TrackedFile.save_provenance` with a DataElement."""
    input_file = MockDataElement(
        name="/path/to/input_file.nc",
        facets={},
        attributes={},
    )
    tracked_file = TrackedFile(filename=input_file)
    tracked_file.initialize_provenance(activity)

    with pytest.raises(
        NotImplementedError,
        match=r"Writing attributes is only supported for pathlib.Path.*",
    ):
        tracked_file.save_provenance()
