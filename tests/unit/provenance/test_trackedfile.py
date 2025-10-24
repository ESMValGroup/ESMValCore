from pathlib import Path

import pytest
from prov.model import ProvDocument

from esmvalcore._provenance import ESMVALTOOL_URI_PREFIX, TrackedFile
from esmvalcore.local import LocalFile


@pytest.fixture
def tracked_input_file_nc():
    input_file_nc = LocalFile("/path/to/file.nc")
    input_file_nc.attributes = {"a": "A"}
    return TrackedFile(filename=input_file_nc)


@pytest.fixture
def tracked_output_file_nc():
    return TrackedFile(
        filename=Path("/path/to/file.nc"),
        attributes={"a": "A"},
    )


@pytest.fixture
def tracked_input_file_grb():
    input_file_grb = LocalFile("/path/to/file.grb")
    input_file_grb.attributes = {"a": "A"}
    return TrackedFile(filename=input_file_grb)


def test_init_input_nc(tracked_input_file_nc):
    """Test `esmvalcore._provenance.TrackedFile.__init__`."""
    assert tracked_input_file_nc.filename == LocalFile("/path/to/file.nc")
    assert tracked_input_file_nc.attributes is None


def test_init_output_nc(tracked_output_file_nc):
    """Test `esmvalcore._provenance.TrackedFile.__init__`."""
    assert tracked_output_file_nc.filename == Path("/path/to/file.nc")
    assert tracked_output_file_nc.attributes == {"a": "A"}


def test_init_grb(tracked_input_file_grb):
    """Test `esmvalcore._provenance.TrackedFile.__init__`."""
    assert tracked_input_file_grb.filename == LocalFile("/path/to/file.grb")
    assert tracked_input_file_grb.attributes is None


@pytest.mark.parametrize(
    "fixture_name",
    ["tracked_input_file_nc", "tracked_output_file_nc"],
)
def test_initialize_provenance_nc(fixture_name, request):
    """Test `esmvalcore._provenance.TrackedFile.initialize_provenance`."""
    tracked_file_nc = request.getfixturevalue(fixture_name)
    provenance = ProvDocument()
    provenance.add_namespace("task", uri=ESMVALTOOL_URI_PREFIX + "task")
    activity = provenance.activity("task:test-task-name")

    tracked_file_nc.initialize_provenance(activity)
    assert isinstance(tracked_file_nc.provenance, ProvDocument)
    assert tracked_file_nc.activity == activity
    assert str(tracked_file_nc.entity.identifier) == "file:/path/to/file.nc"
    assert tracked_file_nc.attributes == {"a": "A"}


def test_initialize_provenance_grb(tracked_input_file_grb):
    """Test `esmvalcore._provenance.TrackedFile.initialize_provenance`."""
    provenance = ProvDocument()
    provenance.add_namespace("task", uri=ESMVALTOOL_URI_PREFIX + "task")
    activity = provenance.activity("task:test-task-name")

    tracked_input_file_grb.initialize_provenance(activity)
    assert isinstance(tracked_input_file_grb.provenance, ProvDocument)
    assert tracked_input_file_grb.activity == activity
    assert (
        str(tracked_input_file_grb.entity.identifier)
        == "file:/path/to/file.grb"
    )
    assert tracked_input_file_grb.attributes == {"a": "A"}


@pytest.mark.parametrize(
    "fixture_name",
    ["tracked_input_file_nc", "tracked_output_file_nc"],
)
def test_copy_provenance(fixture_name, request):
    """Test `esmvalcore._provenance.TrackedFile.copy_provenance`."""
    tracked_file_nc = request.getfixturevalue(fixture_name)
    provenance = ProvDocument()
    provenance.add_namespace("task", uri=ESMVALTOOL_URI_PREFIX + "task")
    activity = provenance.activity("task:test-task-name")

    tracked_file_nc.initialize_provenance(activity)

    copied_file = tracked_file_nc.copy_provenance()
    assert copied_file.activity == tracked_file_nc.activity
    assert copied_file.entity == tracked_file_nc.entity
    assert copied_file.provenance == tracked_file_nc.provenance
    assert copied_file.provenance is not tracked_file_nc.provenance
