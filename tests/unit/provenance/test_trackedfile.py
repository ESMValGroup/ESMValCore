import pytest
from prov.model import ProvDocument

from esmvalcore._provenance import ESMVALTOOL_URI_PREFIX, TrackedFile


@pytest.fixture
def tracked_file_nc():
    return TrackedFile(
        filename="/path/to/file.nc",
        attributes={"a": "A"},
        prov_filename="/original/path/to/file.nc",
    )


@pytest.fixture
def tracked_file_grb():
    return TrackedFile(
        filename="/path/to/file.grb",
        prov_filename="/original/path/to/file.grb",
    )


def test_init_nc(tracked_file_nc):
    """Test `esmvalcore._provenance.TrackedFile.__init__`."""
    assert tracked_file_nc.filename == "/path/to/file.nc"
    assert tracked_file_nc.attributes == {"a": "A"}
    assert tracked_file_nc.prov_filename == "/original/path/to/file.nc"


def test_init_grb(tracked_file_grb):
    """Test `esmvalcore._provenance.TrackedFile.__init__`."""
    assert tracked_file_grb.filename == "/path/to/file.grb"
    assert tracked_file_grb.attributes is None
    assert tracked_file_grb.prov_filename == "/original/path/to/file.grb"


def test_initialize_provenance_nc(tracked_file_nc):
    """Test `esmvalcore._provenance.TrackedFile.initialize_provenance`."""
    provenance = ProvDocument()
    provenance.add_namespace("task", uri=ESMVALTOOL_URI_PREFIX + "task")
    activity = provenance.activity("task:test-task-name")

    tracked_file_nc.initialize_provenance(activity)
    assert isinstance(tracked_file_nc.provenance, ProvDocument)
    assert tracked_file_nc.activity == activity
    assert str(tracked_file_nc.entity.identifier) == "file:/path/to/file.nc"
    assert tracked_file_nc.attributes == {"a": "A"}


def test_initialize_provenance_grb(tracked_file_grb):
    """Test `esmvalcore._provenance.TrackedFile.initialize_provenance`."""
    provenance = ProvDocument()
    provenance.add_namespace("task", uri=ESMVALTOOL_URI_PREFIX + "task")
    activity = provenance.activity("task:test-task-name")

    tracked_file_grb.initialize_provenance(activity)
    assert isinstance(tracked_file_grb.provenance, ProvDocument)
    assert tracked_file_grb.activity == activity
    assert str(tracked_file_grb.entity.identifier) == "file:/path/to/file.grb"
    assert tracked_file_grb.attributes == {}


def test_copy_provenance(tracked_file_nc):
    """Test `esmvalcore._provenance.TrackedFile.copy_provenance`."""
    provenance = ProvDocument()
    provenance.add_namespace("task", uri=ESMVALTOOL_URI_PREFIX + "task")
    activity = provenance.activity("task:test-task-name")

    tracked_file_nc.initialize_provenance(activity)

    copied_file = tracked_file_nc.copy_provenance()
    assert copied_file.activity == tracked_file_nc.activity
    assert copied_file.entity == tracked_file_nc.entity
    assert copied_file.provenance == tracked_file_nc.provenance
    assert copied_file.provenance is not tracked_file_nc.provenance
