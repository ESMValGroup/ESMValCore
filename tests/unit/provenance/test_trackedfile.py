import pytest
from prov.model import ProvDocument

from esmvalcore._provenance import ESMVALTOOL_URI_PREFIX, TrackedFile


@pytest.fixture
def tracked_file():
    file = TrackedFile(
        filename='/path/to/file.nc',
        attributes={'a': 'A'},
        prov_filename='/original/path/to/file.nc',
    )
    return file


def test_init(tracked_file):
    """Test `esmvalcore._provenance.TrackedFile.__init__`."""
    assert tracked_file.filename == '/path/to/file.nc'
    assert tracked_file.attributes == {'a': 'A'}
    assert tracked_file.prov_filename == '/original/path/to/file.nc'


def test_initialize_provenance(tracked_file):
    """Test `esmvalcore._provenance.TrackedFile.initialize_provenancee`."""
    provenance = ProvDocument()
    provenance.add_namespace('task', uri=ESMVALTOOL_URI_PREFIX + 'task')
    activity = provenance.activity('task:test-task-name')

    tracked_file.initialize_provenance(activity)
    assert isinstance(tracked_file.provenance, ProvDocument)
    assert tracked_file.activity == activity
    assert str(tracked_file.entity.identifier) == 'file:/path/to/file.nc'


def test_copy_provenance(tracked_file):
    """Test `esmvalcore._provenance.TrackedFile.copy_provenance`."""
    provenance = ProvDocument()
    provenance.add_namespace('task', uri=ESMVALTOOL_URI_PREFIX + 'task')
    activity = provenance.activity('task:test-task-name')

    tracked_file.initialize_provenance(activity)

    copied_file = tracked_file.copy_provenance()
    assert copied_file.activity == tracked_file.activity
    assert copied_file.entity == tracked_file.entity
    assert copied_file.provenance == tracked_file.provenance
    assert copied_file.provenance is not tracked_file.provenance
