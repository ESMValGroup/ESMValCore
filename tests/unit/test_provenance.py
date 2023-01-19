"""Test `esmvalcore._provenance`."""
from esmvalcore._provenance import TrackedFile


def test_sorting():
    file1 = TrackedFile('file1.nc', attributes={})
    file2 = TrackedFile('file2.nc', attributes={})
    assert sorted([file2, file1]) == [file1, file2]
