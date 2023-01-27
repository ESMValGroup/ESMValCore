"""Test `esmvalcore._provenance`."""
from esmvalcore._provenance import TrackedFile


def test_set():
    assert {
        TrackedFile('file1.nc', attributes={}),
        TrackedFile('file1.nc', attributes={}),
        TrackedFile('file2.nc', attributes={}),
    } == {
        TrackedFile('file1.nc', attributes={}),
        TrackedFile('file2.nc', attributes={}),
    }


def test_sort():
    file1 = TrackedFile('file1.nc', attributes={})
    file2 = TrackedFile('file2.nc', attributes={})
    assert sorted([file2, file1]) == [file1, file2]


def test_equals():
    file = TrackedFile('file.nc', attributes={})
    assert file == TrackedFile('file.nc', attributes={})
