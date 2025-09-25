"""Test `esmvalcore._provenance`."""

from pathlib import Path

from esmvalcore._provenance import TrackedFile


def test_set():
    assert {
        TrackedFile(Path("file1.nc"), attributes={}),
        TrackedFile(Path("file1.nc"), attributes={}),
        TrackedFile(Path("file2.nc"), attributes={}),
    } == {
        TrackedFile(Path("file1.nc"), attributes={}),
        TrackedFile(Path("file2.nc"), attributes={}),
    }


def test_sort():
    file1 = TrackedFile(Path("file1.nc"), attributes={})
    file2 = TrackedFile(Path("file2.nc"), attributes={})
    assert sorted([file2, file1]) == [file1, file2]


def test_equals():
    file = TrackedFile(Path("file.nc"), attributes={})
    assert file == TrackedFile(Path("file.nc"), attributes={})
