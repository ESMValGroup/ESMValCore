from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from esmvalcore.experimental.recipe_output import (
    DataFile,
    ImageFile,
    OutputFile,
    TableFile,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_output_file_create():
    """Test creation of output file objects."""
    image_file = OutputFile.create("some/image.png")
    assert isinstance(image_file, ImageFile)

    data_file = OutputFile.create("some/data.nc")
    assert isinstance(data_file, DataFile)


def test_output_file_locations():
    """Test methods for location output files."""
    file = OutputFile("output/drc/file.suffix")

    assert file.citation_file.name.endswith("_citation.bibtex")
    assert file.data_citation_file.name.endswith("_data_citation_info.txt")
    assert file.provenance_xml_file.name.endswith("_provenance.xml")


def test_table_file_create() -> None:
    """Create a table file from a CSV path."""
    table_file = OutputFile.create("some/table.csv")
    assert isinstance(table_file, TableFile)
    assert table_file.kind == "table"


def test_table_file_read_rows(tmp_path: Path) -> None:
    """Read quoted and Unicode values without changing the text."""
    path = tmp_path / "table.csv"
    path.write_bytes(
        'city,value,label\r\n"Canberra, Australia",001,"first\r\nsecond"\r\n'
        ",,température\r\n".encode("utf-8-sig"),
    )

    assert TableFile(str(path)).read_rows() == [
        ["city", "value", "label"],
        ["Canberra, Australia", "001", "first\r\nsecond"],
        ["", "", "température"],
    ]


@pytest.mark.parametrize("contents", [None, b"\xff", b'heading\n"unclosed'])
def test_table_file_unreadable(
    tmp_path: Path,
    contents: bytes | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warn if a CSV cannot be read."""
    path = tmp_path / "table.csv"
    if contents is not None:
        path.write_bytes(contents)

    assert TableFile(str(path)).read_rows() == []
    assert f"Could not read CSV table {path}" in caplog.text
