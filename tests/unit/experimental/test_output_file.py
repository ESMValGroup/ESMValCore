from esmvalcore.experimental.recipe_output import (
    DataFile,
    ImageFile,
    OutputFile,
)


def test_output_file_create():
    """Test creation of output file objects."""
    image_file = OutputFile.create('some/image.png')
    assert isinstance(image_file, ImageFile)

    data_file = OutputFile.create('some/data.nc')
    assert isinstance(data_file, DataFile)


def test_output_file_locations():
    """Test methods for location output files."""
    file = OutputFile('output/drc/file.suffix')

    assert file.citation_file.name.endswith('_citation.bibtex')
    assert file.data_citation_file.name.endswith('_data_citation_info.txt')
    assert file.provenance_xml_file.name.endswith('_provenance.xml')
