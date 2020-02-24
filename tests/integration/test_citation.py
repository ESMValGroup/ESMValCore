"""Test _citation.py."""
from pathlib import Path
from prov.model import ProvDocument

import esmvalcore
from esmvalcore._citation import _write_citation_file, ESMVALTOOL_PAPER
from esmvalcore._provenance import ESMVALTOOL_URI_PREFIX

# Two test cases:
# 1: references are replaced with bibtex
# 2: CMIP6 citation info is retrieved from ES-DOC


def test_references(tmp_path, monkeypatch):
    """Test1: references are replaced with bibtex."""
    # Create fake provenance
    provenance = ProvDocument()
    provenance.add_namespace('file', uri=ESMVALTOOL_URI_PREFIX + 'file')
    provenance.add_namespace('attribute',
                             uri=ESMVALTOOL_URI_PREFIX + 'attribute')

    filename = str(tmp_path / 'output.nc')
    attributes = {'attribute:references': 'test_tag'}
    provenance.entity('file:' + filename, attributes)

    # Create fake bibtex references tag file
    references_path = tmp_path / 'references'
    references_path.mkdir()
    monkeypatch.setattr(
        esmvalcore._citation, 'REFERENCES_PATH', references_path
    )
    fake_bibtex_file = references_path / 'test_tag.bibtex'
    fake_bibtex = "Fake bibtex file content\n"
    fake_bibtex_file.write_text(fake_bibtex)

    _write_citation_file(filename, provenance)
    citation_file = tmp_path / 'output_citation.bibtex'
    citation = citation_file.read_text()
    assert citation == '\n'.join([ESMVALTOOL_PAPER, fake_bibtex])


# def test_cmip6_data_citation(tmp_path, monkeypatch):
