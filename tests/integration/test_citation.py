"""Test _citation.py."""
import textwrap

from prov.model import ProvDocument

import esmvalcore
from esmvalcore._citation import (CMIP6_URL_STEM, ESMVALTOOL_PAPER,
                                  _write_citation_files)
from esmvalcore._provenance import ESMVALTOOL_URI_PREFIX


def test_references(tmp_path, monkeypatch):
    """Test1: references are replaced with bibtex."""
    # Create fake provenance
    provenance = ProvDocument()
    provenance.add_namespace('file', uri=ESMVALTOOL_URI_PREFIX + 'file')
    provenance.add_namespace('attribute',
                             uri=ESMVALTOOL_URI_PREFIX + 'attribute')
    filename = str(tmp_path / 'output.nc')
    attributes = {
        'attribute:references': 'test_tag',
        'attribute:script_file': 'diagnostics.py'
    }
    provenance.entity('file:' + filename, attributes)

    # Create fake bibtex references tag file
    references_path = tmp_path / 'references'
    references_path.mkdir()
    monkeypatch.setattr(esmvalcore._citation.DIAGNOSTICS, 'path', tmp_path)
    fake_bibtex_file = references_path / 'test_tag.bibtex'
    fake_bibtex = "Fake bibtex file content\n"
    fake_bibtex_file.write_text(fake_bibtex)

    _write_citation_files(filename, provenance)
    citation_file = tmp_path / 'output_citation.bibtex'
    citation = citation_file.read_text()
    assert citation == '\n'.join([ESMVALTOOL_PAPER, fake_bibtex])


def mock_get_response(url):
    """Mock _get_response() function."""
    json_data = False
    if url.lower().startswith('https'):
        json_data = {'titles': ['title is found']}
    return json_data


def test_cmip6_data_citation(tmp_path, monkeypatch):
    """Test2: CMIP6 citation info is retrieved from ES-DOC."""
    # Create fake provenance
    provenance = ProvDocument()
    provenance.add_namespace('file', uri=ESMVALTOOL_URI_PREFIX + 'file')
    provenance.add_namespace('attribute',
                             uri=ESMVALTOOL_URI_PREFIX + 'attribute')
    attributes = {
        'attribute:mip_era': 'CMIP6',
        'attribute:activity_id': 'activity',
        'attribute:institution_id': 'institution',
        'attribute:source_id': 'source',
        'attribute:experiment_id': 'experiment',
    }
    filename = str(tmp_path / 'output.nc')
    provenance.entity('file:' + filename, attributes)

    monkeypatch.setattr(esmvalcore._citation, '_get_response',
                        mock_get_response)
    _write_citation_files(filename, provenance)
    citation_file = tmp_path / 'output_citation.bibtex'

    # Create fake bibtex entry
    url = 'url not found'
    title = 'title is found'
    publisher = 'publisher not found'
    year = 'publicationYear not found'
    authors = 'creators not found'
    doi = 'doi not found'
    fake_bibtex_entry = textwrap.dedent(f"""
        @misc{{{url},
        \turl = {{{url}}},
        \ttitle = {{{title}}},
        \tpublisher = {{{publisher}}},
        \tyear = {year},
        \tauthor = {{{authors}}},
        \tdoi = {{{doi}}},
        }}
        """).lstrip()
    assert citation_file.read_text() == '\n'.join(
        [ESMVALTOOL_PAPER, fake_bibtex_entry])


def test_cmip6_data_citation_url(tmp_path):
    """Test3: CMIP6 info_url is retrieved from ES-DOC."""
    # Create fake provenance
    provenance = ProvDocument()
    provenance.add_namespace('file', uri=ESMVALTOOL_URI_PREFIX + 'file')
    provenance.add_namespace('attribute',
                             uri=ESMVALTOOL_URI_PREFIX + 'attribute')
    attributes = {
        'attribute:mip_era': 'CMIP6',
        'attribute:activity_id': 'activity',
        'attribute:institution_id': 'institution',
        'attribute:source_id': 'source',
        'attribute:experiment_id': 'experiment',
    }
    filename = str(tmp_path / 'output.nc')
    provenance.entity('file:' + filename, attributes)
    _write_citation_files(filename, provenance)
    citation_url = tmp_path / 'output_data_citation_info.txt'

    # Create fake info url
    fake_url_prefix = '.'.join(attributes.values())
    text = '\n'.join([
        "Follow the links below to find more information about CMIP6 data:",
        f"- {CMIP6_URL_STEM}/cmip6?input={fake_url_prefix}",
        '',
    ])
    assert citation_url.read_text() == text
