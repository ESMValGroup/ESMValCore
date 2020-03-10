"""Citation module."""
import os
import logging
import re
from pathlib import Path
import requests

from ._config import DIAGNOSTICS_PATH

if DIAGNOSTICS_PATH:
    REFERENCES_PATH = Path(DIAGNOSTICS_PATH) / 'references'
else:
    REFERENCES_PATH = ''

logger = logging.getLogger(__name__)

CMIP6_URL_STEM = 'https://cera-www.dkrz.de/WDCC/ui/cerasearch'

# it is the technical overview and should always be cited
ESMVALTOOL_PAPER_TAG = 'righi19gmdd'
ESMVALTOOL_PAPER = (
    '@article{righi19gmdd,\n\t'
    'doi = {10.5194/gmd-2019-226},\n\t'
    'url = {https://doi.org/10.5194%2Fgmd-2019-226},\n\t'
    'year = 2019,\n\t'
    'month = {sep},\n\t'
    'publisher = {Copernicus {GmbH}},\n\t'
    'author = {Mattia Righi and Bouwe Andela and Veronika Eyring '
    'and Axel Lauer and Valeriu Predoi and Manuel Schlund '
    'and Javier Vegas-Regidor and Lisa Bock and Björn Brötz '
    'and Lee de Mora and Faruk Diblen and Laura Dreyer '
    'and Niels Drost and Paul Earnshaw and Birgit Hassler '
    'and Nikolay Koldunov and Bill Little and Saskia Loosveldt Tomas '
    'and Klaus Zimmermann},\n\t'
    'title = {{ESMValTool} v2.0 '
    '{\\&}amp$\\mathsemicolon${\\#}8211$\\mathsemicolon$ '
    'Technical overview}\n'
    '}\n'
)


def _write_citation_file(filename, provenance):
    """
    Write citation information provided by the recorded provenance.

    Recipe and cmip6 data references are saved into one bibtex file.
    cmip6 data references are provided by CMIP6 data citation service.
    each cmip6 data reference has a json link. In the case of internet
    connection, cmip6 data references are saved into a bibtex file.
    Otherwise, cmip6 data reference links are saved into a text file.
    """
    product_name = os.path.splitext(filename)[0]
    info_urls = []
    json_urls = []
    product_tags = []
    for item in provenance.records:
        # get cmip6 citation info
        value = item.get_attribute('attribute:' + 'mip_era')
        if 'CMIP6' in list(value):
            url_prefix = _make_url_prefix(item.attributes)
            info_urls.append(_make_info_url(url_prefix))
            json_urls.append(_make_json_url(url_prefix))
        # get diagnostics citation tags
        if item.get_attribute('attribute:' + 'script_file'):
            product_tags.append(
                item.get_attribute('attribute:' + 'references').pop()
            )
        # get recipe citation tags
        if item.get_attribute('attribute:' + 'references'):
            if item.identifier.namespace.prefix == 'recipe':
                product_tags.append(
                    item.get_attribute('attribute:' + 'references').pop()
                )
    # get other references information recorded by provenance
    tags = list(set(_clean_tags(product_tags + [ESMVALTOOL_PAPER_TAG])))
    for item in provenance.records:
        if item.get_attribute('attribute:' + 'references'):
            value = item.get_attribute('attribute:' + 'references').pop()
            if value not in tags:
                info_urls.append(value)

    _save_citation_info(product_name, product_tags, json_urls, info_urls)


def _save_citation_info(product_name, product_tags, json_urls, info_urls):
    citation_entries = [ESMVALTOOL_PAPER]

    # save CMIP6 url_info, if any
    # save any refrences info that is not related to recipe or diagnostics
    if info_urls:
        with open(f'{product_name}_data_citation_info.txt', 'w') as file:
            file.write('\n'.join(list(set(info_urls))))

    # convert json_urls to bibtex entries
    for json_url in json_urls:
        cmip_citation = _collect_cmip_citation(json_url)
        if cmip_citation:
            citation_entries.append(cmip_citation)

    # convert tags to bibtex entries
    if REFERENCES_PATH and product_tags:
        # make tags clean and unique
        tags = list(set(_clean_tags(product_tags)))
        for tag in tags:
            citation_entries.append(_collect_bibtex_citation(tag))

    with open(f'{product_name}_citation.bibtex', 'w') as file:
        file.write('\n'.join(citation_entries))


def _clean_tags(tags):
    """Clean the tags that are recorded as str by provenance."""
    pattern = re.compile(r'\w+')
    return pattern.findall(str(tags))


def _get_response(url):
    """Return information from CMIP6 Data Citation service in json format."""
    json_data = False
    if url.lower().startswith('https'):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                json_data = response.json()
            else:
                logger.info('Error in the CMIP json link: %s', url)
        except IOError:
            logger.info('Error in receiving the CMIP json file')
    return json_data


def _json_to_bibtex(data):
    """Make a bibtex entry from CMIP6 Data Citation json data."""
    url = 'url not found'
    title = data.get('titles', ['title not found'])[0]
    publisher = data.get('publisher', 'publisher not found')
    year = data.get('publicationYear', 'publicationYear not found')
    authors = 'creators not found'
    doi = 'doi not found'

    author_list = []
    if data.get('creators', False):
        author_list = [
            item.get('creatorName', '') for item in data['creators']
        ]
    if author_list:
        if author_list[0] == author_list[-1]:
            authors = author_list[0]
            if not authors:
                authors = 'creatorName not found'
        else:
            authors = ' and '.join(author_list)

    if data.get('identifier', False):
        doi = data.get('identifier').get('id', 'doi not found')
        url = f'https://doi.org/{doi}'

    bibtex_entry = (
        f'{"@misc{"}{url},\n\t'
        f'url = {{{url}}},\n\t'
        f'title = {{{title}}},\n\t'
        f'publisher = {{{publisher}}},\n\t'
        f'year = {year},\n\t'
        f'author = {{{authors}}},\n\t'
        f'doi = {{{doi}}},\n'
        f'{"}"}\n'
    )
    return bibtex_entry


def _collect_bibtex_citation(tag):
    """Collect information from bibtex files."""
    bibtex_file = REFERENCES_PATH / f'{tag}.bibtex'
    if bibtex_file.is_file():
        entry = bibtex_file.read_text()
    else:
        logger.info(
            'The reference file %s does not exist.', bibtex_file
        )
        entry = ''
    return entry


def _collect_cmip_citation(json_url):
    """Collect information from CMIP6 Data Citation Service."""
    json_data = _get_response(json_url)
    if json_data:
        bibtex_entry = _json_to_bibtex(json_data)
    else:
        logger.info('Invalid json link %s', json_url)
        bibtex_entry = False
    return bibtex_entry


def _make_url_prefix(attribute):
    """Make url prefix based on CMIP6 Data Citation Service."""
    # the order of keys is important
    localpart = {
        'mip_era': '',
        'activity_id': '',
        'institution_id': '',
        'source_id': '',
        'experiment_id': '',
    }
    for key, value in attribute:
        if key.localpart in localpart:
            localpart[key.localpart] = value
    url_prefix = '.'.join(localpart.values())
    return url_prefix


def _make_json_url(url_prefix):
    """Make json url based on CMIP6 Data Citation Service."""
    json_url = f'{CMIP6_URL_STEM}/cerarest/exportcmip6?input={url_prefix}'
    return json_url


def _make_info_url(url_prefix):
    """Make info url based on CMIP6 Data Citation Service."""
    info_url = f'{CMIP6_URL_STEM}/cmip6?input=CMIP6.{url_prefix}'
    return info_url


def cite_tag_value(tags):
    """Convert tags to bibtex entries."""
    reference_entries = ''
    if REFERENCES_PATH:
        reference_entries = [_collect_bibtex_citation(tag) for tag in [tags]]
        reference_entries = '\n'.join(reference_entries)
    return reference_entries
