"""Citation module."""
import os
import logging
import re
import requests
from ._config import REFERENCES_PATH

logger = logging.getLogger(__name__)

CMIP6_URL_STEM = 'https://cera-www.dkrz.de/WDCC/ui/cerasearch'


def _write_citation_file(product):
    """
    Write citation information provided by the recorded provenance.

    Recipe and cmip6 data references are saved into one bibtex file.
    cmip6 data references are provided by CMIP6 data citation service.
    each cmip6 data reference has a json link. In the case of internet
    connection, cmip6 data references are saved into a bibtex file.
    Otherwise, cmip6 data reference links are saved into a text file.
    """
    # collect info from provenance
    product_name = os.path.splitext(product.filename)[0]
    info_urls = []
    json_urls = []
    products_tags = []
    for item in product.provenance.records:
        for key, value in item.attributes:
            if key.namespace.prefix == 'attribute':
                if key.localpart in {'reference', 'references'}:
                    products_tags.append(value)
                elif key.localpart == 'mip_era' and value == 'CMIP6':
                    url_prefix = _make_url_prefix(item.attributes)
                    info_urls.append(_make_info_url(url_prefix))
                    json_urls.append(_make_json_url(url_prefix))

    _save_citation_info(product_name, products_tags, json_urls, info_urls)


def _save_citation_info(product_name, products_tags, json_urls, info_urls):
    product_entries = ''
    product_urls = ''

    # save CMIP6 url_info, if any
    if info_urls:
        for info_url in info_urls:
            product_urls += '{}\n'.format(info_url)
        with open(f'{product_name}_data_citation_url.txt', 'w') as file:
            file.write(product_urls)

    # convert json_urls to bibtex entries
    if json_urls:
        for json_url in json_urls:
            product_entries += '{}\n'.format(_collect_cmip_citation(json_url))

    # convert tags to bibtex entries
    if products_tags:
        # make tags clean and unique
        tags = list(set(_clean_tags(products_tags)))
        for tag in tags:
            product_entries += '{}\n'.format(_collect_bibtex_citation(tag))

    # write one bibtex file
    if product_entries:
        with open(f'{product_name}_citation.bibtex', 'w') as file:
            file.write(product_entries)


def _clean_tags(tags):
    """Clean the tages that are recorded as str by provenance."""
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
                logger.info('Error in the CMIP json link')
        except IOError:
            logger.info('Error in receiving the CMIP json file')
    return json_data


def _json_to_bibtex(data):
    """Make a bibtex entry from CMIP6 Data Citation json data."""
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

    title = data.get('titles', ['title not found'])[0]
    publisher = data.get('publisher', 'publisher not found')
    year = data.get('publicationYear', 'publicationYear not found')

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
    if REFERENCES_PATH:
        bibtex_file = os.path.join(REFERENCES_PATH, tag + '.bibtex')
        if os.path.isfile(bibtex_file):
            with open(bibtex_file, 'r') as file:
                entry = '{}'.format(file.read())
        else:
            raise ValueError(
                'The reference file {} does not exist.'.format(bibtex_file)
            )
    else:
        logger.info('The reference folder does not exist.')
        entry = ''
    return entry


def _collect_cmip_citation(json_url):
    """Collect information from CMIP6 Data Citation Service."""
    json_data = _get_response(json_url)
    if json_data:
        bibtex_entry = _json_to_bibtex(json_data)
    else:
        logger.info('Invalid json link %s', json_url)
        bibtex_entry = 'Invalid json link'
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
