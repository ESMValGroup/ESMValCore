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
    products_tags = []
    product_entries = ''
    product_urls = ''
    for item in product.provenance.records:
        for key, value in item.attributes:
            if key.namespace.prefix == 'attribute':
                if key.localpart in {'reference', 'references'}:
                    products_tags.append(value)
                elif key.localpart == 'mip_era' and value == 'CMIP6':
                    json_url, info_url = _make_url(item.attributes)
                    cmip_entry = _collect_cmip_citation(json_url, info_url)
                    if cmip_entry == info_url:
                        product_urls += '{}\n'.format(cmip_entry)
                    else:
                        product_entries += '{}\n'.format(cmip_entry)

    # save CMIP6 url_info, if any
    if product_urls:
        with open(f'{product_name}_data_citation_url.txt', 'w') as file:
            file.write(product_urls)

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
    """some tages are combined in one string variable in provenance."""
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


def _valid_json_data(data):
    valid_data = False
    keys = ['identifier', 'creators', 'titles', 'publisher', 'publicationYear']
    if all(key in data for key in keys):
        check_names = all('creatorName' in item for item in data['creators'])
        if 'id' in data['identifier'] and check_names:
            valid_data = True
    return valid_data


def _json_to_bibtex(data):
    """Make a bibtex entry from CMIP6 Data Citation json data."""
    author_list = [item['creatorName'] for item in data['creators']]
    if author_list[0] == author_list[-1]:
        authors = author_list[0]
    else:
        authors = ' and '.join(author_list)
    title = data['titles'][0]
    publisher = data['publisher']
    year = data['publicationYear']
    doi = data['identifier']['id']
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
    bibtex_file = os.path.join(REFERENCES_PATH, tag + '.bibtex')
    if os.path.isfile(bibtex_file):
        with open(bibtex_file, 'r') as file:
            entry = '{}'.format(file.read())
    else:
        logger.info('The reference file %s does not exist.',
                    bibtex_file)
    return entry


def _collect_cmip_citation(json_url, info_url):
    """Collect information from CMIP6 Data Citation Service."""
    bibtex_entry = info_url
    json_data = _get_response(json_url)
    if json_data and _valid_json_data(json_data):
        bibtex_entry = _json_to_bibtex(json_data)
    else:
        logger.info('Invalid json link %s', json_url)
    return bibtex_entry


def _make_url(attribute):
    """make json and info urls based on CMIP6 Data Citation Service."""
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
    json_url = f'{CMIP6_URL_STEM}/cerarest/exportcmip6?input={url_prefix}'
    info_url = f'{CMIP6_URL_STEM}/cmip6?input=CMIP6.{url_prefix}'
    return json_url, info_url
