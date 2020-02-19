"""Citation module."""
import contextlib
import datetime
import errno
import logging
import numbers
import os
import pprint
import subprocess
import threading
import time
from copy import deepcopy
from multiprocessing import Pool

import requests
import psutil
import yaml

from ._config import DIAGNOSTICS_PATH, TAGS, replace_tags, REFERENCES_PATH
from ._provenance import TrackedFile, get_task_provenance

logger = logging.getLogger(__name__)

DATASET_KEYS = {
    'mip',
}

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
    product_tags = []
    product_entries = ''
    product_urls = ''
    citation = {
        'references': [],
        'info_urls': [],
        'tag': [],
        'entry': '',
        'url': '',
        }
    for item in product.provenance.records:
        for key, value in item.attributes:
            if key.namespace.prefix == 'attribute':
                print(item.attributes[0])
                print('&&&&&&&&&&&&&&&&&&&&&&&&')
                if key.localpart in {'reference', 'references'}:
                    product_entries += '{}\n'.format(_collect_bibtex_citation(product_tags))
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
            file.write(citation['url'])

    # write one bibtex file
    if product_entries:
        with open(f'{product_name}_citation.bibtex.txt', 'w') as file:
            file.write(product_entries)


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
    url = ''.join(['https://doi.org/', data['identifier']['id']])
    author_list = []
    for item in data['creators']:
        author_list.append(item['creatorName'])
    bibtex_entry = ('@misc{' + url + ',\n\t'
                    'url = {' + url + '},\n\t'
                    'title = {' + data['titles'][0] + '},\n\t'
                    'publisher = {' + data['publisher'] + '},\n\t'
                    'year = ' + data['publicationYear'] + ',\n\t'
                    'author = {' + ' and '.join(author_list) + '},\n\t'
                    'doi = {' + data['identifier']['id'] + '},\n'
                    '}')
    return bibtex_entry


def _replace_entry(product_entry):
    """Find tags of the references in provenance."""
    entry_tags = {v: k for k, v in TAGS['references'].items()}
    tag_list = []
    for key in entry_tags.keys():
        for entry in product_entry:
            if key in entry and entry_tags[key] not in tag_list:
                tag_list.append(entry_tags[key])
    return tag_list


def _collect_bibtex_citation(tags):
    """Collect information from bibtex files."""
    entry = ''
    for tag in tags:
        bibtex_file = os.path.join(REFERENCES_PATH, tag + '.bibtex')
        if os.path.isfile(bibtex_file):
            with open(bibtex_file, 'r') as file:
                entry += '{}\n'.format(file.read())
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
    mip_era = attribute.get('attribute:mip_era')
    activity_id = attribute.get('attribute:activity_id')
    institution_id = attribute.get('attribute:institution_id')
    source_id = attribute.get('attribute:source_id')
    experiment_id = attribute.get('attribute:experiment_id')
    url_prefix = f'{mip_era}.{activity_id}.{institution_id}.{source_id}.{experiment_id}'
    json_url = f'{CMIP6_URL_STEM}/cerarest/exportcmip6?input={url_prefix}'
    info_url = f'{CMIP6_URL_STEM}/cmip6?input=CMIP6.{url_prefix}'
    return json_url, info_url
