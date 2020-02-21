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
    product_entries = ''
    product_urls = ''
    for item in product.provenance.records:
        for key, value in item.attributes:
            if key.namespace.prefix == 'attribute':
                if key.localpart in {'reference', 'references'}:
                    product_entries += '{}\n'.format(_collect_bibtex_citation(value))
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

    # write one bibtex file
    if product_entries:
        with open(f'{product_name}_citation.bibtex', 'w') as file:
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
    author_list = [item['creatorName'] for item in data['creators']]
    if len(author_list) > 1:
        authors = ' and '.join(author_list)
    else:
        authors = author_list[0]
    title = data['titles'][0]
    publisher = data['publisher']
    year = data['publicationYear']
    doi = data['identifier']['id']
    url = f'https://doi.org/{doi}'

    newlinetab = '\n\t'
    newline = '\n'

    bibtex_entry = (
        f'{"@misc{"}{url},{newlinetab}'
        f'url = {{{url}}},{newlinetab}'
        f'title = {{{title}}},{newlinetab}'
        f'publisher = {{{publisher}}},{newlinetab}'
        f'year = {year},{newlinetab}'
        f'author = {{{authors}}},{newlinetab}'
        f'doi = {{{doi}}},{newline}'
        f'{"}"}'
        )
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


def _collect_bibtex_citation(value):
    """Collect information from bibtex files."""
    tags = value.split(',')
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
