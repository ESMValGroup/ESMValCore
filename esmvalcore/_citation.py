"""Citation module."""
import logging
import os
import re
import textwrap
from functools import lru_cache

import requests

from ._config import DIAGNOSTICS

logger = logging.getLogger(__name__)

CMIP6_URL_STEM = 'https://cera-www.dkrz.de/WDCC/ui/cerasearch'

# The technical overview paper should always be cited
ESMVALTOOL_PAPER = (
    "@article{righi20gmd,\n"
    "\tdoi = {10.5194/gmd-13-1179-2020},\n"
    "\turl = {https://doi.org/10.5194/gmd-13-1179-2020},\n"
    "\tyear = {2020},\n"
    "\tmonth = mar,\n"
    "\tpublisher = {Copernicus {GmbH}},\n"
    "\tvolume = {13},\n"
    "\tnumber = {3},\n"
    "\tpages = {1179--1199},\n"
    "\tauthor = {Mattia Righi and Bouwe Andela and Veronika Eyring "
    "and Axel Lauer and Valeriu Predoi and Manuel Schlund "
    "and Javier Vegas-Regidor and Lisa Bock and Bj\"{o}rn Br\"{o}tz "
    "and Lee de Mora and Faruk Diblen and Laura Dreyer "
    "and Niels Drost and Paul Earnshaw and Birgit Hassler "
    "and Nikolay Koldunov and Bill Little and Saskia Loosveldt Tomas "
    "and Klaus Zimmermann},\n"
    "\ttitle = {Earth System Model Evaluation Tool (ESMValTool) v2.0 "
    "-- technical overview},\n"
    "\tjournal = {Geoscientific Model Development}\n"
    "}\n")


def _write_citation_files(filename, provenance):
    """
    Write citation information provided by the recorded provenance.

    Recipe and cmip6 data references are saved into one bibtex file.
    cmip6 data references are provided by CMIP6 data citation service.
    Each cmip6 data reference has a json link. In the case of internet
    connection, cmip6 data references are saved into a bibtex file.
    Also, cmip6 data reference links are saved into a text file.
    """
    product_name = os.path.splitext(filename)[0]

    tags = set()
    cmip6_json_urls = set()
    cmip6_info_urls = set()
    other_info = set()

    for item in provenance.records:
        # get cmip6 data citation info
        cmip6_data = 'CMIP6' in item.get_attribute('attribute:mip_era')
        if cmip6_data:
            url_prefix = _make_url_prefix(item.attributes)
            cmip6_info_urls.add(_make_info_url(url_prefix))
            cmip6_json_urls.add(_make_json_url(url_prefix))

        # get other citation info
        references = item.get_attribute('attribute:references')
        if not references:
            # ESMValTool CMORization scripts use 'reference' (without final s)
            references = item.get_attribute('attribute:reference')
        if references:
            if item.identifier.namespace.prefix == 'recipe':
                # get recipe citation tags
                tags.update(references)
            elif item.get_attribute('attribute:script_file'):
                # get diagnostics citation tags
                tags.update(references)
            elif not cmip6_data:
                # get any other data citation tags, e.g. CMIP5
                other_info.update(references)

    _save_citation_bibtex(product_name, tags, cmip6_json_urls)
    _save_citation_info_txt(product_name, cmip6_info_urls, other_info)


def _save_citation_bibtex(product_name, tags, json_urls):
    """Save the bibtex entries in a bibtex file."""
    citation_entries = [ESMVALTOOL_PAPER]

    # convert tags to bibtex entries
    if tags:
        entries = set()
        for tag in _extract_tags(tags):
            entries.add(_collect_bibtex_citation(tag))
        citation_entries.extend(sorted(entries))

    # convert json_urls to bibtex entries
    entries = set()
    for json_url in json_urls:
        cmip_citation = _collect_cmip_citation(json_url)
        if cmip_citation:
            entries.add(cmip_citation)
    citation_entries.extend(sorted(entries))

    with open(f'{product_name}_citation.bibtex', 'w') as file:
        file.write('\n'.join(citation_entries))


def _save_citation_info_txt(product_name, info_urls, other_info):
    """Save all data citation information in one text file."""
    lines = []
    # Save CMIP6 url_info
    if info_urls:
        lines.append(
            "Follow the links below to find more information about CMIP6 data:"
        )
        lines.extend(f'- {url}' for url in sorted(info_urls))

    # Save any references from the 'references' and 'reference' NetCDF global
    # attributes.
    if other_info:
        if lines:
            lines.append('')
        lines.append("Additional data citation information was found, for "
                     "which no entry is available in the bibtex file:")
        lines.extend('- ' + str(t).replace('\n', ' ')
                     for t in sorted(other_info))

    if lines:
        with open(f'{product_name}_data_citation_info.txt', 'w') as file:
            file.write('\n'.join(lines) + '\n')


def _extract_tags(tags):
    """Extract tags.

    Tags are recorded as a list of strings converted to a string in provenance.
    For example, a single entry in the list `tags` could be the string
    "['acknow_project', 'acknow_author']".
    """
    pattern = re.compile(r'\w+')
    return set(pattern.findall(str(tags)))


def _get_response(url):
    """Return information from CMIP6 Data Citation service in json format."""
    json_data = None
    if url.lower().startswith('https'):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                json_data = response.json()
            else:
                logger.warning('Error in the CMIP6 citation link: %s', url)
        except IOError:
            logger.info('No network connection, '
                        'unable to retrieve CMIP6 citation information')
    return json_data


def _json_to_bibtex(data):
    """Make a bibtex entry from CMIP6 Data Citation json data."""
    url = 'url not found'
    title = data.get('titles', ['title not found'])[0]
    publisher = data.get('publisher', 'publisher not found')
    year = data.get('publicationYear', 'publicationYear not found')
    authors = 'creators not found'
    doi = 'doi not found'

    if 'creators' in data:
        author_list = [
            item.get('creatorName', '') for item in data['creators']
        ]
        authors = ' and '.join(author_list)
        if not authors:
            authors = 'creators not found'

    if 'identifier' in data:
        doi = data['identifier'].get('id', 'doi not found')
        url = f'https://doi.org/{doi}'

    bibtex_entry = textwrap.dedent(f"""
        @misc{{{url},
        \turl = {{{url}}},
        \ttitle = {{{title}}},
        \tpublisher = {{{publisher}}},
        \tyear = {year},
        \tauthor = {{{authors}}},
        \tdoi = {{{doi}}},
        }}
        """).lstrip()
    return bibtex_entry


@lru_cache(maxsize=1024)
def _collect_bibtex_citation(tag):
    """Collect information from bibtex files."""
    bibtex_file = DIAGNOSTICS.references / f'{tag}.bibtex'
    if bibtex_file.is_file():
        entry = bibtex_file.read_text()
    else:
        entry = ''
        logger.warning(
            "The reference file %s does not exist, citation information "
            "incomplete.", bibtex_file)
    return entry


@lru_cache(maxsize=1024)
def _collect_cmip_citation(json_url):
    """Collect information from CMIP6 Data Citation Service."""
    json_data = _get_response(json_url)
    if json_data:
        bibtex_entry = _json_to_bibtex(json_data)
    else:
        bibtex_entry = ''
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
    info_url = f'{CMIP6_URL_STEM}/cmip6?input={url_prefix}'
    return info_url
