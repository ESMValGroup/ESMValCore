"""Functions for downloading climate data files."""
import logging
import os
import pprint
import shutil
import subprocess
import tempfile
from itertools import groupby
from pathlib import Path

import pyesgf.logon
import pyesgf.search

from .._config._esgf_pyclient import _load_esgf_pyclient_config
from .._data_finder import get_start_end_year, select_files

logger = logging.getLogger(__name__)

FACETS = {
    'CMIP3': {
        'dataset': 'model',
        'ensemble': 'ensemble',
        'exp': 'experiment',
        'frequency': 'time_frequency',
        'short_name': 'variable',
    },
    'CMIP5': {
        'dataset': 'model',
        'ensemble': 'ensemble',
        'exp': 'experiment',
        'mip': 'cmor_table',
        'short_name': 'variable',
    },
    'CMIP6': {
        'activity': 'activity_id',
        'dataset': 'source_id',
        'ensemble': 'variant_label',
        'exp': 'experiment_id',
        'grid': 'grid_label',
        'mip': 'table_id',
        'short_name': 'variable',
    },
    'CORDEX': {
        'dataset': 'rcm_name',
        'driver': 'driving_model',
        'domain': 'domain',
        'ensemble': 'ensemble',
        'exp': 'experiment',
        'frequency': 'time_frequency',
        'short_name': 'variable',
    },
    'obs4MIPs': {
        'dataset': 'source_id',
        'frequency': 'time_frequency',
        'short_name': 'variable',
    }
}


def _get_esgf_facets(variable):
    """Translate variable to facets for searching on ESGF."""
    project = variable.get('project', '')
    if project not in FACETS:
        logger.debug(
            "Unable to download from ESGF, because project '%s' is not on it.",
            project)
        return None

    facets = {'project': project}
    if project == 'CMIP5':
        facets['product'] = 'output1'
    for our_name, esgf_name in FACETS[project].items():
        if our_name in variable:
            facets[esgf_name] = variable[our_name]

    return facets


def select_hosts(hosts, preferred_hosts, ignore_hosts):
    """Select a list of suitable hosts from a list of hosts.

    Parameters
    ----------
    hosts : :obj:`list` of :obj:`str`
        List of all available hosts.
    preferred_hosts : :obj:`list` of :obj:`str`
        List of preferred hosts.
    ignore_hosts : :obj:`list` of :obj:`str`
        List of hosts to ignore.

    Returns
    -------
    :obj:`list` of :obj:`str`
        The name of the most suitable host or None of all available hosts are
        in `ignore_hosts`.

    Notes
    -----
        Not sure if this is reliable: sometimes no files are found on the
        selected host.
    """
    hosts = [h for h in hosts if h not in ignore_hosts]

    for host in preferred_hosts[::-1]:
        if host in hosts:
            # Move host to the beginning of the list
            hosts.insert(0, hosts.pop(hosts.index(host)))

    return hosts


def select_latest_versions(datasets: dict) -> dict:
    """Return a dict with only the latest version of each dataset.

    Parameters
    ----------
    datasets : dict
        A dict with dataset objects

    Returns
    -------
    most_recent_datasets : dict
        A dict containing only the most recent version of each dataset object,
        in case multiple versions have been passed.
    """
    keys = (key.rsplit('.', 1) for key in datasets)
    keys = sorted(keys)
    grouped = groupby(keys, key=lambda key: key[0])

    most_recent_keys = (list(versions)[-1] for group, versions in grouped)
    most_recent_datasets = {}

    for name, version in most_recent_keys:
        key = f'{name}.{version}'
        most_recent_datasets[key] = datasets[key]

    return most_recent_datasets


def search(connection, preferred_hosts, ignore_hosts, facets):
    """Search for files on ESGF.

    Parameters
    ----------
    connection: pyesgf.search.SearchConnection
        Search connection
    preferred_hosts: :obj:`list` of :obj:`str`
        List of preferred hosts.
    ignore_hosts: :obj:`list` of :obj:`str`
        List of hosts to ignore.
    facets: :obj:`dict` of :obj:`str`
        Facets to constrain the search.

    Returns
    -------
    :obj:`list` of :obj:`str`
        A dict with dataset names as keys and a list of filenames
        (OPeNDAP URLs) as values.
    """
    logger.debug("Searching on ESGF using facets=%s", facets)
    ctx = connection.new_context(**facets, latest=True)
    logger.debug(
        "Found %s datasets with facets=%s (any version,"
        " including copies)", facets, ctx.hit_count)

    # Find available datasets
    datasets = {}
    for dataset in ctx.search():
        dataset_name, host = dataset.dataset_id.split('|')
        if dataset_name not in datasets:
            datasets[dataset_name] = {}
        datasets[dataset_name][host] = dataset

    datasets = select_latest_versions(datasets)

    # Select host and find files on host
    files = {}
    for dataset_name in sorted(datasets):
        copies = datasets[dataset_name]
        logger.debug(
            "Searching for files for dataset %s, available on hosts %s",
            dataset_name,
            sorted(copies.keys()),
        )
        hosts = select_hosts(copies.keys(), preferred_hosts, ignore_hosts)
        if not hosts:
            logger.warning("All hosts that have datasets %s are ignored.",
                           dataset_name)
            continue
        for host in hosts:
            dataset = copies[host]
            dataset_result = dataset.file_context().search(
                variable=facets['variable'])
            files[dataset_name] = [f.download_url for f in dataset_result]
            if files[dataset_name]:
                break

    return files


def select_by_time(files, start_year, end_year):

    files = select_files(files, start_year, end_year)

    # filter partially overlapping files
    intervals = {get_start_end_year(name): name for name in files}
    files = []
    for (start, end), filename in intervals.items():
        for _start, _end in intervals:
            if start == _start and end == _end:
                continue
            if start >= _start and end <= _end:
                break
        else:
            files.append(filename)

    return files


class ESGFFile:
    """ESGF filename.

    Attributes
    ----------
    url: str
        The URL of the dataset on ESGF.
    dataset: str
        The name of the dataset.

    """
    def __init__(self, url, dataset):
        self.url = url
        self.dataset = dataset

    def __str__(self):
        return self.url

    def local_file(self, dest_folder):
        """The path to the local file after download."""
        return Path(dest_folder, *self.dataset.split('.')) / Path(
            self.url).name

    def download(self, dest_folder):
        """Download file using wget."""
        local_file = self.local_file(dest_folder)

        if not local_file.exists():
            os.makedirs(dest_folder, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                    prefix=f"{local_file}.") as tmp_file:
                tmp_filename = tmp_file.name
            logger.debug("Downloading %s to %s", self.url, tmp_filename)
            cmd = ['wget', f'--output-document={tmp_filename}', self.url]
            logger.debug("Running: %s", cmd)
            subprocess.check_call(cmd)
            shutil.move(tmp_filename, local_file)
            logger.info("Downloaded %s", local_file)

        return str(local_file)


def esgf_search(variable):
    """Search files using esgf-pyclient."""
    cfg = _load_esgf_pyclient_config()

    manager = pyesgf.logon.LogonManager()
    if not manager.is_logged_on():
        manager.logon(**cfg['logon'])
        logger.debug("Logged", "on" if manager.is_logged_on() else "off")

    connection = pyesgf.search.SearchConnection(**cfg["search_connection"])

    facets = _get_esgf_facets(variable)
    if facets is None:
        return []

    datasets = search(
        connection,
        cfg['preferred_hosts'],
        cfg['ignore_hosts'],
        facets,
    )
    logger.debug("Found files %s", pprint.pformat(datasets))
    if not datasets:
        return []

    if len(datasets) > 1:
        raise ValueError("Expected to find a single dataset, found {}".format(
            pprint.pformat(datasets)))

    dataset_name = next(iter(datasets))
    files = datasets[dataset_name]

    if variable.get('frequency', '') != 'fx':
        files = select_by_time(files, variable['start_year'],
                               variable['end_year'])
        logger.debug("Selected files:\n%s", '\n'.join(files))

    filelist = []
    for filename in files:
        file = ESGFFile(filename, dataset_name)
        filelist.append(file)

    return filelist


def download(files, dest_folder):
    """Download files that are not available locally.

    Parameters
    ----------
    files: :obj:`list` of :obj:`str` or :obj:`ESGFFile`
        List of tuples: (dataset, url).
    dest_folder: str
        Directory where downloaded files will be stored.
    """
    local_files = []
    for file in files:
        if isinstance(file, ESGFFile):
            local_file = file.download(dest_folder)
            local_files.append(local_file)
        else:
            local_files.append(file)

    return local_files
