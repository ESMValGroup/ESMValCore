"""Functions for downloading climate data files."""
import asyncio
import datetime
import hashlib
import itertools
import logging
import os
import pprint
import shutil
import tempfile
import urllib
from pathlib import Path

import aiohttp
import pyesgf.logon
import pyesgf.search
import requests

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
"""Mapping between the recipe and ESGF facet names."""

DATASET_MAP = {
    'CMIP3': {},
    'CMIP5': {
        'ACCESS1-0': 'ACCESS1.0',
        'ACCESS1-3': 'ACCESS1.3',
        'bcc-csm1-1': 'BCC-CSM1.1',
        'bcc-csm1-1-m': 'BCC-CSM1.1(m)',
        'CESM1-BGC': 'CESM1(BGC)',
        'CESM1-CAM5': 'CESM1(CAM5)',
        'CESM1-CAM5-1-FV2': 'CESM1(CAM5.1,FV2)',
        'CESM1-FASTCHEM': 'CESM1(FASTCHEM)',
        'CESM1-WACCM': 'CESM1(WACCM)',
        'CSIRO-Mk3-6-0': 'CSIRO-Mk3.6.0',
        'fio-esm': 'FIO-ESM',
        'GFDL-CM2p1': 'GFDL-CM2.1',
        'inmcm4': 'INM-CM4',
        'MRI-AGCM3-2H': 'MRI-AGCM3.2H',
        'MRI-AGCM3-2S': 'MRI-AGCM3.2S'
    },
    'CMIP6': {},
    'CORDEX': {},
    'obs4MIPs': {},
}
"""Cache for the mapping between recipe/filesystem and ESGF dataset names."""


def get_esgf_facets(variable):
    """Translate variable to facets for searching on ESGF."""
    project = variable.get('project', '')
    if project not in FACETS:
        logger.info(
            "Unable to download from ESGF, because project '%s' is not on it.",
            project)
        return None

    facets = {'project': project}
    if project == 'CMIP5':
        facets['product'] = 'output1'
    for our_name, esgf_name in FACETS[project].items():
        if our_name in variable:
            value = variable[our_name]
            if our_name == 'dataset':
                # Replace dataset name by ESGF name for dataset
                value = DATASET_MAP[project].get(value, value)
            facets[esgf_name] = value

    return facets


def sort_hosts(hosts, preferred_hosts):
    """Select a list of suitable hosts from a list of hosts.

    Parameters
    ----------
    hosts : :obj:`list` of :obj:`str`
        List of all available hosts.
    preferred_hosts : :obj:`list` of :obj:`str`
        List of preferred hosts.

    Returns
    -------
    :obj:`list` of :obj:`str`
        The list of hosts, with preferred hosts first.
    """
    hosts = list(hosts)
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
    def name(dataset_name):
        """Return the name of the dataset without the version."""
        return dataset_name.rsplit('.', 1)[0]

    latest_versions = {}
    for _, versions in itertools.groupby(sorted(datasets), key=name):
        latest = list(versions)[-1]
        latest_versions[latest] = datasets[latest]

    return latest_versions


def merge_datasets(datasets):
    """Merge datasets that only differ in capitalization.

    Example of two datasets that will be merged:

    Dataset available on hosts 'esgf-data1.ceda.ac.uk', 'esgf.nci.org.au',
    'esgf2.dkrz.de':
    cmip5.output1.FIO.FIO-ESM.historical.mon.atmos.Amon.r1i1p1.v20121010
    Dataset available on host 'aims3.llnl.gov':
    cmip5.output1.FIO.fio-esm.historical.mon.atmos.Amon.r1i1p1.v20121010
    """
    merged = {}
    names = sorted(datasets, key=str.lower)
    for _, aliases in itertools.groupby(names, key=str.lower):
        name = next(aliases)
        files = {file.name: file for file in datasets[name]}
        for alias in aliases:
            logger.info("Combining dataset %s with %s", alias, name)
            for file in datasets[alias]:
                if file.name not in files:
                    files[file.name] = file
                else:
                    for url in file.urls:
                        if url not in files[file.name].urls:
                            files[file.name].urls.append(url)
        merged[name] = list(files.values())

    return merged


def create_dataset_map(connection=None):
    """Create a dataset map from recipe datasets to ESGF facet values."""
    dataset_map = {}

    if connection is None:
        connection = get_connection()

    indices = {
        'CMIP3': 2,
        'CMIP5': 3,
        'CMIP6': 3,
        'CORDEX': 7,
        'obs4MIPs': 2,
    }

    for project in FACETS:
        dataset_map[project] = {}
        ctx = connection.new_context(project=project, latest=True)

        dataset_key = FACETS[project]['dataset']
        available_datasets = sorted(ctx.facet_counts[dataset_key])
        print(f"The following datasets are available for project {project}:")
        for dataset in available_datasets:
            print(dataset)

        # Figure out the ESGF name of the requested dataset
        n_available = len(available_datasets)
        for i, dataset in enumerate(available_datasets, 1):
            print(f"Looking for dataset name of facet name"
                  f" {dataset} ({i} of {n_available})")
            query = {dataset_key: dataset}
            dataset_result = next(iter(ctx.search(batch_size=1, **query)))
            print(f"Dataset id: {dataset_result.dataset_id}")
            dataset_id = dataset_result.dataset_id
            if dataset in dataset_id:
                print(f"Dataset facet is identical to "
                      f"dataset name for '{dataset}'")
            else:
                idx = indices[project]
                dataset_alias = dataset_id.split('.')[idx]
                print(f"Found dataset name '{dataset_alias}'"
                      f" for facet '{dataset}',")
                dataset_map[project][dataset_alias] = dataset
        print(dataset_map)

    return dataset_map


def search(connection, preferred_hosts, facets):
    """Search for files on ESGF.

    Parameters
    ----------
    connection: pyesgf.search.SearchConnection
        Search connection
    preferred_hosts: :obj:`list` of :obj:`str`
        List of preferred hosts.
    facets: :obj:`dict` of :obj:`str`
        Facets to constrain the search.

    Returns
    -------
    :obj:`list` of :obj:`str`
        A dict with dataset names as keys and a list of filenames
        (OPeNDAP URLs) as values.
    """
    logger.info("Searching on ESGF using facets=%s", facets)
    ctx = connection.new_context(**facets, latest=True)
    logger.info(
        "Found %s datasets (any version, including copies)"
        " with facets=%s", ctx.hit_count, facets)

    # Obtain dataset results
    datasets = {}
    for dataset_result in ctx.search(**facets, latest=True):
        dataset_name, host = dataset_result.dataset_id.split('|')
        if dataset_name not in datasets:
            datasets[dataset_name] = {}
        datasets[dataset_name][host] = dataset_result

    logger.info("Found the following datasets matching facets %s:\n%s", facets,
                '\n'.join(datasets))

    if not datasets:
        # Figure out the ESGF name of the requested dataset here?
        project = facets['project']
        dataset_key = FACETS[project]['dataset']
        available_datasets = sorted(ctx.facet_counts[dataset_key])
        print(f"Available datasets for project {project}:"
              "\n" + "\n".join(available_datasets))
        raise ValueError(f"Dataset {facets[dataset_key]} not found")

    datasets = select_latest_versions(datasets)

    # Select host and find files on host
    files = {}
    for dataset_name in sorted(datasets):
        copies = datasets[dataset_name]
        logger.info(
            "Searching for files for dataset %s, available on hosts %s",
            dataset_name,
            sorted(copies.keys()),
        )
        hosts = sort_hosts(copies.keys(), preferred_hosts)

        dataset_files = {}
        for host in hosts:
            dataset_result = copies[host]
            file_result = dataset_result.file_context().search(
                variable=facets['variable'])
            for file in file_result:
                if file.filename in dataset_files:
                    dataset_files[file.filename].urls.append(file.download_url)
                else:
                    dataset_files[file.filename] = ESGFFile(
                        urls=[file.download_url],
                        dataset=dataset_name,
                        name=file.filename,
                        size=file.size,
                        checksum=file.checksum,
                        checksum_type=file.checksum_type,
                    )
        files[dataset_name] = list(dataset_files.values())

    files = merge_datasets(files)

    return files


def select_by_time(files, start_year, end_year):
    """Select files containing data between start_year and end_year."""
    filedict = {file.name: file for file in files}
    files = select_files(filedict, start_year, end_year)

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

    return [filedict[f] for f in files]


_RANGE_HOSTS = {}


def host_accepts_range(url):
    """Test whether a host accepts the Range parameter."""
    # https://stackoverflow.com/questions/720419/how-can-i-find-out-whether-a-server-supports-the-range-header
    hostname = urllib.parse.urlparse(url).hostname
    if hostname not in _RANGE_HOSTS:
        try:
            response = requests.get(
                url,
                timeout=5,
                headers={'Range': 'bytes=0-0'},
            )
        except requests.exceptions.RequestException:
            _RANGE_HOSTS[hostname] = False
        else:
            _RANGE_HOSTS[hostname] = (
                response.status_code == 206
                # http://esgf-data1.ceda.ac.uk does return status code 206, but
                # it does not support ranges and will redirect to a html page
                and response.headers['Content-Type'] == 'application/x-netcdf')

    return _RANGE_HOSTS[hostname]


class Queue(asyncio.Queue):
    """Queue that keeps track of the number of unfinished tasks."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unfinished_tasks = 0

    def put_nowait(self, *args, **kwargs):
        """Put an item into the queue without blocking."""
        super().put_nowait(*args, **kwargs)
        self.unfinished_tasks += 1

    def task_done(self, *args, **kwargs):
        """Indicate that a formerly enqueued task is complete."""
        super().task_done(*args, **kwargs)
        self.unfinished_tasks -= 1


class ESGFFile:
    """File on the ESGF.

    Attributes
    ----------
    urls: :class:`list` of :class:`str`
        The URLs where the file can be downloaded.
    dataset: str
        The name of the dataset that the file is part of.
    name: str
        The name of the file.
    size: int
        The size of the file in bytes.
    checksum: str
        The checksum of the file.
    checksum_type: str
        The checksum type of the file (e.g. MD5).
    """
    def __init__(self, urls, dataset, name, size, checksum, checksum_type):
        self.urls = urls
        self.dataset = dataset
        self.name = name
        self.size = size
        self.checksum = checksum
        self.checksum_type = checksum_type

    def __repr__(self):
        """Represent the file as a string."""
        return f"ESGFFile:{self.dataset.replace('.', '/')}/{self.name}"

    def local_file(self, dest_folder):
        """Return the path to the local file after download."""
        # TODO: this fails for obs4MIPs datasets with a . in their name
        return Path(
            dest_folder,
            *self.dataset.split('.'),
            self.name,
        ).absolute()

    @staticmethod
    def _tmp_local_file(local_file):
        """Return the path to a temporary local file for downloading to."""
        with tempfile.NamedTemporaryFile(prefix=f"{local_file}.") as tmp_file:
            return Path(tmp_file.name)

    def download(self, dest_folder):
        """Download file."""
        local_file = self.local_file(dest_folder)
        if local_file.exists():
            logger.info("Skipping download of existing file %s", local_file)
            return str(local_file)

        os.makedirs(local_file.parent, exist_ok=True)
        start_time = datetime.datetime.now()

        range_urls = [url for url in self.urls if host_accepts_range(url)]
        if len(range_urls) > 1:
            self._download_multiple_urls(local_file, range_urls)
        else:
            for url in self.urls:
                try:
                    self._download_single_url(local_file, url)
                except requests.exceptions.RequestException:
                    logger.info("Failed to download from %s", url)
                else:
                    break
        duration = datetime.datetime.now() - start_time
        logger.info("Downloaded %s (%.0f MB) in %s (%.1f MB/s)", local_file,
                    self.size / 2**20, duration,
                    self.size / 2**20 / duration.total_seconds())
        return str(local_file)

    def _download_single_url(self, local_file, url):
        """Download file from a single url."""
        hasher = hashlib.new(self.checksum_type)

        tmp_file = self._tmp_local_file(local_file)

        logger.info("Downloading %s to %s", url, tmp_file)
        response = requests.get(url, timeout=5)
        with tmp_file.open("wb") as file:
            chunk_size = 1 << 20  # 1 MB
            for chunk in response.iter_content(chunk_size=chunk_size):
                hasher.update(chunk)
                file.write(chunk)

        checksum = hasher.hexdigest()

        self._finalize_download(tmp_file, local_file, checksum)

    def _download_multiple_urls(self, local_file, urls):
        """Download file from multiple urls."""
        asyncio.run(self._download_from_multiple_urls(local_file, urls))

    async def _download_from_multiple_urls(self, local_file, urls):
        """Download a file from multiple urls."""
        queue = Queue()
        chunk_size = 10 * 2**20  # 10 MB
        for start in range(0, self.size, chunk_size):
            end = min(start + chunk_size, self.size - 1)
            queue.put_nowait([start, end])

        tmp_file = self._tmp_local_file(local_file)
        with tmp_file.open('wb') as file:
            workers = [
                asyncio.create_task(self._downloader(url, file, queue))
                for url in urls
            ]

            n_chunks = queue.qsize()
            while queue.unfinished_tasks:
                await asyncio.sleep(1)
                print(f"Queued chunks: {queue.qsize()}, running"
                      f" {queue.unfinished_tasks - queue.qsize()},"
                      f" total {n_chunks}, ")
                online_workers = [w for w in workers if not w.cancelled()]
                if not online_workers:
                    errors = await asyncio.gather(*workers,
                                                  return_exceptions=True)
                    for error, url in zip(errors, urls):
                        if error:
                            logger.warning(
                                "An exception occurred while downloading"
                                " from %s:\n%s", url, error)
                    raise IOError(f"Unable to download {self.name}"
                                  f" from {urls}: no hosts are online.")

            # Clean up workers
            await queue.join()
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        self._finalize_download(tmp_file, local_file)

    async def _downloader(self, url, tmp_file, queue):
        """Start a worker that downloads and saves chunks from a single URL."""
        hostname = urllib.parse.urlparse(url).hostname
        # 12 hours should be enough to download a single file.
        timeout = aiohttp.ClientTimeout(total=12 * 60 * 60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                print(f"Requesting work for host {hostname}")
                chunk = await queue.get()
                start, end = chunk
                headers = {'Range': f'bytes={start}-{end}'}
                print(f"Start download {start}-{end} of {url}")
                try:
                    async with session.get(url, timeout=60,
                                           headers=headers) as response:
                        content = await response.content.read()
                except (aiohttp.ClientError,
                        asyncio.exceptions.TimeoutError) as exc:
                    print(f"Not able to download from host {hostname}")
                    await queue.put(chunk)
                    raise asyncio.CancelledError from exc

                tmp_file.seek(start)
                tmp_file.write(content)
                print(f"Done: chunk {start} of {url}")
                queue.task_done()

    def _finalize_download(self, tmp_file, local_file, checksum=None):
        """Move file to correct location if checksum is correct."""
        if checksum is None:
            hasher = hashlib.new(self.checksum_type)
            with tmp_file.open('rb') as file:
                hasher.update(file.read())
            checksum = hasher.hexdigest()

        if checksum != self.checksum:
            raise ValueError(
                f"Wrong {self.checksum_type} checksum for file {tmp_file},"
                f" expected: {self.checksum}, got {checksum}.")
        shutil.move(tmp_file, local_file)


def get_connection():
    """Connect to ESGF."""
    cfg = _load_esgf_pyclient_config()

    manager = pyesgf.logon.LogonManager()
    if not manager.is_logged_on():
        manager.logon(**cfg['logon'])
        logger.info("Logged %s", "on" if manager.is_logged_on() else "off")

    connection = pyesgf.search.SearchConnection(**cfg["search_connection"])
    return connection


def esgf_search(variable):
    """Search files using esgf-pyclient."""
    cfg = _load_esgf_pyclient_config()
    connection = get_connection()

    facets = get_esgf_facets(variable)
    if facets is None:
        return []

    datasets = search(
        connection,
        cfg['preferred_hosts'],
        facets,
    )
    logger.info("Found files %s", pprint.pformat(datasets))
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
        logger.info("Selected files:\n%s", '\n'.join(str(f) for f in files))

    return files


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
