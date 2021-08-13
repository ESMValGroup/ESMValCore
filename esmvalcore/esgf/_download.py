"""Module for downloading files from ESGF."""
import concurrent.futures
import datetime
import functools
import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pyesgf.search.results
import requests

from .._config._esgf_pyclient import get_esgf_config
from ._logon import get_credentials
from .facets import DATASET_MAP, FACETS

logger = logging.getLogger(__name__)

TIMEOUT = 5 * 60
"""Timeout (in seconds) for downloads."""


def sort_hosts(urls):
    """Sort a list of URLs by preferred hosts.

    Parameters
    ----------
    urls : :obj:`list` of :obj:`str`
        List of all available URLs.

    Returns
    -------
    :obj:`list` of :obj:`str`
        The list of URLs, with URLs from a preferred hosts first.
    """
    urls = list(urls)
    hosts = [urlparse(url).hostname for url in urls]
    cfg = get_esgf_config()
    preferred_hosts = cfg.get('preferred_hosts', [])
    for host in preferred_hosts[::-1]:
        if host in hosts:
            # Move host and corresponding URL to the beginning of the list
            idx = hosts.index(host)
            hosts.insert(0, hosts.pop(idx))
            urls.insert(0, urls.pop(idx))

    return urls


@functools.total_ordering
class ESGFFile:
    """File on the ESGF.

    This is the object returned by the function :func:`esmvalcore.esgf.search`.

    Attributes
    ----------
    urls : :class:`list` of :class:`str`
        The URLs where the file can be downloaded.
    dataset : str
        The name of the dataset that the file is part of.
    name : str
        The name of the file.
    size : int
        The size of the file in bytes.
    """
    def __init__(self, results: list[pyesgf.search.results.FileResult]):
        self.name = results[0].filename
        self.size = results[0].size
        self.dataset = self._get_dataset_id(results)
        self.urls = []
        self._checksums = []
        for result in results:
            self.urls.append(result.download_url)
            self._checksums.append((result.checksum_type, result.checksum))

    @staticmethod
    def _get_dataset_id(results):
        """Simplify dataset_id so it is always composed of the same facets."""
        # Pick the first dataset_id if there are differences in case
        dataset_id = sorted(r.json['dataset_id'].split('|')[0]
                            for r in results)[0]

        project = results[0].json['project'][0]
        if project != 'obs4MIPs':
            return dataset_id

        # Simplify the obs4MIPs dataset_id so it contains only facets that are
        # present for all datasets.
        version = dataset_id.rsplit('.', 1)[1]
        dataset_key = FACETS[project]['dataset']
        dataset_name = results[0].json[dataset_key][0]
        dataset_name = DATASET_MAP[project].get(dataset_name, dataset_name)
        return f"{project}.{dataset_name}.{version}"

    def __repr__(self):
        """Represent the file as a string."""
        hosts = [urlparse(u).hostname for u in self.urls]
        return (f"ESGFFile:{self.dataset.replace('.', '/')}/{self.name}"
                f" on hosts {hosts}")

    def __eq__(self, other):
        """Compare `self` to `other`."""
        return (self.dataset, self.name) == (other.dataset, other.name)

    def __lt__(self, other):
        """Compare `self` to `other`."""
        return (self.dataset, self.name) < (other.dataset, other.name)

    def __hash__(self):
        """Compute a unique hash value."""
        return hash((self.dataset, self.name))

    def local_file(self, dest_folder):
        """Return the path to the local file after download.

        Arguments
        ---------
        dest_folder:
            The destination folder.

        Returns
        -------
        pathlib.Path
            The path where the file will be located after download.
        """
        return Path(
            dest_folder,
            *self.dataset.split('.'),
            self.name,
        ).absolute()

    def download(self, dest_folder):
        """Download the file.

        Arguments
        ---------
        dest_folder:
            The destination folder.

        Returns
        -------
        pathlib.Path
            The path where the file will be located after download.
        """
        local_file = self.local_file(dest_folder)
        if local_file.exists():
            logger.info("Skipping download of existing file %s", local_file)
            return local_file

        os.makedirs(local_file.parent, exist_ok=True)
        start_time = datetime.datetime.now()

        for url in sort_hosts(self.urls):
            try:
                self._download(local_file, url)
            except requests.exceptions.RequestException as exc:
                logger.info("Not able to download %s. Error message: %s", url,
                            exc)
            else:
                break

        if not local_file.exists():
            raise IOError(
                f"Failed to download file {local_file} from {self.urls}")

        duration = datetime.datetime.now() - start_time
        logger.info("Downloaded %s (%.0f MB) in %s (%.1f MB/s)", local_file,
                    self.size / 2**20, duration,
                    self.size / 2**20 / duration.total_seconds())
        return local_file

    @staticmethod
    def _tmp_local_file(local_file):
        """Return the path to a temporary local file for downloading to."""
        with tempfile.NamedTemporaryFile(prefix=f"{local_file}.") as tmp_file:
            return Path(tmp_file.name)

    def _download(self, local_file, url):
        """Download file from a single url."""
        idx = self.urls.index(url)
        checksum_type, checksum = self._checksums[idx]
        if checksum_type is None:
            hasher = None
        else:
            hasher = hashlib.new(checksum_type)

        tmp_file = self._tmp_local_file(local_file)

        logger.info("Downloading %s to %s", url, tmp_file)
        response = requests.get(url, timeout=TIMEOUT, cert=get_credentials())
        response.raise_for_status()
        with tmp_file.open("wb") as file:
            chunk_size = 1 << 20  # 1 MB
            for chunk in response.iter_content(chunk_size=chunk_size):
                if hasher is not None:
                    hasher.update(chunk)
                file.write(chunk)

        if hasher is None:
            logger.warning(
                "No checksum available, unable to check data"
                " integrity for %s, ", url)
        else:
            local_checksum = hasher.hexdigest()
            if local_checksum != checksum:
                raise ValueError(
                    f"Wrong {checksum_type} checksum for file {tmp_file},"
                    f" downloaded from {url}: expected {checksum}, but got"
                    f" {local_checksum}. Try downloading the file again.")

        shutil.move(tmp_file, local_file)


def get_download_message(files):
    """Create a log message describing what will be downloaded."""
    megabyte = 2**20
    gigabyte = 2**30
    total_size = 0
    lines = []
    for file in sorted(files):
        total_size += file.size
        lines.append(f"{file.size / megabyte:.0f} MB" "\t" f"{file}")
    if total_size:
        lines.insert(0, "Will download the following files:")
    lines.insert(0, f"Will download {total_size / gigabyte:.1f} GB")
    return "\n".join(lines)


def download(files, dest_folder, n_jobs=4):
    """Download multiple ESGFFiles in parallel."""
    logger.info(get_download_message(files))

    def _download(file: ESGFFile):
        """Download file to dest_folder."""
        file.download(dest_folder)

    total_size = sum(file.size for file in files)
    start_time = datetime.datetime.now()

    errored = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:

        future_to_file = {
            executor.submit(_download, file): file
            for file in files
        }

        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                future.result()
            except Exception as exc:
                logger.error("Failed to download %s, error message %s", file,
                             str(exc))
                errored.append(file)

    duration = datetime.datetime.now() - start_time
    logger.info("Downloaded %.0f GB in %s (%.1f MB/s)", total_size / 2**30,
                duration, total_size / 2**20 / duration.total_seconds())

    if errored:
        logger.error("Failed to download the following files:\n%s",
                     '\n'.join(str(f) for f in errored))
        raise IOError("Download of some requested files failed.")
    else:
        logger.info("Successfully downloaded all requested files.")
