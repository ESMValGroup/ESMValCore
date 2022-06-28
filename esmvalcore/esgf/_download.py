"""Module for downloading files from ESGF."""
import concurrent.futures
import contextlib
import datetime
import functools
import hashlib
import itertools
import logging
import os
import random
import shutil
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse

import requests
import yaml
from humanfriendly import format_size, format_timespan

from ._logon import get_credentials
from .facets import DATASET_MAP, FACETS

logger = logging.getLogger(__name__)

TIMEOUT = 5 * 60
"""Timeout (in seconds) for downloads."""

HOSTS_FILE = Path.home() / '.esmvaltool' / 'cache' / 'esgf-hosts.yml'
SIZE = 'size (bytes)'
DURATION = 'duration (s)'
SPEED = 'speed (MB/s)'


class DownloadError(Exception):
    """An error occurred while downloading."""


def compute_speed(size, duration):
    """Compute download speed in MB/s."""
    if duration != 0:
        speed = size / duration / 10**6
    else:
        speed = 0
    return speed


def load_speeds():
    """Load average download speeds from HOSTS_FILE."""
    try:
        content = HOSTS_FILE.read_text()
    except FileNotFoundError:
        content = '{}'
    speeds = yaml.safe_load(content)
    return speeds


def log_speed(url, size, duration):
    """Write the downloaded file size and duration to HOSTS_FILE."""
    speeds = load_speeds()
    host = urlparse(url).hostname
    size += speeds.get(host, {}).get(SIZE, 0)
    duration += speeds.get(host, {}).get(DURATION, 0)
    speed = compute_speed(size, duration)

    speeds[host] = {
        SIZE: size,
        DURATION: round(duration),
        SPEED: round(speed, 1),
        'error': False,
    }
    with atomic_write(HOSTS_FILE) as file:
        yaml.safe_dump(speeds, file)


def log_error(url):
    """Write the hosts that errored to HOSTS_FILE."""
    speeds = load_speeds()
    host = urlparse(url).hostname
    entry = speeds.get(host, {SIZE: 0, DURATION: 0, SPEED: 0})
    entry['error'] = True
    speeds[host] = entry
    with atomic_write(HOSTS_FILE) as file:
        yaml.safe_dump(speeds, file)


@contextlib.contextmanager
def atomic_write(filename):
    """Write a file without the risk of interfering with other processes."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=f"{filename}.") as file:
        tmp_file = file.name
    with open(tmp_file, 'w') as file:
        yield file
    shutil.move(tmp_file, filename)


def get_preferred_hosts():
    """Get a list of preferred hosts.

    The list will be sorted by download speed. Hosts that recentely
    returned an error will be at the end.
    """
    speeds = load_speeds()
    if not speeds:
        return []

    # Compute speeds from size and duration
    for entry in speeds.values():
        entry[SPEED] = compute_speed(entry[SIZE], entry[DURATION])

    # Hosts from which no data has been downloaded yet get median speed; if no
    # host with non-zero entries is found assign a value of 0.0
    speeds_list = [speeds[h][SPEED] for h in speeds if
                   speeds[h][SPEED] != 0.0]
    if not speeds_list:
        median_speed = 0.0
    else:
        median_speed = median(speeds_list)
    for host in speeds:
        if speeds[host][SIZE] == 0:
            speeds[host][SPEED] = median_speed

    # Sort hosts by download speed
    hosts = sorted(speeds, key=lambda h: speeds[h][SPEED], reverse=True)

    # Figure out which hosts recently returned an error
    mtime = HOSTS_FILE.stat().st_mtime
    now = datetime.datetime.now().timestamp()
    age = now - mtime
    if age > 60 * 60:
        # Ignore errors older than an hour
        errored = []
    else:
        errored = [h for h in speeds if speeds[h]['error']]

    # Move hosts with an error to the end of the list
    for host in errored:
        if host in hosts:
            hosts.pop(hosts.index(host))
            hosts.append(host)

    return hosts


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
    preferred_hosts = get_preferred_hosts()
    for host in preferred_hosts:
        if host in hosts:
            # Move host and corresponding URL to the beginning of the list,
            # but after any unknown hosts so these will get used too.
            idx = hosts.index(host)
            hosts.append(hosts.pop(idx))
            urls.append(urls.pop(idx))

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

    def __init__(self, results):
        results = list(results)
        self.name = str(Path(results[0].filename).with_suffix('.nc'))
        self.size = results[0].size
        self.dataset = self._get_dataset_id(results)
        self.urls = []
        self._checksums = []
        for result in results:
            self.urls.append(result.download_url)
            self._checksums.append((result.checksum_type, result.checksum))

    @classmethod
    def _from_results(cls, results, facets):
        """Return a list of files from a pyesgf.search.results.ResultSet."""

        def same_file(result):
            # Remove the hostname from the dataset_id
            dataset = result.json['dataset_id'].split('|')[0]
            # Ignore the extension (some files are called .nc_0, .nc_1)
            filename = Path(result.filename).stem
            # Ignore case
            return (dataset.lower(), filename.lower())

        files = []
        results = sorted(results, key=same_file)
        for _, file_results in itertools.groupby(results, key=same_file):
            file = cls(file_results)
            # Filter out files containing the wrong variable, e.g. for
            # cmip5.output1.ICHEC.EC-EARTH.historical
            # .mon.atmos.Amon.r1i1p1.v20121115
            variable = file.name.split('_')[0]
            if 'variable' in facets and facets['variable'] == variable:
                files.append(file)
            else:
                logger.debug(
                    "Ignoring file(s) %s containing wrong variable '%s' in"
                    " found in search for variable '%s'", file.urls, variable,
                    facets['variable'])

        return files

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
        dest_folder: Path
            The destination folder.

        Returns
        -------
        Path
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
        dest_folder: Path
            The destination folder.

        Raises
        ------
        DownloadError:
            Raised if downloading the file failed.

        Returns
        -------
        Path
            The path where the file will be located after download.
        """
        local_file = self.local_file(dest_folder)
        if local_file.exists():
            logger.debug("Skipping download of existing file %s", local_file)
            return local_file

        os.makedirs(local_file.parent, exist_ok=True)

        errors = {}
        for url in sort_hosts(self.urls):
            try:
                self._download(local_file, url)
            except (DownloadError,
                    requests.exceptions.RequestException) as error:
                logger.debug("Not able to download %s. Error message: %s", url,
                             error)
                errors[url] = error
                log_error(url)
            else:
                break

        if not local_file.exists():
            raise DownloadError(
                f"Failed to download file {local_file}, errors:"
                "\n" + "\n".join(f"{url}: {errors[url]}" for url in errors))

        return local_file

    @staticmethod
    def _tmp_local_file(local_file):
        """Return the path to a temporary local file for downloading to."""
        with NamedTemporaryFile(prefix=f"{local_file}.") as tmp_file:
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

        logger.debug("Downloading %s to %s", url, tmp_file)
        start_time = datetime.datetime.now()
        response = requests.get(url,
                                stream=True,
                                timeout=TIMEOUT,
                                cert=get_credentials())
        response.raise_for_status()
        with tmp_file.open("wb") as file:
            for chunk in response.iter_content(chunk_size=None):
                if hasher is not None:
                    hasher.update(chunk)
                file.write(chunk)

        duration = datetime.datetime.now() - start_time

        if hasher is None:
            logger.warning(
                "No checksum available, unable to check data"
                " integrity for %s, ", url)
        else:
            local_checksum = hasher.hexdigest()
            if local_checksum != checksum:
                raise DownloadError(
                    f"Wrong {checksum_type} checksum for file {tmp_file},"
                    f" downloaded from {url}: expected {checksum}, but got"
                    f" {local_checksum}. Try downloading the file again.")

        shutil.move(tmp_file, local_file)
        log_speed(url, self.size, duration.total_seconds())
        logger.info("Downloaded %s (%s) in %s (%s/s) from %s", local_file,
                    format_size(self.size),
                    format_timespan(duration.total_seconds()),
                    format_size(self.size / duration.total_seconds()),
                    urlparse(url).hostname)


def get_download_message(files):
    """Create a log message describing what will be downloaded."""
    total_size = 0
    lines = []
    for file in files:
        total_size += file.size
        lines.append(f"{format_size(file.size)}" "\t" f"{file}")

    lines.insert(0, "Will download the following files:")
    lines.insert(0, f"Will download {format_size(total_size)}")
    lines.append(f"Downloading {format_size(total_size)}..")
    return "\n".join(lines)


def download(files, dest_folder, n_jobs=4):
    """Download multiple ESGFFiles in parallel.

    Arguments
    ---------
    files: list of :obj:`ESGFFile`
        The files to download.
    dest_folder: Path
        The destination folder.
    n_jobs: int
        The number of files to download in parallel.

    Raises
    ------
    DownloadError:
        Raised if one or more files failed to download.
    """
    if not files:
        logger.info("All required data is available locally,"
                    " not downloading anything.")
        return

    files = sorted(files)
    logger.info(get_download_message(files))

    def _download(file: ESGFFile):
        """Download file to dest_folder."""
        file.download(dest_folder)

    total_size = 0
    start_time = datetime.datetime.now()

    errors = []
    random.shuffle(files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        future_to_file = {
            executor.submit(_download, file): file
            for file in files
        }

        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                future.result()
            except DownloadError as error:
                logger.error("Failed to download %s, error message %s", file,
                             error)
                errors.append(error)
            else:
                total_size += file.size

    duration = datetime.datetime.now() - start_time
    logger.info(
        "Downloaded %s in %s (%s/s)",
        format_size(total_size),
        format_timespan(duration.total_seconds()),
        format_size(total_size / duration.total_seconds()),
    )

    if errors:
        msg = ("Failed to download the following files:\n" +
               "\n".join(sorted(str(error) for error in errors)))
        raise DownloadError(msg)

    logger.info("Successfully downloaded all requested files.")
