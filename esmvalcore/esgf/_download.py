"""Module for downloading files from ESGF."""
# TODO: fix obs4MIPs issue with the path for dataset names containing a period
import asyncio
import datetime
import hashlib
import logging
import os
import shutil
import tempfile
import urllib
from pathlib import Path

import aiohttp
import requests

from ._logon import get_credentials

logger = logging.getLogger(__name__)

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
            # host is offline
            _RANGE_HOSTS[hostname] = False
        else:
            _RANGE_HOSTS[hostname] = (
                response.status_code == 206
                # esgf-data1.ceda.ac.uk does return status code 206, but it
                # does not support ranges and will redirect to a html page
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
    def __init__(self, urls, dataset, name, size, checksum, checksum_type):
        self.urls = urls
        self.dataset = dataset
        self.name = name
        self.size = size
        self._checksum = checksum
        self._checksum_type = checksum_type

    def __repr__(self):
        """Represent the file as a string."""
        return f"ESGFFile:{self.dataset.replace('.', '/')}/{self.name}"

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
        # TODO: this fails for obs4MIPs datasets with a . in their name
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

        range_urls = [url for url in self.urls if host_accepts_range(url)]
        if len(range_urls) > 1:
            self._download_multiple_urls(local_file, range_urls)
        else:
            for url in self.urls:
                try:
                    self._download_single_url(local_file, url)
                except requests.exceptions.RequestException as exc:
                    logger.info("Failed to download from %s. Message:\n%s",
                                url, exc)
                else:
                    break

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

    def _download_single_url(self, local_file, url):
        """Download file from a single url."""
        hasher = hashlib.new(self._checksum_type)

        tmp_file = self._tmp_local_file(local_file)

        logger.info("Downloading %s to %s", url, tmp_file)
        response = requests.get(url, timeout=60, cert=get_credentials())
        response.raise_for_status()
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

    async def _check_worker_health(self, workers, urls):
        """Check that there is at least 1 worker downloading the file."""
        online_workers = [w for w in workers if not w.cancelled()]
        if not online_workers:
            errors = await asyncio.gather(*workers, return_exceptions=True)
            for error, url in zip(errors, urls):
                if error:
                    logger.warning(
                        "An exception occurred while downloading from %s:\n%s",
                        url, error)
            raise IOError(f"Unable to download {self.name}"
                          f" from {urls}: no hosts are online.")

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
                logger.info("Queued chunks: %s, running %s, total %s",
                            queue.qsize(),
                            queue.unfinished_tasks - queue.qsize(), n_chunks)
                await self._check_worker_health(workers, urls)

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
                logger.info("Requesting work for host %s", hostname)
                chunk = await queue.get()
                start, end = chunk
                headers = {'Range': f'bytes={start}-{end}'}
                logger.info("Start download %s-%s MB of %s", start / 2**20,
                            end / 2**20, url)
                try:
                    async with session.get(url, timeout=60,
                                           headers=headers) as response:
                        content = await response.content.read()
                except (aiohttp.ClientError,
                        asyncio.exceptions.TimeoutError) as exc:
                    logger.info("Not able to download from host %s", hostname)
                    await queue.put(chunk)
                    raise asyncio.CancelledError from exc

                tmp_file.seek(start)
                tmp_file.write(content)
                logger.info("Done: chunk %s-%s MB of %s", start / 2**20,
                            end / 2**20, url)
                queue.task_done()

    def _finalize_download(self, tmp_file, local_file, checksum=None):
        """Move file to correct location if checksum is correct."""
        if not tmp_file.exists():
            raise IOError(
                f"Failed to download file {local_file} from {self.urls}")

        if checksum is None:
            hasher = hashlib.new(self._checksum_type)
            with tmp_file.open('rb') as file:
                hasher.update(file.read())
            checksum = hasher.hexdigest()

        if checksum != self._checksum:
            raise ValueError(
                f"Wrong {self._checksum_type} checksum for file {tmp_file},"
                f" expected: {self._checksum}, got {checksum}.")
        shutil.move(tmp_file, local_file)
