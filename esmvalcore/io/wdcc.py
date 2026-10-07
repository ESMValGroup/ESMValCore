"""Find and download data from the World Data Center for Climate (WDCC).

The `World Data Center for Climate <https://www.wdc-climate.de>`_ (WDCC) at
the German Climate Computing Center (DKRZ) archives a large collection of
climate model output, including copies of the CMIP5 and CORDEX (CMIP5-driven)
data.

This module provides the :class:`WDCCDataSource`, which uses the
`WDCC API <https://www.wdc-climate.de/ui/wdcc-api-docs/>`_ to search for data.
Searching does not require an account. The data is downloaded the first time
it is used. Downloading data requires a WDCC or DKRZ account; WDCC accounts can
be requested at https://www.wdc-climate.de/ui/register. Add the username and
password to your `~/.netrc <https://everything.curl.dev/usingcurl/netrc>`__
file:

.. code-block::

    machine www.wdc-climate.de login <username> password <password>

and make sure that this file is only readable by you by running
``chmod 600 ~/.netrc``.

Much of the data archived by WDCC is stored on tape. If that is the case, it
may take a while before the download starts. To speed this up, multiple files
are downloaded in parallel. This can be configured with the
:attr:`WDCCDataSource.max_parallel_downloads` setting.

Search results are cached on disk in the user cache directory (e.g.
``~/.cache/esmvalcore/wdcc.sqlite`` on Linux) for
:attr:`WDCCDataSource.cache_expire_after` seconds.

To use this module, run the command

.. code:: bash

    esmvaltool config copy data-wdcc.yml

to copy the example configuration file for this module to your configuration
directory. This will create a file with the following content:

.. literalinclude:: ../configurations/data-wdcc.yml
   :caption: Contents of ``data-wdcc.yml``
   :language: yaml

Downloaded files are stored in the same directory structure as used in
``data-local.yml``, so run

.. code:: bash

    esmvaltool config copy data-local.yml

as well to find previously downloaded data without searching WDCC.

See :ref:`config-data-sources` for more information on configuring data
sources.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import functools
import hashlib
import logging
import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import platformdirs
import requests
from humanfriendly import format_size, format_timespan

from esmvalcore.exceptions import DownloadError
from esmvalcore.io.esgf._download import get_download_message
from esmvalcore.io.local import (
    LocalDataSource,
    LocalFile,
    _MissingFacetError,
    _replace_tags,
    _select_files,
    _select_latest_version,
)
from esmvalcore.io.protocol import DataElement, DataSource

if TYPE_CHECKING:
    from collections.abc import Iterable

    import iris.cube
    import requests_cache

    import esmvalcore.io.local
    from esmvalcore.typing import Facets, FacetValue

__all__ = [
    "DownloadSettings",
    "WDCCDataSource",
    "WDCCFile",
    "download",
]

logger = logging.getLogger(__name__)

_SOLR_PATH = "/ui/solr/select"
_DOWNLOAD_FORM_PATH = "/ui/cerarest/downloadForm"
_CONTAINER_INFO_PATH = "/ui/cerarest/containerInfo"
_LOGIN_PATH = "/ui/cerarest/login"
_LOGIN_PAGE_PATH = "/ui/login"
_DOWNLOAD_PATH = "/WDCC/ui/download/transferGeneric"

_SOLR_ROWS = 1000
"""Number of search results to request at once."""

_CONNECT_TIMEOUT = 60
"""Timeout (in seconds) for connecting to WDCC."""

_SOLR_SPECIAL_CHARS = re.compile(r'([+\-!(){}\[\]^"~:\\/&|\s])')
"""Characters that need to be escaped in a Solr query, except wildcards."""


def _escape_solr(value: str) -> str:
    """Escape special characters in a Solr query term, except wildcards."""
    return _SOLR_SPECIAL_CHARS.sub(r"\\\1", value)


@functools.cache
def _get_cached_session(expire_after: int) -> requests_cache.CachedSession:
    """Get a session that caches responses on disk.

    Only used for requests that do not require authentication.
    """
    import requests_cache  # noqa: PLC0415

    cache_dir = platformdirs.user_cache_path("esmvalcore")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return requests_cache.CachedSession(
        cache_name=str(cache_dir / "wdcc.sqlite"),
        backend="sqlite",
        expire_after=expire_after,
        allowable_codes=(200,),
        allowable_methods=("GET",),
    )


@dataclass(frozen=True)
class DownloadSettings:
    """Settings for downloading files from WDCC."""

    url: str
    """The URL of WDCC."""

    auth_type: str
    """The type of account used to log in, either ``WDCC`` or ``DKRZ``."""

    timeout: float
    """Timeout (in seconds) for downloads."""

    max_parallel_downloads: int
    """The maximum number of files to download in parallel."""


_LOGIN_LOCK = threading.Lock()
_DOWNLOAD_SESSIONS: dict[tuple[str, str], requests.Session] = {}


def _get_download_session(
    settings: DownloadSettings,
    expired: requests.Session | None = None,
) -> requests.Session:
    """Get a session that is logged in to WDCC.

    The credentials are read from the ``~/.netrc`` file, or the file specified
    by the ``NETRC`` environment variable.

    If ``expired`` is the current session, log in again. Other threads may
    already have replaced an expired session, in which case the new session
    is returned.
    """
    url = settings.url
    auth_type = settings.auth_type
    with _LOGIN_LOCK:
        key = (url, auth_type)
        if expired is not None and _DOWNLOAD_SESSIONS.get(key) is expired:
            del _DOWNLOAD_SESSIONS[key]
        if key not in _DOWNLOAD_SESSIONS:
            credentials = requests.utils.get_netrc_auth(url)
            if credentials is None:
                host = url.split("://", 1)[-1].split("/", 1)[0]
                msg = (
                    f"No credentials for {url} found. Downloading data from "
                    "WDCC requires a WDCC or DKRZ account. Please add the line "
                    f"'machine {host} login <username> password <password>' "
                    "to your ~/.netrc file."
                )
                raise DownloadError(msg)
            user, password = credentials
            session = requests.Session()
            logger.debug("Logging in to %s as %s", url, user)
            try:
                response = session.post(
                    f"{url}{_LOGIN_PATH}",
                    json={"user": user, "pw": password, "authType": auth_type},
                    timeout=_CONNECT_TIMEOUT,
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as exc:
                msg = (
                    f"Failed to log in to {url} as user '{user}' with a "
                    f"{auth_type} account: {exc}"
                )
                raise DownloadError(msg) from exc
            _DOWNLOAD_SESSIONS[key] = session
        return _DOWNLOAD_SESSIONS[key]


@functools.total_ordering
class WDCCFile(DataElement):
    """A file that can be downloaded from WDCC.

    This is the data element returned by :meth:`WDCCDataSource.find_data`.
    """

    def __init__(
        self,
        *,
        acronym: str,
        blob_id: int,
        size: int,
        checksum: str | None,
        checksum_type: str | None,
        local_file: esmvalcore.io.local.LocalFile,
        download_settings: DownloadSettings,
    ) -> None:
        self.name = local_file.name
        """A unique name identifying the data."""
        self.acronym = acronym
        """The WDCC acronym of the dataset that the file belongs to."""
        self.blob_id = blob_id
        """The number of the file in the WDCC dataset."""
        self.size = size
        """The size of the file in bytes."""
        self.checksum = checksum
        """The checksum of the file."""
        self.checksum_type = checksum_type
        """The type of checksum, e.g. MD5."""
        self.local_file = local_file
        """The location of the file after it has been downloaded."""
        self.download_settings = download_settings
        """Settings for downloading the file."""
        self._attributes: dict[str, Any] | None = None

    @property
    def facets(self) -> Facets:
        """Facets are key-value pairs that were used to find this data."""
        return self.local_file.facets

    @facets.setter
    def facets(self, value: Facets) -> None:
        self.local_file.facets = value

    @property
    def attributes(self) -> dict[str, Any]:
        """Attributes are key-value pairs describing the data."""
        if self._attributes is None:
            msg = (
                "Attributes have not been read yet. Call the `to_iris` method "
                "first to read the attributes from the file."
            )
            raise ValueError(msg)
        return self._attributes

    @attributes.setter
    def attributes(self, value: dict[str, Any]) -> None:
        self._attributes = value

    def __repr__(self) -> str:
        """Represent the file as a string."""
        return f"WDCCFile:{self.acronym}/{self.name}"

    def __eq__(self, other: object) -> bool:
        """Compare `self` to `other`."""
        return isinstance(other, WDCCFile) and (self.acronym, self.name) == (
            other.acronym,
            other.name,
        )

    def __lt__(self, other: WDCCFile) -> bool:
        """Compare `self` to `other`."""
        return (self.acronym, self.name) < (other.acronym, other.name)

    def __hash__(self) -> int:
        """Return a number uniquely representing the data element."""
        return hash((self.acronym, self.name))

    def prepare(self) -> None:
        """Prepare the data for access by downloading it if needed.

        Raises
        ------
        esmvalcore.exceptions.DownloadError
            Raised if downloading the file failed.
        """
        if self.local_file.exists():
            return
        try:
            self._download()
        except requests.exceptions.RequestException as exc:
            msg = f"Failed to download {self}: {exc}"
            raise DownloadError(msg) from exc

    def to_iris(self) -> iris.cube.CubeList:
        """Load the data as Iris cubes.

        Returns
        -------
        iris.cube.CubeList
            The loaded data.
        """
        self.prepare()
        cubes = self.local_file.to_iris()
        self.attributes = self.local_file.attributes
        return cubes

    def _is_on_tape(self) -> bool:
        """Check if the file needs to be retrieved from tape storage."""
        response = requests.get(
            f"{self.download_settings.url}{_CONTAINER_INFO_PATH}",
            params={"acronym": self.acronym},
            timeout=_CONNECT_TIMEOUT,
        )
        if not response.ok:
            return False
        containers = response.json().get("containers") or []
        return any(
            not container.get("IS_CACHED", True)
            for container in containers
            if container.get("MIN_BLOB", 0)
            <= self.blob_id
            <= container.get("MAX_BLOB", 0)
        )

    def _write(self, response: requests.Response, tmp_file: Path) -> None:
        """Write the downloaded data to file and check its integrity."""
        hasher = (
            None
            if self.checksum_type is None
            else hashlib.new(self.checksum_type.lower())
        )
        size = 0
        with tmp_file.open("wb") as file:
            megabyte = 2**20
            for chunk in response.iter_content(chunk_size=megabyte):
                if hasher is not None:
                    hasher.update(chunk)
                size += len(chunk)
                file.write(chunk)

        if size != self.size:
            msg = (
                f"Wrong size for file {tmp_file}, downloaded from "
                f"{response.url}: expected {self.size} bytes, but got {size} "
                "bytes."
            )
            raise DownloadError(msg)
        if hasher is None:
            logger.warning(
                "No checksum available, unable to check data integrity for %s",
                self,
            )
        elif (local_checksum := hasher.hexdigest()) != str(
            self.checksum,
        ).lower():
            msg = (
                f"Wrong {self.checksum_type} checksum for file {tmp_file}, "
                f"downloaded from {response.url}: expected {self.checksum}, "
                f"but got {local_checksum}. Try downloading the file again."
            )
            raise DownloadError(msg)

    def _request(self, session: requests.Session) -> requests.Response | None:
        """Request the file.

        Returns ``None`` if the server redirected to the login page, i.e.
        the session is not logged in.
        """
        settings = self.download_settings
        response = session.get(
            f"{settings.url}{_DOWNLOAD_PATH}",
            params={
                "acronym": self.acronym,
                "rmin": str(self.blob_id),
                "rmax": str(self.blob_id),
            },
            stream=True,
            timeout=(_CONNECT_TIMEOUT, settings.timeout),
        )
        if urlparse(response.url).path.rstrip("/").endswith(_LOGIN_PAGE_PATH):
            response.close()
            return None
        return response

    def _download(self) -> None:
        """Download the file."""
        settings = self.download_settings
        session = _get_download_session(settings)
        if self._is_on_tape():
            logger.info(
                "%s is stored on tape, it may take a while before the "
                "download starts",
                self,
            )

        self.local_file.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(prefix=f"{self.local_file}.") as file:
            tmp_file = Path(file.name)

        logger.debug("Downloading %s to %s", self, tmp_file)
        start_time = datetime.datetime.now()
        response = self._request(session)
        if response is None:
            logger.debug("Session for %s expired, logging in again", self)
            session = _get_download_session(settings, expired=session)
            response = self._request(session)
        if response is None:
            msg = (
                f"Failed to download {self}: not logged in to {settings.url}. "
                "Please check the credentials in your ~/.netrc file."
            )
            raise DownloadError(msg)
        response.raise_for_status()
        try:
            self._write(response, tmp_file)
        except BaseException:
            tmp_file.unlink(missing_ok=True)
            raise

        shutil.move(tmp_file, self.local_file)
        duration = (datetime.datetime.now() - start_time).total_seconds()
        logger.info(
            "Downloaded %s (%s) in %s (%s/s) from WDCC",
            self.local_file,
            format_size(self.size),
            format_timespan(duration),
            format_size(round(self.size / duration)) if duration else "-",
        )


@dataclass
class WDCCDataSource(DataSource):
    """Data source for finding and downloading data from WDCC."""

    name: str
    """A name identifying the data source."""

    project: str
    """The project that the data source provides data for."""

    priority: int
    """The priority of the data source. Lower values have priority."""

    download_dir: Path
    """The destination directory where data will be downloaded."""

    dirname_template: str
    """The template for the directory names where data will be downloaded.

    See :attr:`esmvalcore.io.local.LocalDataSource.dirname_template`.
    """

    filename_template: str
    """The template for the file names.

    See :attr:`esmvalcore.io.local.LocalDataSource.filename_template`.
    """

    entry_name_template: str
    """The template for the WDCC entry name of a dataset.

    This is used to search WDCC and to read facets from the search results.
    It should be written like
    :attr:`esmvalcore.io.local.LocalDataSource.dirname_template`, but with
    spaces instead of ``/`` as separators, because WDCC entry names use spaces.
    """

    filter_queries: list[str] = field(default_factory=list)
    """Additional `Solr filter queries <https://solr.apache.org/guide/solr/latest/query-guide/common-query-parameters.html#fq-filter-query-parameter>`__
    used when searching WDCC, e.g. to only find data from a particular project.
    """

    auth_type: str = "WDCC"
    """The type of account used to log in, either ``WDCC`` or ``DKRZ``."""

    url: str = "https://www.wdc-climate.de"
    """The URL of WDCC."""

    timeout: float = 3600
    """Timeout (in seconds) for downloads.

    Downloads can take a long time to start if the data is stored on tape.
    """

    cache_expire_after: int = 86400
    """Time (in seconds) that search results are cached."""

    max_parallel_downloads: int = 4
    """The maximum number of files to download in parallel."""

    debug_info: str = field(init=False, repr=False, default="")
    """A string containing debug information when no data is found."""

    def __post_init__(self) -> None:
        """Set further attributes."""
        self.download_dir = Path(
            os.path.expandvars(self.download_dir),
        ).expanduser()
        self.url = self.url.rstrip("/")
        # Use a LocalDataSource to read facets from the entry names
        # and file names found on WDCC.
        self._parser = LocalDataSource(
            name=self.name,
            project=self.project,
            priority=self.priority,
            rootpath=Path(os.sep),
            dirname_template=self.entry_name_template.replace(" ", "/"),
            filename_template=self.filename_template,
        )

    def _search(self, filter_queries: list[str]) -> list[dict[str, str]]:
        """Search WDCC for datasets."""
        session = _get_cached_session(self.cache_expire_after)
        results: list[dict[str, str]] = []
        while True:
            params: dict[str, str | int | list[str]] = {
                "q": "*:*",
                "fq": filter_queries,
                "fl": "entry_acronym_s,entry_name_s",
                "rows": _SOLR_ROWS,
                "start": len(results),
                "sort": "title_sort asc",
                "wt": "json",
            }
            response = session.get(
                f"{self.url}{_SOLR_PATH}",
                params=params,
                timeout=_CONNECT_TIMEOUT,
            )
            response.raise_for_status()
            content = response.json()["response"]
            results.extend(content["docs"])
            if not content["docs"] or len(results) >= content["numFound"]:
                return results

    def _list_files(self, acronym: str) -> list[dict[str, Any]]:
        """List the files that are part of a WDCC dataset."""
        session = _get_cached_session(self.cache_expire_after)
        response = session.get(
            f"{self.url}{_DOWNLOAD_FORM_PATH}",
            params={"acronym": acronym},
            timeout=_CONNECT_TIMEOUT,
        )
        if not response.ok:
            logger.debug(
                "Unable to list files of WDCC dataset %s: %s",
                acronym,
                response.text,
            )
            return []
        return response.json()["downloadInfo"].get("metaTable") or []

    def _find_files(
        self,
        filter_queries: list[str],
        entry_globs: list[str],
    ) -> list[tuple[dict[str, str], list[dict[str, Any]]]]:
        """Find datasets matching the entry name globs and list their files."""
        datasets = [
            d
            for d in self._search(filter_queries)
            if any(fnmatchcase(d["entry_name_s"], g) for g in entry_globs)
        ]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            file_lists = executor.map(
                self._list_files,
                [d["entry_acronym_s"] for d in datasets],
            )
        return list(zip(datasets, file_lists, strict=True))

    def _get_local_file(
        self,
        entry_name: str,
        filename: str,
        facets: Facets,
        *,
        add_timerange: bool,
    ) -> LocalFile:
        """Get the local path where a file found on WDCC will be stored.

        Raises
        ------
        _MissingFacetError
            If the download directory cannot be determined.
        """
        entry_dir = Path(os.sep, *entry_name.split(" "))
        file_facets: Facets = dict(
            self._parser._path2facets(  # noqa: SLF001
                entry_dir / filename,
                add_timerange=add_timerange,
            ),
        )
        file_facets["project"] = self.project
        dirnames = _replace_tags(
            self.dirname_template,
            {**facets, **file_facets},
        )
        if len(dirnames) != 1:
            msg = (
                f"the template '{self.dirname_template}' results in multiple "
                "directories: " + ", ".join(sorted(str(d) for d in dirnames))
            )
            raise _MissingFacetError(msg)
        local_file = LocalFile(self.download_dir / dirnames[0] / filename)
        local_file.facets = file_facets
        return local_file

    def find_data(self, **facets: FacetValue) -> list[WDCCFile]:
        """Find data.

        Parameters
        ----------
        **facets :
            Find data matching these facets.

        Returns
        -------
        :
            A list of files that have been found on WDCC.
        """
        facets = dict(facets)
        if "original_short_name" in facets:
            facets["short_name"] = facets["original_short_name"]

        try:
            entry_globs = [
                str(p) for p in _replace_tags(self.entry_name_template, facets)
            ]
            filename_globs = [
                str(p) for p in _replace_tags(self.filename_template, facets)
            ]
        except _MissingFacetError as exc:
            self.debug_info = exc.args[0]
            return []

        entry_query = " OR ".join(_escape_solr(g) for g in sorted(entry_globs))
        filter_queries = [
            *self.filter_queries,
            f"entry_name_s:({entry_query})",
        ]
        self.debug_info = (
            "No files found on WDCC with query: "
            f"{self.url}{_SOLR_PATH}?q=*:*&"
            + "&".join(f"fq={q}" for q in filter_queries)
        )

        add_timerange = facets.get("frequency", "fx") != "fx"
        local_files: list[LocalFile] = []
        file_kwargs: dict[LocalFile, dict[str, Any]] = {}
        for dataset, files in self._find_files(filter_queries, entry_globs):
            for info in files:
                filename = Path(info["FILE_NAME"]).name
                if not any(fnmatchcase(filename, g) for g in filename_globs):
                    continue
                try:
                    local_file = self._get_local_file(
                        dataset["entry_name_s"],
                        filename,
                        facets,
                        add_timerange=add_timerange,
                    )
                except _MissingFacetError as exc:
                    self.debug_info = (
                        "Unable to determine the download directory: "
                        f"{exc.args[0]}"
                    )
                    return []
                local_files.append(local_file)
                file_kwargs[local_file] = {
                    "acronym": dataset["entry_acronym_s"],
                    "blob_id": info["BLOB_ID"],
                    "size": info["FILE_SIZE"],
                    "checksum": info.get("CHECKSUM"),
                    "checksum_type": info.get("CHECKSUM_TYPE"),
                }

        if "version" not in facets:
            local_files = _select_latest_version(local_files)
        if "timerange" in facets:
            local_files = _select_files(local_files, facets["timerange"])

        download_settings = DownloadSettings(
            url=self.url,
            auth_type=self.auth_type,
            timeout=self.timeout,
            max_parallel_downloads=self.max_parallel_downloads,
        )
        result = [
            WDCCFile(
                **file_kwargs[local_file],
                local_file=local_file,
                download_settings=download_settings,
            )
            for local_file in sorted(local_files)
        ]
        if result:
            self.debug_info = f"F{self.debug_info[len('No f') :]}"
        return result


def download(
    files: Iterable[DataElement],
    n_jobs: int | None = None,
) -> None:
    """Download multiple files from WDCC in parallel.

    Files that are not of type :class:`WDCCFile` or that have already been
    downloaded are ignored.

    Parameters
    ----------
    files:
        The files to download.
    n_jobs:
        The number of files to download in parallel. If not specified, the
        smallest :attr:`DownloadSettings.max_parallel_downloads` of the files
        is used.

    Raises
    ------
    esmvalcore.exceptions.DownloadError
        Raised if one or more files failed to download.
    """
    to_download = sorted(
        file
        for file in files
        if isinstance(file, WDCCFile) and not file.local_file.exists()
    )
    if not to_download:
        return

    if n_jobs is None:
        n_jobs = min(
            f.download_settings.max_parallel_downloads for f in to_download
        )
    logger.info(get_download_message(to_download))

    total_size = 0
    start_time = datetime.datetime.now()
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        future_to_file = {
            executor.submit(file.prepare): file for file in to_download
        }
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                future.result()
            except DownloadError as error:
                logger.error("Failed to download %s: %s", file, error)
                errors.append(error)
            else:
                total_size += file.size

    duration = (datetime.datetime.now() - start_time).total_seconds()
    logger.info(
        "Downloaded %s from WDCC in %s (%s/s)",
        format_size(total_size),
        format_timespan(duration),
        format_size(round(total_size / duration)) if duration else "-",
    )
    if errors:
        msg = "Failed to download the following files:\n" + "\n".join(
            sorted(str(error) for error in errors),
        )
        raise DownloadError(msg)
