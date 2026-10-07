"""Tests for :mod:`esmvalcore.io.wdcc`."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import pytest
import requests
import requests.adapters
import urllib3

import esmvalcore.exceptions
import esmvalcore.io.esgf._download
from esmvalcore.io import wdcc
from esmvalcore.io.local import LocalDataSource, LocalFile

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_mock import MockerFixture

URL = "https://wdcc.example.org"

ENTRY_NAME_TEMPLATE = (
    "cmip5 {product} {institute} {dataset} {exp} {frequency} "
    "{modeling_realm} {mip} {ensemble} {version} {short_name}"
)
DIRNAME_TEMPLATE = (
    "{project.lower}/{product}/{institute}/{dataset}/{exp}/{frequency}/"
    "{modeling_realm}/{mip}/{ensemble}/{version}"
)
FILENAME_TEMPLATE = "{short_name}_{mip}_{dataset}_{exp}_{ensemble}*.nc"

SETTINGS = wdcc.DownloadSettings(
    url=URL,
    auth_type="WDCC",
    timeout=10,
    max_parallel_downloads=4,
)

FACETS = {
    "project": "CMIP5",
    "product": "output1",
    "institute": "MPI-M",
    "dataset": "MPI-ESM-LR",
    "exp": "historical",
    "frequency": "mon",
    "modeling_realm": "atmos",
    "mip": "Amon",
    "ensemble": "r1i1p1",
    "short_name": "tas",
}

NEW = "cmip5 output1 MPI-M MPI-ESM-LR historical mon atmos Amon r1i1p1 v20120315 tas"
OLD = "cmip5 output1 MPI-M MPI-ESM-LR historical mon atmos Amon r1i1p1 v20110101 tas"
OTHER = "cmip5 output1 MPI-M MPI-ESM-LR historical mon atmos Amon r2i1p1 v20120315 tas"

FILE1 = "tas_Amon_MPI-ESM-LR_historical_r1i1p1_185001-189912.nc"
FILE2 = "tas_Amon_MPI-ESM-LR_historical_r1i1p1_190001-200512.nc"


def _file_info(entry_name: str, blob_id: int, filename: str) -> dict:
    content = filename.encode()
    return {
        "BLOB_ID": blob_id,
        "FILE_NAME": "/".join([*entry_name.split(" "), filename]),
        "FILE_SIZE": len(content),
        "CHECKSUM": hashlib.md5(content).hexdigest(),  # noqa: S324
        "CHECKSUM_TYPE": "MD5",
    }


@dataclass
class FakeWDCC:
    """A fake WDCC server."""

    datasets: dict[str, str] = field(
        default_factory=lambda: {"NEW": NEW, "OLD": OLD, "OTHER": OTHER},
    )
    files: dict[str, list[dict]] = field(
        default_factory=lambda: {
            "NEW": [
                _file_info(NEW, 1, FILE1),
                _file_info(NEW, 2, FILE2),
                _file_info(NEW, 3, "README.txt"),
            ],
            "OLD": [_file_info(OLD, 1, FILE1)],
            "OTHER": [
                _file_info(
                    OTHER,
                    1,
                    "tas_Amon_MPI-ESM-LR_historical_r2i1p1_185001-200512.nc",
                ),
            ],
        },
    )
    on_tape: bool = False
    corrupt: bool = False
    redirect_downloads: bool = False
    reject_sessions: bool = False
    received: list[requests.PreparedRequest] = field(default_factory=list)

    def paths(self) -> list[str]:
        """Return the paths of all requests received."""
        return [urlparse(str(r.url)).path for r in self.received]

    def handle(self, request: requests.PreparedRequest) -> tuple[int, Any]:
        """Return the status code and content for a request."""
        url = urlparse(str(request.url))
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        handlers = {
            "/ui/solr/select": self._search,
            "/ui/cerarest/downloadForm": self._download_form,
            "/ui/cerarest/containerInfo": self._container_info,
            "/ui/cerarest/login": self._login,
            "/WDCC/ui/download/transferGeneric": self._transfer,
            "/ui/login": self._login_page,
            "/cache": self._cache,
        }
        if url.path not in handlers:
            return 404, None
        return handlers[url.path](request, params)

    def _search(
        self,
        _: requests.PreparedRequest,
        params: dict[str, str],
    ) -> tuple[int, Any]:
        start = int(params["start"])
        rows = int(params["rows"])
        docs = [
            {"entry_acronym_s": k, "entry_name_s": v}
            for k, v in self.datasets.items()
        ]
        return 200, {
            "response": {
                "numFound": len(docs),
                "docs": docs[start : start + rows],
            },
        }

    def _download_form(
        self,
        _: requests.PreparedRequest,
        params: dict[str, str],
    ) -> tuple[int, Any]:
        acronym = params["acronym"]
        if acronym not in self.files:
            return 400, {"error": "Dataset does not exist", "status": 400}
        return 200, {"downloadInfo": {"metaTable": self.files[acronym]}}

    def _container_info(
        self,
        _: requests.PreparedRequest,
        __: dict[str, str],
    ) -> tuple[int, Any]:
        container = {
            "MIN_BLOB": 1,
            "MAX_BLOB": 3,
            "IS_CACHED": not self.on_tape,
        }
        return 200, {"containers": [container]}

    def _login(
        self,
        _: requests.PreparedRequest,
        __: dict[str, str],
    ) -> tuple[int, Any]:
        return 200, {"login": "ok"}

    def _transfer(
        self,
        request: requests.PreparedRequest,
        params: dict[str, str],
    ) -> tuple[int, Any]:
        if self.reject_sessions or "session=secret" not in request.headers.get(
            "Cookie",
            "",
        ):
            return 302, f"{URL}/ui/login"
        if self.redirect_downloads:
            return 302, f"{URL}/cache?{urlparse(str(request.url)).query}"
        return self._cache(request, params)

    def _cache(
        self,
        _: requests.PreparedRequest,
        params: dict[str, str],
    ) -> tuple[int, Any]:
        acronym = params["acronym"]
        blob_id = int(params["rmin"])
        info = next(i for i in self.files[acronym] if i["BLOB_ID"] == blob_id)
        content = Path(info["FILE_NAME"]).name.encode()
        return 200, b"corrupt" + content if self.corrupt else content

    def _login_page(
        self,
        _: requests.PreparedRequest,
        __: dict[str, str],
    ) -> tuple[int, Any]:
        return 200, b"<html>Please log in</html>"

    def send(
        self,
        adapter: requests.adapters.HTTPAdapter,
        request: requests.PreparedRequest,
        **_: Any,
    ) -> requests.Response:
        """Send a request to the fake server."""
        self.received.append(request)
        status, content = self.handle(request)
        headers = {}
        if status == 302:
            headers["Location"] = content
            content = b""
        if urlparse(str(request.url)).path == "/ui/cerarest/login":
            headers["Set-Cookie"] = "session=secret; Path=/"
        if isinstance(content, bytes):
            body = content
        else:
            body = json.dumps(content).encode()
            headers["Content-Type"] = "application/json"
        raw = urllib3.HTTPResponse(
            body=io.BytesIO(body),
            headers=headers,
            status=status,
            preload_content=False,
        )
        if "Set-Cookie" in headers:
            # requests reads cookies from the original http.client response.
            message = http.client.HTTPMessage()
            message["Set-Cookie"] = headers["Set-Cookie"]
            raw._original_response = SimpleNamespace(  # type: ignore[assignment]
                msg=message,
                isclosed=lambda: True,
            )
        return adapter.build_response(request, raw)


@pytest.fixture(autouse=True)
def cache_dir(tmp_path: Path, mocker: MockerFixture) -> Iterator[Path]:
    """Use a temporary cache directory and fresh sessions in each test."""
    cache_dir = tmp_path / "cache"
    mocker.patch.object(
        wdcc.platformdirs,
        "user_cache_path",
        return_value=cache_dir,
    )
    wdcc._get_cached_session.cache_clear()
    wdcc._DOWNLOAD_SESSIONS.clear()
    yield cache_dir
    wdcc._get_cached_session.cache_clear()
    wdcc._DOWNLOAD_SESSIONS.clear()


@pytest.fixture
def server(mocker: MockerFixture) -> FakeWDCC:
    """Fake WDCC server."""
    fake = FakeWDCC()
    mocker.patch.object(
        requests.adapters.HTTPAdapter,
        "send",
        autospec=True,
        side_effect=fake.send,
    )
    return fake


@pytest.fixture
def netrc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a netrc file with credentials for the fake server."""
    netrc_file = tmp_path / "netrc"
    netrc_file.write_text(
        "machine wdcc.example.org login user password pass\n",
        encoding="utf-8",
    )
    netrc_file.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc_file))
    return netrc_file


@pytest.fixture
def data_source(tmp_path: Path) -> wdcc.WDCCDataSource:
    """WDCC data source configured for CMIP5."""
    return wdcc.WDCCDataSource(
        name="wdcc",
        project="CMIP5",
        priority=10,
        download_dir=tmp_path / "climate_data",
        dirname_template=DIRNAME_TEMPLATE,
        filename_template=FILENAME_TEMPLATE,
        entry_name_template=ENTRY_NAME_TEMPLATE,
        filter_queries=['project_acronym_ss:"IPCC-AR5_CMIP5"'],
        url=URL,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("MPI-ESM-LR", r"MPI\-ESM\-LR"),
        ("a b", r"a\ b"),
        ("r*i1p?", "r*i1p?"),
        ("CESM1(CAM5)", r"CESM1\(CAM5\)"),
        ('a:b/c"', r"a\:b\/c\""),
    ],
)
def test_escape_solr(value: str, expected: str) -> None:
    assert wdcc._escape_solr(value) == expected


def test_find_data(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
    tmp_path: Path,
) -> None:
    files = data_source.find_data(**FACETS)

    assert [f.name for f in files] == [FILE1, FILE2]
    assert all(f.acronym == "NEW" for f in files)
    assert [f.blob_id for f in files] == [1, 2]
    assert files[0].size == len(FILE1)
    assert files[0].checksum_type == "MD5"
    assert files[0].download_settings == wdcc.DownloadSettings(
        url=URL,
        auth_type="WDCC",
        timeout=3600,
        max_parallel_downloads=4,
    )
    assert files[0].facets == {
        **FACETS,
        "version": "v20120315",
        "timerange": "185001/189912",
    }
    assert files[0].local_file == (
        tmp_path
        / "climate_data"
        / "cmip5/output1/MPI-M/MPI-ESM-LR/historical/mon/atmos/Amon"
        / "r1i1p1/v20120315"
        / FILE1
    )
    assert data_source.debug_info.startswith("Files found on WDCC")

    # Check the search query.
    solr_request = server.received[0]
    assert urlparse(solr_request.url).path == "/ui/solr/select"
    params = parse_qs(urlparse(str(solr_request.url)).query)
    assert params["fq"] == [
        'project_acronym_ss:"IPCC-AR5_CMIP5"',
        r"entry_name_s:(cmip5\ output1\ MPI\-M\ MPI\-ESM\-LR\ historical\ "
        r"mon\ atmos\ Amon\ r1i1p1\ *\ tas)",
    ]
    assert params["sort"] == ["title_sort asc"]


def test_find_data_downloaded_files_found_locally(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
) -> None:
    """Test that downloaded files can be found by a LocalDataSource."""
    files = data_source.find_data(**FACETS)
    for file in files:
        file.local_file.parent.mkdir(parents=True, exist_ok=True)
        file.local_file.touch()

    local_source = LocalDataSource(
        name="local",
        project="CMIP5",
        priority=1,
        rootpath=data_source.download_dir,
        dirname_template=DIRNAME_TEMPLATE,
        filename_template=FILENAME_TEMPLATE,
    )
    local_files = local_source.find_data(**FACETS)
    assert local_files == [f.local_file for f in files]


def test_find_data_version(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
) -> None:
    files = data_source.find_data(**FACETS, version="v20110101")
    assert [(f.acronym, f.name) for f in files] == [("OLD", FILE1)]


def test_find_data_glob(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
) -> None:
    files = data_source.find_data(**{**FACETS, "ensemble": "r*i1p1"})
    assert [f.acronym for f in files] == ["NEW", "NEW", "OTHER"]
    assert [f.facets["ensemble"] for f in files] == [
        "r1i1p1",
        "r1i1p1",
        "r2i1p1",
    ]


def test_find_data_timerange(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
) -> None:
    files = data_source.find_data(**FACETS, timerange="1950/1960")
    assert [f.name for f in files] == [FILE2]


def test_find_data_list_facet(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
) -> None:
    data_source.find_data(**{**FACETS, "product": ["output1", "output2"]})
    params = parse_qs(urlparse(str(server.received[0].url)).query)
    assert params["fq"][-1] == (
        r"entry_name_s:("
        r"cmip5\ output1\ MPI\-M\ MPI\-ESM\-LR\ historical\ mon\ atmos\ "
        r"Amon\ r1i1p1\ *\ tas OR "
        r"cmip5\ output2\ MPI\-M\ MPI\-ESM\-LR\ historical\ mon\ atmos\ "
        r"Amon\ r1i1p1\ *\ tas)"
    )


def test_find_data_original_short_name(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
) -> None:
    facets = {**FACETS, "short_name": "tas2", "original_short_name": "tas"}
    files = data_source.find_data(**facets)
    assert [f.name for f in files] == [FILE1, FILE2]


def test_find_data_missing_facet(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
) -> None:
    facets = dict(FACETS)
    facets.pop("institute")
    assert data_source.find_data(**facets) == []
    assert "institute" in data_source.debug_info
    assert not server.received


def test_find_data_missing_download_facet(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
) -> None:
    data_source.dirname_template = f"{DIRNAME_TEMPLATE}/{{grid}}"
    assert data_source.find_data(**FACETS) == []
    assert data_source.debug_info.startswith(
        "Unable to determine the download directory",
    )
    assert "grid" in data_source.debug_info


def test_find_data_ambiguous_download_dir(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
) -> None:
    data_source.dirname_template = f"{DIRNAME_TEMPLATE}/{{grid}}"
    assert data_source.find_data(**FACETS, grid=["gn", "gr"]) == []
    assert data_source.debug_info.startswith(
        "Unable to determine the download directory: the template",
    )
    assert "multiple directories" in data_source.debug_info


def test_find_data_nothing_found(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
) -> None:
    server.datasets = {}
    assert data_source.find_data(**FACETS) == []
    assert data_source.debug_info.startswith(
        f"No files found on WDCC with query: {URL}/ui/solr/select?q=*:*&fq=",
    )


def test_find_data_list_files_fails(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
) -> None:
    server.files.pop("NEW")
    files = data_source.find_data(**FACETS)
    assert [(f.acronym, f.name) for f in files] == [("OLD", FILE1)]


def test_search_pagination(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(wdcc, "_SOLR_ROWS", 1)
    files = data_source.find_data(**FACETS)
    assert [f.name for f in files] == [FILE1, FILE2]
    assert server.paths().count("/ui/solr/select") == 3


def test_search_cached(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
    cache_dir: Path,
) -> None:
    files1 = data_source.find_data(**FACETS)
    n_requests = len(server.received)
    assert n_requests > 0
    assert (cache_dir / "wdcc.sqlite").exists()

    files2 = data_source.find_data(**FACETS)
    assert files1 == files2
    assert len(server.received) == n_requests


def test_prepare(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
    netrc: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server.on_tape = True
    file = data_source.find_data(**FACETS)[0]
    assert not file.local_file.exists()

    with caplog.at_level(logging.INFO):
        file.prepare()

    assert file.local_file.read_bytes() == FILE1.encode()
    assert "stored on tape" in caplog.text
    login = next(
        r for r in server.received if str(r.url).endswith("/ui/cerarest/login")
    )
    assert json.loads(login.body or b"") == {
        "user": "user",
        "pw": "pass",
        "authType": "WDCC",
    }
    # No temporary files are left behind.
    assert list(file.local_file.parent.iterdir()) == [file.local_file]


def test_prepare_existing_file(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
) -> None:
    file = data_source.find_data(**FACETS)[0]
    file.local_file.parent.mkdir(parents=True)
    file.local_file.write_bytes(b"existing")
    n_requests = len(server.received)

    file.prepare()

    assert len(server.received) == n_requests
    assert file.local_file.read_bytes() == b"existing"


def test_login_and_download_not_cached(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
    netrc: Path,
) -> None:
    file = data_source.find_data(**FACETS)[0]
    file.prepare()
    file.local_file.unlink()
    file.prepare()

    paths = server.paths()
    assert paths.count("/ui/cerarest/login") == 1
    assert paths.count("/WDCC/ui/download/transferGeneric") == 2
    assert paths.count("/ui/cerarest/containerInfo") == 2
    session = wdcc._DOWNLOAD_SESSIONS[(URL, "WDCC")]
    assert type(session) is requests.Session


@pytest.mark.parametrize(
    ("corrupt", "match"),
    [
        (True, "Wrong size"),
        (False, "Wrong MD5 checksum"),
    ],
)
def test_prepare_corrupt(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
    netrc: Path,
    corrupt: bool,
    match: str,
) -> None:
    server.corrupt = corrupt
    file = data_source.find_data(**FACETS)[0]
    if not corrupt:
        file.checksum = "wrong"

    with pytest.raises(esmvalcore.exceptions.DownloadError, match=match):
        file.prepare()
    assert not file.local_file.exists()
    assert list(file.local_file.parent.iterdir()) == []


def test_prepare_no_checksum(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
    netrc: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    file = data_source.find_data(**FACETS)[0]
    file.checksum = None
    file.checksum_type = None
    file.prepare()
    assert file.local_file.exists()
    assert "No checksum available" in caplog.text


def test_prepare_session_expired(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
    netrc: Path,
) -> None:
    file = data_source.find_data(**FACETS)[0]
    # Simulate an expired session.
    expired = requests.Session()
    wdcc._DOWNLOAD_SESSIONS[(URL, "WDCC")] = expired

    file.prepare()

    assert file.local_file.read_bytes() == FILE1.encode()
    assert server.paths().count("/ui/cerarest/login") == 1
    assert wdcc._DOWNLOAD_SESSIONS[(URL, "WDCC")] is not expired


def test_get_download_session_already_renewed(
    server: FakeWDCC,
    netrc: Path,
) -> None:
    """Test that a session renewed by another thread is reused."""
    session = wdcc._get_download_session(SETTINGS)
    assert wdcc._get_download_session(SETTINGS, expired=session) is not session
    renewed = wdcc._get_download_session(SETTINGS)
    assert wdcc._get_download_session(SETTINGS, expired=session) is renewed
    assert server.paths().count("/ui/cerarest/login") == 2


def test_prepare_not_logged_in(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
    netrc: Path,
) -> None:
    server.reject_sessions = True
    file = data_source.find_data(**FACETS)[0]
    with pytest.raises(
        esmvalcore.exceptions.DownloadError,
        match="not logged in",
    ):
        file.prepare()
    # Log in once, and once more after the first attempt was rejected.
    assert server.paths().count("/ui/cerarest/login") == 2
    assert not file.local_file.exists()


def test_prepare_redirect(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
    netrc: Path,
) -> None:
    server.redirect_downloads = True
    file = data_source.find_data(**FACETS)[0]
    file.prepare()
    assert file.local_file.read_bytes() == FILE1.encode()
    assert "/cache" in server.paths()


def test_prepare_uppercase_checksum(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
    netrc: Path,
) -> None:
    file = data_source.find_data(**FACETS)[0]
    assert file.checksum is not None
    file.checksum = file.checksum.upper()
    file.prepare()
    assert file.local_file.read_bytes() == FILE1.encode()


def test_prepare_no_credentials(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NETRC", str(tmp_path / "does-not-exist"))
    file = data_source.find_data(**FACETS)[0]
    with pytest.raises(
        esmvalcore.exceptions.DownloadError,
        match=r"machine wdcc\.example\.org login <username>",
    ):
        file.prepare()


def test_prepare_login_fails(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
    netrc: Path,
    mocker: MockerFixture,
) -> None:
    file = data_source.find_data(**FACETS)[0]
    mocker.patch.object(
        server,
        "handle",
        return_value=(401, {"error": "Unauthorized"}),
    )
    with pytest.raises(
        esmvalcore.exceptions.DownloadError,
        match="Failed to log in",
    ):
        file.prepare()


def test_prepare_request_error(
    data_source: wdcc.WDCCDataSource,
    netrc: Path,
    mocker: MockerFixture,
) -> None:
    file = wdcc.WDCCFile(
        acronym="NEW",
        blob_id=1,
        size=1,
        checksum=None,
        checksum_type=None,
        local_file=LocalFile(data_source.download_dir / FILE1),
        download_settings=SETTINGS,
    )
    mocker.patch.object(
        requests.adapters.HTTPAdapter,
        "send",
        side_effect=requests.exceptions.ConnectionError("no connection"),
    )
    with pytest.raises(
        esmvalcore.exceptions.DownloadError,
        match="no connection",
    ):
        file.prepare()


@pytest.fixture
def wdcc_file(data_source: wdcc.WDCCDataSource) -> wdcc.WDCCFile:
    local_file = LocalFile(data_source.download_dir / FILE1)
    local_file.facets = {"short_name": "tas"}
    return wdcc.WDCCFile(
        acronym="NEW",
        blob_id=1,
        size=10,
        checksum=None,
        checksum_type=None,
        local_file=local_file,
        download_settings=SETTINGS,
    )


def test_file_properties(wdcc_file: wdcc.WDCCFile) -> None:
    assert wdcc_file.name == FILE1
    assert wdcc_file.facets == {"short_name": "tas"}
    wdcc_file.facets = {"short_name": "pr"}
    assert wdcc_file.local_file.facets == {"short_name": "pr"}
    assert repr(wdcc_file) == f"WDCCFile:NEW/{FILE1}"
    with pytest.raises(ValueError, match="Attributes have not been read yet"):
        wdcc_file.attributes  # noqa: B018


def test_file_comparison(
    wdcc_file: wdcc.WDCCFile,
    data_source: wdcc.WDCCDataSource,
) -> None:
    other = wdcc.WDCCFile(
        acronym="OLD",
        blob_id=1,
        size=10,
        checksum=None,
        checksum_type=None,
        local_file=LocalFile(data_source.download_dir / FILE1),
        download_settings=SETTINGS,
    )
    same = wdcc.WDCCFile(
        acronym="NEW",
        blob_id=2,
        size=10,
        checksum=None,
        checksum_type=None,
        local_file=LocalFile(data_source.download_dir / FILE1),
        download_settings=SETTINGS,
    )
    assert wdcc_file < other
    assert wdcc_file == same
    assert wdcc_file != other
    assert wdcc_file != "NEW"
    assert len({wdcc_file, same, other}) == 2


def test_to_iris(wdcc_file: wdcc.WDCCFile, mocker: MockerFixture) -> None:
    prepare = mocker.patch.object(wdcc.WDCCFile, "prepare")
    cubes = mocker.sentinel.cubes

    def to_iris(self: LocalFile) -> object:
        self.attributes = {"attribute": "value"}
        return cubes

    mocker.patch.object(
        LocalFile,
        "to_iris",
        autospec=True,
        side_effect=to_iris,
    )

    assert wdcc_file.to_iris() is cubes
    prepare.assert_called_once_with()
    assert wdcc_file.attributes == {"attribute": "value"}


def test_download(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
    netrc: Path,
    mocker: MockerFixture,
) -> None:
    data_source.max_parallel_downloads = 2
    files = data_source.find_data(**{**FACETS, "ensemble": "r*i1p1"})
    files[0].local_file.parent.mkdir(parents=True)
    files[0].local_file.write_bytes(b"existing")
    executor = mocker.spy(wdcc.concurrent.futures, "ThreadPoolExecutor")
    local_file = LocalFile("local.nc")

    wdcc.download([*files, local_file])

    executor.assert_called_once_with(max_workers=2)
    assert all(f.local_file.exists() for f in files)
    assert files[0].local_file.read_bytes() == b"existing"
    assert server.paths().count("/WDCC/ui/download/transferGeneric") == 2


def test_download_noop(mocker: MockerFixture) -> None:
    executor = mocker.spy(wdcc.concurrent.futures, "ThreadPoolExecutor")
    wdcc.download([LocalFile("local.nc")])
    executor.assert_not_called()


def test_download_fail(
    server: FakeWDCC,
    data_source: wdcc.WDCCDataSource,
    netrc: Path,
) -> None:
    server.corrupt = True
    files = data_source.find_data(**FACETS)
    with pytest.raises(esmvalcore.exceptions.DownloadError) as exc:
        wdcc.download(files, n_jobs=1)
    message = str(exc.value)
    assert message.startswith("Failed to download the following files:")
    assert FILE1 in message
    assert FILE2 in message


def test_esgf_download_error_is_shared() -> None:
    assert (
        esmvalcore.io.esgf._download.DownloadError
        is esmvalcore.exceptions.DownloadError
    )
