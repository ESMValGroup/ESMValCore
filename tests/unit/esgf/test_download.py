"""Test `esmvalcore.esgf._download`."""
import datetime
import logging
import os
import re
import textwrap
from pathlib import Path

import pytest
import requests
import yaml
from pyesgf.search.results import FileResult

import esmvalcore.esgf
from esmvalcore.esgf import _download


def test_log_speed(monkeypatch, tmp_path):
    hosts_file = tmp_path / '.esmvaltool' / 'cache' / 'esgf-hosts.yml'
    monkeypatch.setattr(_download, 'HOSTS_FILE', hosts_file)

    megabyte = 10**6
    _download.log_speed('http://somehost.org/some_file.nc', 100 * megabyte, 10)
    _download.log_speed('http://somehost.org/some_other_file.nc',
                        200 * megabyte, 16)
    _download.log_speed('http://otherhost.org/other_file.nc', 4 * megabyte, 1)

    with hosts_file.open('r') as file:
        result = yaml.safe_load(file)

    expected = {
        'somehost.org': {
            'speed (MB/s)': 11.5,
            'duration (s)': 26,
            'size (bytes)': 300 * megabyte,
            'error': False,
        },
        'otherhost.org': {
            'speed (MB/s)': 4,
            'duration (s)': 1,
            'size (bytes)': 4 * megabyte,
            'error': False,
        },
    }
    assert result == expected


def test_error(monkeypatch, tmp_path):
    hosts_file = tmp_path / '.esmvaltool' / 'cache' / 'esgf-hosts.yml'
    monkeypatch.setattr(_download, 'HOSTS_FILE', hosts_file)

    megabyte = 10**6
    _download.log_speed('http://somehost.org/some_file.nc', 3 * megabyte, 2)
    _download.log_error('http://somehost.org/some_file.nc')

    with hosts_file.open('r') as file:
        result = yaml.safe_load(file)

    expected = {
        'somehost.org': {
            'speed (MB/s)': 1.5,
            'duration (s)': 2,
            'size (bytes)': 3 * megabyte,
            'error': True,
        }
    }
    assert result == expected


@pytest.mark.parametrize('age_in_hours', [0.5, 2])
def test_get_preferred_hosts(monkeypatch, tmp_path, age_in_hours):
    hosts_file = tmp_path / 'esgf-hosts.yml'
    content = textwrap.dedent("""
    aims3.llnl.gov:
      duration (s): 8
      error: false
      size (bytes): 42065408
      speed (MB/s): 4.9
    esg.lasg.ac.cn:
      duration (s): 37
      error: false
      size (bytes): 3702108
      speed (MB/s): 0.1
    esgdata.gfdl.noaa.gov:
      duration (s): 3
      error: true
      size (bytes): 3124084
      speed (MB/s): 1.0
    esgf.ichec.ie:
      duration (s): 0
      error: false
      size (bytes): 0
      speed (MB/s): 0
    esgf.nci.org.au:
      duration (s): 13
      error: false
      size (bytes): 12461112
      speed (MB/s): 0.9
    """).lstrip()
    hosts_file.write_text(content)
    now = datetime.datetime.now().timestamp()
    file_age = now - age_in_hours * 3600
    os.utime(hosts_file, (file_age, file_age))
    monkeypatch.setattr(_download, 'HOSTS_FILE', hosts_file)

    preferred_hosts = _download.get_preferred_hosts()
    # hosts should be sorted by download speed
    # host with no data downloaded yet in the middle
    # host with a recent error last
    if age_in_hours < 1:
        expected = [
            'aims3.llnl.gov',
            'esgf.ichec.ie',
            'esgf.nci.org.au',
            'esg.lasg.ac.cn',
            'esgdata.gfdl.noaa.gov',
        ]
    else:
        expected = [
            'aims3.llnl.gov',
            'esgdata.gfdl.noaa.gov',
            'esgf.ichec.ie',
            'esgf.nci.org.au',
            'esg.lasg.ac.cn',
        ]
    assert preferred_hosts == expected


def test_get_preferred_hosts_only_zeros(monkeypatch, tmp_path):
    """Test ``get_preferred_hosts`` when speed is zero for all entries."""
    hosts_file = tmp_path / 'esgf-hosts.yml'
    content = textwrap.dedent("""
    aims3.llnl.gov:
      duration (s): 0
      error: false
      size (bytes): 0
      speed (MB/s): 0
    esg.lasg.ac.cn:
      duration (s): 0.0
      error: false
      size (bytes): 0.0
      speed (MB/s): 0.0
    """).lstrip()
    hosts_file.write_text(content)
    monkeypatch.setattr(_download, 'HOSTS_FILE', hosts_file)

    preferred_hosts = _download.get_preferred_hosts()

    # The following assert is safe since "the built-in sorted() function is
    # guaranteed to be stable"
    # (https://docs.python.org/3/library/functions.html)
    expected = ['aims3.llnl.gov', 'esg.lasg.ac.cn']
    assert preferred_hosts == expected


def test_sort_hosts(mocker):
    """Test that hosts are sorted according to priority by sort_hosts."""
    urls = [
        'http://esgf.nci.org.au/abc.nc',
        'http://esgf2.dkrz.de/abc.nc',
        'http://esgf-data1.ceda.ac.uk/abc.nc',
    ]
    preferred_hosts = [
        'esgf2.dkrz.de', 'esgf-data1.ceda.ac.uk', 'aims3.llnl.gov'
    ]
    mocker.patch.object(_download,
                        'get_preferred_hosts',
                        autospec=True,
                        return_value=preferred_hosts)
    sorted_urls = _download.sort_hosts(urls)
    assert sorted_urls == [
        'http://esgf.nci.org.au/abc.nc',
        'http://esgf2.dkrz.de/abc.nc',
        'http://esgf-data1.ceda.ac.uk/abc.nc',
    ]


def test_get_dataset_id_noop():
    file_results = [
        FileResult(
            json={
                'project': ['CMIP6'],
                'source_id': ['ABC'],
                'dataset_id': 'ABC.v1|hostname.org',
            },
            context=None,
        )
    ]
    dataset_id = _download.ESGFFile._get_dataset_id(file_results)
    assert dataset_id == 'ABC.v1'


def test_get_dataset_id_obs4mips():
    file_results = [
        FileResult(
            json={
                'project': ['obs4MIPs'],
                'source_id': ['CERES-EBAF'],
                'dataset_id':
                'obs4MIPs.NASA-LaRC.CERES-EBAF.atmos.mon.v20160610|abc.org',
            },
            context=None,
        )
    ]
    dataset_id = _download.ESGFFile._get_dataset_id(file_results)
    assert dataset_id == 'obs4MIPs.CERES-EBAF.v20160610'


def test_init():
    """Test ESGFFile.__init__()."""
    filename = 'tas_ABC_2000-2001.nc'
    url = f'http://something.org/ABC/v1/{filename}'
    result = FileResult(
        json={
            'dataset_id': 'ABC.v1|something.org',
            'project': ['CMIP6'],
            'size': 10,
            'source_id': ['ABC'],
            'checksum_type': ['MD5'],
            'checksum': ['abc'],
            'title': filename,
            'url': [url + '|application/netcdf|HTTPServer']
        },
        context=None,
    )

    file = _download.ESGFFile([result])
    assert file.name == filename
    assert file.size == 10
    assert file.urls == [url]
    assert file._checksums == [('MD5', 'abc')]
    txt = f"ESGFFile:ABC/v1/{filename} on hosts ['something.org']"
    assert repr(file) == txt
    assert hash(file) == hash(('ABC.v1', filename))


def test_from_results():
    """Test ESGFFile._from_results()."""
    facets = {
        'project': 'CMIP6',
        'variable': 'tas',
    }
    results = []
    for i in range(2):
        filename = f'tas_ABC{i}_2000-2001.nc'
        url = f'http://something.org/ABC/v1/{filename}'
        result = FileResult(
            json={
                'dataset_id': f'ABC{i}.v1|something.org',
                'project': ['CMIP6'],
                'size': 10,
                'source_id': [f'ABC{i}'],
                'title': filename,
                'url': [url + '|application/netcdf|HTTPServer']
            },
            context=None,
        )
        results.append(result)

    # Append an invalid result
    wrong_var_filename = 'zg_ABC0_2000-2001.nc'
    results.append(
        FileResult(
            json={
                'dataset_id': f'ABC{i}.v1|something.org',
                'project': ['CMIP6'],
                'size': 10,
                'source_id': [f'ABC{i}'],
                'title': wrong_var_filename,
            },
            context=None,
        ))

    files = _download.ESGFFile._from_results(results, facets)
    assert len(files) == 2
    for i in range(2):
        assert files[i].name == f'tas_ABC{i}_2000-2001.nc'


def test_sorting():

    result1 = FileResult(
        json={
            'dataset_id': 'ABC.v1|something.org',
            'project': ['CMIP6'],
            'size': 1,
            'title': 'abc_2000-2001.nc',
        },
        context=None,
    )
    result2 = FileResult(
        json={
            'dataset_id': 'ABC.v1|something.org',
            'project': ['CMIP6'],
            'size': 1,
            'title': 'abc_2001-2002.nc',
        },
        context=None,
    )

    file1 = _download.ESGFFile([result1])
    file2 = _download.ESGFFile([result2])
    assert file1 == file1
    assert file1 != file2
    assert file1 < file2
    assert file2 > file1
    assert sorted([file2, file1]) == [file1, file2]


def test_local_file():
    local_path = '/path/to/somewhere'
    filename = 'abc_2000-2001.nc'
    dataset = 'CMIP6.ABC.v1'
    result = FileResult(
        json={
            'dataset_id': f'{dataset}|something.org',
            'project': ['CMIP6'],
            'size': 10,
            'source_id': ['ABC'],
            'title': filename,
        },
        context=None,
    )

    file = _download.ESGFFile([result])
    reference_path = Path(local_path) / 'CMIP6' / 'ABC' / 'v1' / filename
    assert file.local_file(local_path) == reference_path


def test_merge_datasets():
    filename = 'tas_Amon_FIO-ESM_historical_r1i1p1_185001-200512.nc'
    url0 = (
        'http://esgf2.dkrz.de/thredds/fileServer/lta_dataroot/cmip5/output1/'
        'FIO/FIO-ESM/historical/mon/atmos/Amon/r1i1p1/v20121010/tas/'
        'tas_Amon_FIO-ESM_historical_r1i1p1_185001-200512.nc')

    url1 = (
        'http://aims3.llnl.gov/thredds/fileServer/cmip5_css02_data/cmip5/'
        'output1/FIO/fio-esm/historical/mon/atmos/Amon/r1i1p1/v20121010/tas/'
        'tas_Amon_FIO-ESM_historical_r1i1p1_185001-200512.nc')

    dataset0 = ('cmip5.output1.FIO.FIO-ESM.historical.'
                'mon.atmos.Amon.r1i1p1.v20121010')
    dataset1 = ('cmip5.output1.FIO.fio-esm.historical.'
                'mon.atmos.Amon.r1i1p1.v20121010')

    results = [
        FileResult(
            {
                'dataset_id': dataset0 + '|esgf2.dkrz.de',
                'project': ['CMIP5'],
                'size': 200,
                'title': filename,
                'url': [
                    url0 + '|application/netcdf|HTTPServer',
                ],
            }, None),
        FileResult(
            {
                'dataset_id': dataset1 + '|aims3.llnl.gov',
                'project': ['CMIP5'],
                'size': 200,
                'title': filename,
                'url': [
                    url1 + '|application/netcdf|HTTPServer',
                ],
            }, None)
    ]

    file = _download.ESGFFile(results)

    assert file.dataset == dataset0
    assert file.name == filename
    assert file.size == 200
    assert file.urls == [url0, url1]


@pytest.mark.parametrize('checksum', ['yes', 'no', 'wrong'])
def test_single_download(mocker, tmp_path, checksum):
    hosts_file = tmp_path / '.esmvaltool' / 'cache' / 'esgf-hosts.yml'
    mocker.patch.object(_download, 'HOSTS_FILE', hosts_file)

    credentials = '/path/to/creds.pem'
    mocker.patch.object(_download,
                        'get_credentials',
                        autospec=True,
                        return_value=credentials)

    response = mocker.create_autospec(requests.Response,
                                      spec_set=True,
                                      instance=True)
    response.iter_content.return_value = [b'chunk1', b'chunk2']
    get = mocker.patch.object(_download.requests,
                              'get',
                              autospec=True,
                              return_value=response)

    dest_folder = tmp_path
    filename = 'abc_2000-2001.nc'
    dataset = 'CMIP6.ABC.v1'
    url = f'http://something.org/CMIP6/ABC/v1/{filename}'

    json = {
        'dataset_id': f'{dataset}|something.org',
        'project': ['CMIP6'],
        'size': 12,
        'source_id': ['ABC'],
        'title': filename,
        'url': [url + '|application/netcdf|HTTPServer'],
    }
    if checksum == 'yes':
        json['checksum'] = ['097c42989a9e5d9dcced7b35ec4b0486']
        json['checksum_type'] = ['MD5']
    if checksum == 'wrong':
        json['checksum'] = ['123']
        json['checksum_type'] = ['MD5']

    file = _download.ESGFFile([FileResult(json=json, context=None)])

    if checksum == 'wrong':
        with pytest.raises(_download.DownloadError,
                           match='Wrong MD5 checksum'):
            file.download(dest_folder)
        return

    # Add a second url and check that it is not used.
    file.urls.append('http://wrong_url.com')

    local_file = file.download(dest_folder)

    assert local_file.exists()

    reference_path = dest_folder / 'CMIP6' / 'ABC' / 'v1' / filename
    assert local_file == reference_path

    # File was downloaded only once
    get.assert_called_once()
    # From the correct URL
    get.assert_called_with(url, stream=True, timeout=300, cert=credentials)
    # We checked for a valid response
    response.raise_for_status.assert_called_once()
    # And requested a reasonable chunk size
    response.iter_content.assert_called_with(chunk_size=None)


def test_download_skip_existing(tmp_path, caplog):
    filename = 'test.nc'
    dataset = 'dataset'
    dest_folder = tmp_path

    json = {
        'dataset_id': f'{dataset}|something.org',
        'project': ['CMIP6'],
        'size': 12,
        'title': filename,
    }
    file = _download.ESGFFile([FileResult(json=json, context=None)])

    # Create local file
    local_file = file.local_file(dest_folder)
    local_file.parent.mkdir()
    local_file.touch()

    caplog.set_level(logging.DEBUG)

    local_file = file.download(dest_folder)

    assert f"Skipping download of existing file {local_file}" in caplog.text


def test_single_download_fail(mocker, tmp_path):
    hosts_file = tmp_path / '.esmvaltool' / 'cache' / 'esgf-hosts.yml'
    mocker.patch.object(_download, 'HOSTS_FILE', hosts_file)

    response = mocker.create_autospec(requests.Response,
                                      spec_set=True,
                                      instance=True)
    response.raise_for_status.side_effect = (
        requests.exceptions.RequestException("test error"))
    mocker.patch.object(_download.requests,
                        'get',
                        autospec=True,
                        return_value=response)

    filename = 'test.nc'
    dataset = 'dataset'
    dest_folder = tmp_path
    url = f'http://something.org/CMIP6/ABC/v1/{filename}'

    json = {
        'dataset_id': f'{dataset}|something.org',
        'project': ['CMIP6'],
        'size': 12,
        'title': filename,
        'url': [url + '|application/netcdf|HTTPServer'],
    }
    file = _download.ESGFFile([FileResult(json=json, context=None)])
    local_file = file.local_file(dest_folder)
    msg = (f"Failed to download file {local_file}, errors:"
           "\n" + f"{url}: test error")
    with pytest.raises(_download.DownloadError, match=re.escape(msg)):
        file.download(dest_folder)


def test_get_download_message():

    result1 = FileResult(
        json={
            'dataset_id': 'ABC.v1|something.org',
            'project': ['CMIP6'],
            'size': 4 * 10**9,
            'title': 'abc_1850-1900.nc',
            'url': ['http://xyz.org/file1.nc|application/netcdf|HTTPServer'],
        },
        context=None,
    )
    result2 = FileResult(
        json={
            'dataset_id': 'ABC.v1|something.org',
            'project': ['CMIP6'],
            'size': 6 * 10**9,
            'title': 'abc_1900-1950.nc',
            'url': ['http://abc.com/file2.nc|application/netcdf|HTTPServer'],
        },
        context=None,
    )
    files = [_download.ESGFFile([r]) for r in (result1, result2)]
    msg = _download.get_download_message(files)
    expected = textwrap.dedent("""
        Will download 10 GB
        Will download the following files:
        4 GB\tESGFFile:ABC/v1/abc_1850-1900.nc on hosts ['xyz.org']
        6 GB\tESGFFile:ABC/v1/abc_1900-1950.nc on hosts ['abc.com']
        Downloading 10 GB..
        """).strip()
    assert msg == expected


def test_download(mocker, tmp_path, caplog):
    """Test `esmvalcore.esgf.download`."""
    dest_folder = tmp_path
    test_files = [
        mocker.create_autospec(esmvalcore.esgf.ESGFFile, instance=True)
        for _ in range(5)
    ]
    for i, file in enumerate(test_files):
        file.__str__.return_value = f'file{i}.nc'
        file.size = 200 * 10**6
        file.__lt__.return_value = False

    caplog.set_level(logging.INFO)
    esmvalcore.esgf.download(test_files, dest_folder)

    for file in test_files:
        file.download.assert_called_with(dest_folder)

    print(caplog.text)
    assert "Downloaded 1 GB" in caplog.text


def test_download_fail(mocker, tmp_path, caplog):
    """Test `esmvalcore.esgf.download`."""
    dest_folder = tmp_path
    test_files = [
        mocker.create_autospec(esmvalcore.esgf.ESGFFile, instance=True)
        for _ in range(5)
    ]
    for i, file in enumerate(test_files):
        file.__str__.return_value = f'file{i}.nc'
        file.size = 100 * 10**6
        file.__lt__.return_value = False

    # Fail some files
    error0 = "error messages for first file"
    error1 = "error messages for third file"
    test_files[0].download.side_effect = _download.DownloadError(error0)
    test_files[2].download.side_effect = _download.DownloadError(error1)
    msg = textwrap.dedent("""
        Failed to download the following files:
        error messages for first file
        error messages for third file
        """).strip()
    with pytest.raises(_download.DownloadError, match=re.escape(msg)):
        esmvalcore.esgf.download(test_files, dest_folder)
    assert error0 in caplog.text
    assert error1 in caplog.text
    for file in test_files:
        file.download.assert_called_with(dest_folder)


def test_download_noop(caplog):
    """Test downloading no files."""
    caplog.set_level('INFO')
    esmvalcore.esgf.download([], dest_folder='/does/not/exist')

    msg = ("All required data is available locally,"
           " not downloading anything.")
    assert msg in caplog.text
