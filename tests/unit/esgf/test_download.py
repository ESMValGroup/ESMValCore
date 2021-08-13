"""Test `esmvalcore.esgf._download`."""
import logging
import re
import textwrap
from pathlib import Path

import pytest
import requests
from pyesgf.search.results import FileResult

from esmvalcore.esgf import _download


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
                        'load_esgf_pyclient_config',
                        autospec=True,
                        return_value={'preferred_hosts': preferred_hosts})
    sorted_urls = _download.sort_hosts(urls)
    assert sorted_urls == [
        'http://esgf2.dkrz.de/abc.nc',
        'http://esgf-data1.ceda.ac.uk/abc.nc',
        'http://esgf.nci.org.au/abc.nc',
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
    filename = 'abc_2000-2001.nc'
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
def test_download(mocker, tmp_path, checksum):
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
    filename = 'abc_2000-2001.txt'
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
        with pytest.raises(ValueError, match='Wrong MD5 checksum'):
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
    get.assert_called_with(url, timeout=300, cert=credentials)
    # We checked for a valid response
    response.raise_for_status.assert_called_once()
    # And requested a reasonable chunk size
    response.iter_content.assert_called_with(chunk_size=2**20)


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

    caplog.set_level(logging.INFO)

    local_file = file.download(dest_folder)

    assert f"Skipping download of existing file {local_file}" in caplog.text


def test_download_fail(mocker, tmp_path):

    response = mocker.create_autospec(requests.Response,
                                      spec_set=True,
                                      instance=True)
    response.raise_for_status.side_effect = (
        requests.exceptions.RequestException)
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

    msg = f"Failed to download file {local_file} from {[url]}"
    with pytest.raises(IOError, match=re.escape(msg)):
        file.download(dest_folder)


def test_get_download_message():

    result1 = FileResult(
        json={
            'dataset_id': 'ABC.v1|something.org',
            'project': ['CMIP6'],
            'size': 4 * 2**30,
            'title': 'abc_1850-1900.nc',
            'url': ['http://xyz.org/file1.nc|application/netcdf|HTTPServer'],
        },
        context=None,
    )
    result2 = FileResult(
        json={
            'dataset_id': 'ABC.v1|something.org',
            'project': ['CMIP6'],
            'size': 6 * 2**30,
            'title': 'abc_1900-1950.nc',
            'url': ['http://abc.com/file2.nc|application/netcdf|HTTPServer'],
        },
        context=None,
    )
    files = [_download.ESGFFile([r]) for r in (result1, result2)]
    msg = _download.get_download_message(files)
    expected = textwrap.dedent("""
        Will download 10.0 GB
        Will download the following files:
        4096 MB\tESGFFile:ABC/v1/abc_1850-1900.nc on hosts ['xyz.org']
        6144 MB\tESGFFile:ABC/v1/abc_1900-1950.nc on hosts ['abc.com']
        """).strip()
    assert msg == expected
