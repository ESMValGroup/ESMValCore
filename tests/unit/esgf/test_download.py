"""Test 1esmvalcore.esgf._download`."""
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


def test_simplify_dataset_id_noop():
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


def test_simplify_dataset_id_obs4mips():
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
