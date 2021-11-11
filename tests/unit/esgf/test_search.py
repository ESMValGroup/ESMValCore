"""Test 1esmvalcore.esgf._search`."""
import copy

import pytest
from pyesgf.search.context import FileSearchContext
from pyesgf.search.results import FileResult

from esmvalcore.esgf import ESGFFile, _search, find_files

OUR_FACETS = (
    {
        'dataset': 'cccma_cgcm3_1',
        'ensemble': 'run1',
        'exp': 'historical',
        'frequency': 'mon',
        'project': 'CMIP3',
        'short_name': 'tas',
    },
    {
        'dataset': 'inmcm4',
        'ensemble': 'r1i1p1',
        'exp': ['historical', 'rcp85'],
        'mip': 'Amon',
        'project': 'CMIP5',
        'short_name': 'tas',
    },
    {
        'dataset': 'AWI-ESM-1-1-LR',
        'ensemble': 'r1i1p1f1',
        'exp': 'historical',
        'grid': 'gn',
        'mip': 'Amon',
        'project': 'CMIP6',
        'short_name': 'tas',
        'start_year': 2000,
        'end_year': 2001,
    },
    {
        'dataset': 'RACMO22E',
        'driver': 'MOHC-HadGEM2-ES',
        'domain': 'EUR-11',
        'ensemble': 'r1i1p1',
        'exp': 'historical',
        'frequency': 'mon',
        'project': 'CORDEX',
        'short_name': 'tas',
        'start_year': 1950,
        'end_year': 1952,
    },
    {
        'dataset': 'CERES-EBAF',
        'frequency': 'mon',
        'project': 'obs4MIPs',
        'short_name': 'rsutcs',
    },
)

ESGF_FACETS = (
    {
        'project': 'CMIP3',
        'model': 'cccma_cgcm3_1',
        'ensemble': 'run1',
        'experiment': 'historical',
        'time_frequency': 'mon',
        'variable': 'tas',
    },
    {
        'project': 'CMIP5',
        'model': 'INM-CM4',
        'ensemble': 'r1i1p1',
        'experiment': 'historical,rcp85',
        'cmor_table': 'Amon',
        'variable': 'tas',
    },
    {
        'project': 'CMIP6',
        'source_id': 'AWI-ESM-1-1-LR',
        'variant_label': 'r1i1p1f1',
        'experiment_id': 'historical',
        'grid_label': 'gn',
        'table_id': 'Amon',
        'variable': 'tas',
    },
    {
        'project': 'CORDEX',
        'rcm_name': 'RACMO22E',
        'driving_model': 'MOHC-HadGEM2-ES',
        'domain': 'EUR-11',
        'ensemble': 'r1i1p1',
        'experiment': 'historical',
        'time_frequency': 'mon',
        'variable': 'tas',
    },
    {
        'project': 'obs4MIPs',
        'source_id': 'CERES-EBAF',
        'time_frequency': 'mon',
        'variable': 'rsutcs',
    },
)


@pytest.mark.parametrize('our_facets, esgf_facets',
                         zip(OUR_FACETS, ESGF_FACETS))
def test_get_esgf_facets(our_facets, esgf_facets):
    """Test that facet translation by get_esgf_facets works as expected."""
    our_facets = copy.deepcopy(our_facets)
    for facet, value in our_facets.items():
        if isinstance(value, list):
            our_facets[facet] = tuple(value)
    facets = _search.get_esgf_facets(our_facets)
    assert facets == esgf_facets


def get_mock_connection(facets, results):
    """Create a mock pyesgf.search.SearchConnection instance."""
    class MockFileSearchContext:
        def search(self, **kwargs):
            assert kwargs['batch_size'] == 500
            # enable ignore_facet_check once the following issue has been
            # fixed: https://github.com/ESGF/esgf-pyclient/issues/75
            # assert kwargs['ignore_facet_check']
            assert len(kwargs) == 1
            return results

    class MockConnection:
        def new_context(self, *args, **kwargs):
            assert len(args) == 1
            assert args[0] == FileSearchContext
            assert kwargs.pop('latest')
            assert kwargs == facets
            return MockFileSearchContext()

    return MockConnection()


def test_esgf_search_files(mocker):

    # Set up some fake FileResults
    dataset_id = ('cmip5.output1.INM.inmcm4.historical'
                  '.mon.atmos.Amon.r1i1p1.v20130207')
    filename0 = 'tas_Amon_inmcm4_historical_r1i1p1_185001-189912.nc'
    filename1 = 'tas_Amon_inmcm4_historical_r1i1p1_190001-200512.nc'

    aims_url0 = ('http://aims3.llnl.gov/thredds/fileServer/cmip5_css02_data/'
                 'cmip5/output1/INM/inmcm4/historical/mon/atmos/Amon/r1i1p1/'
                 'tas/1/' + filename0)
    aims_url1 = ('http://aims3.llnl.gov/thredds/fileServer/cmip5_css02_data/'
                 'cmip5/output1/INM/inmcm4/historical/mon/atmos/Amon/r1i1p1/'
                 'tas/1/' + filename1)
    dkrz_url = ('http://esgf2.dkrz.de/thredds/fileServer/lta_dataroot/cmip5/'
                'output1/INM/inmcm4/historical/mon/atmos/Amon/r1i1p1/'
                'v20130207/tas/' + filename0)

    file_aims0 = FileResult(
        {
            'checksum': ['123'],
            'checksum_type': ['SHA256'],
            'dataset_id': dataset_id + '|aims3.llnl.gov',
            'project': ['CMIP5'],
            'size': 100,
            'title': filename0,
            'url': [
                aims_url0 + '|application/netcdf|HTTPServer',
            ],
        }, None)

    file_aims1 = FileResult(
        {
            'dataset_id': dataset_id + '|aims3.llnl.gov',
            'project': ['CMIP5'],
            'size': 200,
            'title': filename1,
            'url': [
                aims_url1 + '|application/netcdf|HTTPServer',
            ],
        }, None)

    file_dkrz = FileResult(
        {
            'checksum': ['456'],
            'checksum_type': ['MD5'],
            'dataset_id': dataset_id + '|esgf2.dkrz.de',
            'project': ['CMIP5'],
            'size': 100,
            'title': filename0,
            'url': [dkrz_url + '|application/netcdf|HTTPServer'],
        }, None)

    facets = {
        'project': 'CMIP5',
        'model': 'inmcm4',
        'variable': 'tas',
    }
    file_results = [file_aims0, file_aims1, file_dkrz]
    conn = get_mock_connection(facets, file_results)
    mocker.patch.object(_search,
                        'get_connection',
                        autspec=True,
                        return_value=conn)

    files = _search.esgf_search_files(facets)
    print(files)
    assert len(files) == 2

    file0 = files[0]
    assert file0.name == filename0
    assert file0.dataset == dataset_id
    assert file0.size == 100
    assert file0._checksums == [('SHA256', '123'), ('MD5', '456')]
    urls = sorted(file0.urls)
    assert len(urls) == 2
    assert urls[0] == aims_url0
    assert urls[1] == dkrz_url

    file1 = files[1]
    assert file1.name == filename1
    assert file1.dataset == dataset_id
    assert file1.size == 200
    assert file1._checksums == [(None, None)]
    urls = sorted(file1.urls)
    assert len(urls) == 1
    assert urls[0] == aims_url1


def test_select_by_time():
    dataset_id = ('CMIP6.CMIP.AWI.AWI-ESM-1-1-LR.historical'
                  '.r1i1p1f1.Amon.tas.gn.v20200212')
    filenames = [
        'tas_Amon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_185001-185012.nc',
        'tas_Amon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_185101-185112.nc',
        'tas_Amon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_185201-185212.nc',
        'tas_Amon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_185301-185312.nc',
    ]
    results = [
        FileResult(
            json={
                'title': filename,
                'dataset_id': dataset_id + '|xyz.com',
                'project': ['CMIP5'],
                'size': 100,
            },
            context=None,
        ) for filename in filenames
    ]
    files = [ESGFFile([r]) for r in results]

    result = _search.select_by_time(files, 1851, 1852)
    reference = files[1:3]
    assert sorted(result) == sorted(reference)


def test_select_by_time_nodate():
    dataset_id = (
        'cmip3.MIROC.miroc3_2_hires.historical.mon.atmos.run1.tas.v1')
    filenames = ['tas_A1.nc']
    results = [
        FileResult(
            json={
                'title': filename,
                'dataset_id': dataset_id + '|xyz.com',
                'project': ['CMIP5'],
                'size': 100,
            },
            context=None,
        ) for filename in filenames
    ]
    files = [ESGFFile([r]) for r in results]

    result = _search.select_by_time(files, 1851, 1852)
    assert result == files


def test_search_unknown_project():
    project = 'Unknown'
    msg = (f"Unable to download from ESGF, because project {project} is not on"
           " it or is not supported by the esmvalcore.esgf module.")
    with pytest.raises(ValueError, match=msg):
        find_files(project=project, dataset='', short_name='')
