"""Test 1esmvalcore.esgf._search`."""
import copy
import textwrap

import pyesgf.search
import pytest
import requests.exceptions
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
    {
        'dataset': 'CERES-EBAF',
        'frequency': '*',
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
        'member_id': 'r1i1p1f1',
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
    {
        'project': 'obs4MIPs',
        'source_id': 'CERES-EBAF',
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


def get_mock_connection(mocker, search_results):
    """Create a mock pyesgf.search.SearchConnection class."""
    cfg = {
        'search_connection': {
            'urls': [
                'https://esgf-index1.example.com/esg-search',
                'https://esgf-index2.example.com/esg-search',
            ]
        },
    }
    mocker.patch.object(_search, "get_esgf_config", return_value=cfg)
    mocker.patch.object(_search, 'FIRST_ONLINE_INDEX_NODE', None)

    ctx = mocker.create_autospec(
        pyesgf.search.context.FileSearchContext,
        spec_set=True,
        instance=True,
    )
    ctx.search.side_effect = search_results
    conn_cls = mocker.patch.object(
        _search.pyesgf.search,
        'SearchConnection',
        autospec=True,
    )
    conn_cls.return_value.new_context.return_value = ctx
    return conn_cls, ctx


def test_esgf_search_files(mocker):

    # Set up some fake FileResults
    dataset_id = ('cmip5.output1.INM.inmcm4.historical'
                  '.mon.atmos.Amon.r1i1p1.v20130207')
    dataset_id_template = (
        'cmip5.%(product)s.%(valid_institute)s.%(model)s.'
        '%(experiment)s.%(time_frequency)s.%(realm)s.%(cmor_table)s.'
        '%(ensemble)s'
    )
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
            'dataset_id_template_': [dataset_id_template],
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
            'dataset_id_template_': [dataset_id_template],
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
            'dataset_id_template_': [dataset_id_template],
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

    SearchConnection, context = get_mock_connection(  # noqa: N806
        mocker, search_results=[file_results])

    files = _search.esgf_search_files(facets)

    SearchConnection.assert_called_once_with(
        url='https://esgf-index1.example.com/esg-search')
    connection = SearchConnection.return_value
    connection.new_context.assert_called_with(
        pyesgf.search.context.FileSearchContext,
        **facets,
    )
    context.search.assert_called_with(
        batch_size=500,
        ignore_facet_check=True,
    )

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


def test_esgf_search_uses_second_index_node(mocker):
    """Test that the second index node is used if the first is offline."""
    search_result = [mocker.sentinel.search_result]
    search_results = [
        requests.exceptions.ReadTimeout("Timeout error message"),
        search_result,
    ]
    SearchConnection, context = get_mock_connection(  # noqa: N806
        mocker, search_results)

    result = _search._search_index_nodes(facets={})

    second_index_node = 'https://esgf-index2.example.com/esg-search'
    assert _search.FIRST_ONLINE_INDEX_NODE == second_index_node
    assert result == search_result


def test_esgf_search_fails(mocker):
    """Test that FileNotFoundError is raised if all index nodes are offline."""
    search_results = [
        requests.exceptions.ReadTimeout("Timeout error message 1"),
        requests.exceptions.ConnectTimeout("Timeout error message 2"),
    ]
    SearchConnection, context = get_mock_connection(  # noqa: N806
        mocker, search_results)

    with pytest.raises(FileNotFoundError) as excinfo:
        _search.esgf_search_files(facets={})
    error_message = textwrap.dedent("""
    Failed to search ESGF, unable to connect:
    - Timeout error message 1
    - Timeout error message 2
    """).strip()
    assert str(excinfo.value) == error_message


def test_select_latest_versions_filenotfound(mocker):
    """Test `select_latest_versions` raises FileNotFoundError."""
    file = mocker.create_autospec(ESGFFile, instance=True)
    file.name = 'ta.nc'
    file.dataset = 'CMIP6.MODEL.v1'
    file.facets = {'version': 'v1'}
    file.__repr__ = lambda _: 'ESGFFile:CMIP6/MODEL/v1/ta.nc'
    result = _search.select_latest_versions(files=[file], versions='v2')
    assert result == []


@pytest.mark.parametrize('timerange,selection', [
    ('1851/1852', slice(1, 3)),
    ('1851/P1Y', slice(1, 3)),
    ('*', slice(None)),
])
def test_select_by_time(timerange, selection):
    dataset_id = ('CMIP6.CMIP.AWI.AWI-ESM-1-1-LR.historical'
                  '.r1i1p1f1.Amon.tas.gn.v20200212')
    dataset_id_template = (
        '%(mip_era)s.%(activity_drs)s.%(institution_id)s.'
        '%(source_id)s.%(experiment_id)s.%(member_id)s.%(table_id)s.'
        '%(variable_id)s.%(grid_label)s'
    )
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
                'dataset_id_template_': [dataset_id_template],
                'project': ['CMIP5'],
                'size': 100,
            },
            context=None,
        ) for filename in filenames
    ]
    files = [ESGFFile([r]) for r in results]

    result = _search.select_by_time(files, timerange)
    reference = files[selection]
    assert sorted(result) == sorted(reference)


def test_select_by_time_nodate():
    dataset_id = (
        'cmip3.MIROC.miroc3_2_hires.historical.mon.atmos.run1.tas.v1')
    dataset_id_template = (
        '%(project)s.%(institute)s.%(model)s.%(experiment)s.'
        '%(time_frequency)s.%(realm)s.%(ensemble)s.%(variable)s'
    )
    filenames = ['tas_A1.nc']
    results = [
        FileResult(
            json={
                'title': filename,
                'dataset_id': dataset_id + '|xyz.com',
                'dataset_id_template_': [dataset_id_template],
                'project': ['CMIP5'],
                'size': 100,
            },
            context=None,
        ) for filename in filenames
    ]
    files = [ESGFFile([r]) for r in results]

    result = _search.select_by_time(files, '1851/1852')
    assert result == files


def test_invalid_dataset_id_template():
    dataset_id = (
        'obs4MIPs.IUP.XCH4_CRDP3.xch4.mon.v100')
    dataset_id_template = (
        '%(project)s.%(institute)s.%(source_id)s.%(time_frequency)s'
    )
    filenames = ['xch4_ghgcci_l3_v100_200301_201412.nc']
    results = [
        FileResult(
            json={
                'title': filename,
                'dataset_id': dataset_id + '|esgf.ceda.ac.uk',
                'dataset_id_template_': [dataset_id_template],
                'project': ['obs4MIPs'],
                'size': 100,
                'source_id': 'XCH4_CRDP3',
            },
            context=None,
        ) for filename in filenames
    ]
    file = ESGFFile(results)

    assert file.name == filenames[0]


def test_search_unknown_project():
    project = 'Unknown'
    msg = (f"Unable to download from ESGF, because project {project} is not on"
           " it or is not supported by the esmvalcore.esgf module.")
    with pytest.raises(ValueError, match=msg):
        find_files(project=project, dataset='', short_name='')
