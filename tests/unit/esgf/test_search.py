"""Test 1esmvalcore.esgf._search`."""
import pyesgf.search.results
import pytest

from esmvalcore.esgf import ESGFFile, _search

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
        'experiment': ['historical', 'rcp85'],
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
    facets = _search.get_esgf_facets(our_facets)
    assert facets == esgf_facets


def test_sort_hosts():
    """Test that hosts are sorted according to priority by sort_hosts."""
    hosts = ['esgf.nci.org.au', 'esgf2.dkrz.de', 'esgf-data1.ceda.ac.uk']
    preferred_hosts = [
        'esgf2.dkrz.de', 'esgf-data1.ceda.ac.uk', 'aims3.llnl.gov'
    ]

    result = _search.sort_hosts(hosts, preferred_hosts)
    assert result == [
        'esgf2.dkrz.de', 'esgf-data1.ceda.ac.uk', 'esgf.nci.org.au'
    ]


def test_select_latest_versions():
    datasets = {
        ('cmip5.output1.CSIRO-BOM.ACCESS1-0.historical'
         '.mon.atmos.Amon.r1i1p1.v1'):
        'a',
        ('cmip5.output1.CSIRO-BOM.ACCESS1-0.historical'
         '.mon.atmos.Amon.r1i1p1.v20120329'):
        'b',
        ('cmip5.output1.CSIRO-BOM.ACCESS1-0.historical'
         '.mon.atmos.Amon.r1i1p1.v20120727'):
        'c',
        ('cmip5.output1.INM.inmcm4.historical'
         '.mon.atmos.Amon.r1i1p1.v20130207'):
        'd',
    }
    result = _search.select_latest_versions(datasets)
    assert result == {
        ('cmip5.output1.CSIRO-BOM.ACCESS1-0.historical'
         '.mon.atmos.Amon.r1i1p1.v20120727'):
        'c',
        ('cmip5.output1.INM.inmcm4.historical'
         '.mon.atmos.Amon.r1i1p1.v20130207'):
        'd',
    }


@pytest.mark.parametrize('dataset_id,facets', [
    ('a.b.c.v1', {
        'project': 'CMIP3'
    }),
    ('a.b.c.v1', {
        'project': 'CMIP5'
    }),
    ('a.b.c.v1', {
        'project': 'CMIP6'
    }),
    ('a.b.c.v1', {
        'project': 'CORDEX'
    }),
])
def test_simplify_dataset_id_noop(dataset_id, facets):
    result = _search.simplify_dataset_id(dataset_id, facets)
    assert result == dataset_id


def test_simplify_dataset_id_obs4mips():
    facets = {
        'project': 'obs4MIPs',
        'source_id': 'CERES-EBAF',
    }
    dataset_id = 'obs4MIPs.NASA-LaRC.CERES-EBAF.atmos.mon.v20160610'
    result = _search.simplify_dataset_id(dataset_id, facets)
    assert result == 'obs4MIPs.CERES-EBAF.v20160610'


def test_find_files():

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

    file_aims0 = pyesgf.search.results.FileResult(
        {
            'checksum': ['123'],
            'checksum_type': ['SHA256'],
            'size': 100,
            'title': filename0,
            'url': [
                aims_url0 + '|application/netcdf|HTTPServer',
            ],
        }, None)

    file_aims1 = pyesgf.search.results.FileResult(
        {
            'checksum': ['234'],
            'checksum_type': ['SHA256'],
            'size': 200,
            'title': filename1,
            'url': [
                aims_url1 + '|application/netcdf|HTTPServer',
            ],
        }, None)

    file_dkrz = pyesgf.search.results.FileResult(
        {
            'checksum': ['456'],
            'checksum_type': ['MD5'],
            'size': 100,
            'title': filename0,
            'url': [dkrz_url + '|application/netcdf|HTTPServer'],
        }, None)

    class MockFileSearchContext:
        def __init__(self, results):
            self.results = results

        def search(self, **kwargs):
            assert kwargs == {'variable': 'tas'}
            return self.results

    class MockDatasetResult:
        def __init__(self, results):
            self.results = results

        def file_context(self):
            return MockFileSearchContext(self.results)

    datasets = {
        dataset_id: {
            'aims3.llnl.gov': MockDatasetResult([file_aims0, file_aims1]),
            'esgf2.dkrz.de': MockDatasetResult([file_dkrz]),
        }
    }

    facets = {
        'project': 'CMIP5',
        'variable': 'tas',
    }

    files = _search.find_files(datasets, facets)
    assert len(files) == 1

    dataset_files = files[dataset_id]
    assert len(dataset_files) == 2

    file0 = dataset_files[0]
    assert file0.name == filename0
    assert file0.dataset == dataset_id
    assert file0.size == 100
    assert file0._checksum_type == 'SHA256'
    assert file0._checksum == '123'
    urls = sorted(file0.urls)
    assert len(urls) == 2
    assert urls[0] == aims_url0
    assert urls[1] == dkrz_url

    file1 = dataset_files[1]
    assert file1.name == filename1
    assert file1.dataset == dataset_id
    assert file1.size == 200
    assert file1._checksum_type == 'SHA256'
    assert file1._checksum == '234'
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
        'tas_Amon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_185301-185412.nc',
    ]
    files = [
        ESGFFile(urls=[],
                 dataset=dataset_id,
                 name=filename,
                 size=0,
                 checksum='123',
                 checksum_type='MD5') for filename in filenames
    ]

    result = _search.select_by_time(files, 1851, 1855)
    reference = files[1:3] + files[4:]
    assert sorted(result) == sorted(reference)


def test_expand_facets():
    """Test that facets that are a tuple are correctly expanded."""
    facets = {
        'a': 1,
        'b': 2,
        'c': (3, 4),
        'd': (5, 6),
    }
    result = _search.expand_facets(facets)

    reference = [
        {
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 5,
        },
        {
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 6,
        },
        {
            'a': 1,
            'b': 2,
            'c': 4,
            'd': 5,
        },
        {
            'a': 1,
            'b': 2,
            'c': 4,
            'd': 6,
        },
    ]
    assert result == reference
