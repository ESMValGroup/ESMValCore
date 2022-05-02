import json
import logging
from pathlib import Path

import pytest
import yaml
from pyesgf.search.results import FileResult

from esmvalcore.esgf import _search, download, find_files

VARIABLES = [{
    'dataset': 'cccma_cgcm3_1',
    'ensemble': 'run1',
    'exp': 'historical',
    'frequency': 'mon',
    'project': 'CMIP3',
    'short_name': 'tas',
}, {
    'dataset': 'inmcm4',
    'ensemble': 'r1i1p1',
    'exp': ['historical', 'rcp85'],
    'mip': 'Amon',
    'project': 'CMIP5',
    'short_name': 'tas',
}, {
    'dataset': 'FIO-ESM',
    'ensemble': 'r1i1p1',
    'exp': 'historical',
    'mip': 'Amon',
    'project': 'CMIP5',
    'short_name': 'tas',
}, {
    'dataset': 'HadGEM2-CC',
    'ensemble': 'r1i1p1',
    'exp': 'rcp85',
    'mip': 'Amon',
    'project': 'CMIP5',
    'short_name': 'tas',
    'start_year': 2080,
    'end_year': 2100,
}, {
    'dataset': 'EC-EARTH',
    'ensemble': 'r1i1p1',
    'exp': 'historical',
    'mip': 'Amon',
    'project': 'CMIP5',
    'short_name': 'tas',
    'start_year': 1990,
    'end_year': 1999,
}, {
    'dataset': 'AWI-ESM-1-1-LR',
    'ensemble': 'r1i1p1f1',
    'exp': 'historical',
    'grid': 'gn',
    'mip': 'Amon',
    'project': 'CMIP6',
    'short_name': 'tas',
    'start_year': 2000,
    'end_year': 2001,
}, {
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
}, {
    'dataset': 'CERES-EBAF',
    'frequency': 'mon',
    'project': 'obs4MIPs',
    'short_name': 'rsutcs',
}]


def get_mock_connection(facets, results):
    """Create a mock pyesgf.search.SearchConnection instance."""
    class MockFileSearchContext:
        def search(self, **kwargs):
            return results

    class MockConnection:
        def new_context(self, *args, **kwargs):
            assert kwargs.pop('latest')
            assert kwargs == facets
            return MockFileSearchContext()

    return MockConnection()


@pytest.mark.parametrize('variable', VARIABLES)
def test_mock_search(variable, mocker):
    data_path = Path(__file__).parent / 'search_results'
    facets = _search.get_esgf_facets(variable)
    json_file = '_'.join(str(facets[k]) for k in sorted(facets)) + '.json'
    raw_results = data_path / json_file

    if not raw_results.exists():
        # Skip cases where the raw search results were too large to save.
        pytest.skip(f"Raw search results in {raw_results} not available.")

    with raw_results.open('r') as file:
        search_results = [
            FileResult(json=j, context=None) for j in json.load(file)
        ]
    conn = get_mock_connection(facets, search_results)
    mocker.patch.object(_search.pyesgf.search,
                        'SearchConnection',
                        autspec=True,
                        return_value=conn)

    files = find_files(**variable)

    with open(data_path / 'expected.yml') as file:
        expected_files = yaml.safe_load(file)[json_file]

    assert len(files) == len(expected_files)
    for found_file, expected in zip(files, expected_files):
        assert found_file.name == expected['name']
        assert found_file.dataset == expected['dataset']
        assert found_file.size == expected['size']
        assert found_file.urls == expected['urls']
        assert found_file._checksums == [
            tuple(c) for c in expected['checksums']
        ]


def test_real_search():
    """Test a real search for a single file."""
    variable = {
        'project': 'CMIP6',
        'mip': 'Amon',
        'short_name': 'tas',
        'dataset': 'EC-Earth3',
        'exp': 'historical',
        'ensemble': 'r1i1p1f1',
        'grid': 'gr',
        'start_year': 1990,
        'end_year': 2000,
    }
    files = find_files(**variable)
    dataset = ('CMIP6.CMIP.EC-Earth-Consortium.EC-Earth3'
               '.historical.r1i1p1f1.Amon.tas.gr')
    assert files
    for file in files:
        assert file.dataset.startswith(dataset)


@pytest.mark.skip(reason="This will actually search the ESGF.")
def test_real_search_many():
    expected_files = [
        [
            'tas_a1_20c3m_1_cgcm3.1_t47_1850_2000.nc',
        ],
        [
            'tas_Amon_inmcm4_historical_r1i1p1_185001-200512.nc',
            'tas_Amon_inmcm4_rcp85_r1i1p1_200601-210012.nc',
        ],
        [
            'tas_Amon_FIO-ESM_historical_r1i1p1_185001-200512.nc',
        ],
        [
            'tas_Amon_HadGEM2-CC_rcp85_r1i1p1_205512-208011.nc',
            'tas_Amon_HadGEM2-CC_rcp85_r1i1p1_208012-209912.nc',
            'tas_Amon_HadGEM2-CC_rcp85_r1i1p1_210001-210012.nc',
        ],
        [
            'tas_Amon_EC-EARTH_historical_r1i1p1_199001-199912.nc',
        ],
        [
            'tas_Amon_AWI-ESM-1-1-LR_historical_'
            'r1i1p1f1_gn_200001-200012.nc',
            'tas_Amon_AWI-ESM-1-1-LR_historical_'
            'r1i1p1f1_gn_200101-200112.nc',
        ],
        [
            'tas_EUR-11_MOHC-HadGEM2-ES_historical_r1i1p1'
            '_KNMI-RACMO22E_v2_mon_195001-195012.nc',
            'tas_EUR-11_MOHC-HadGEM2-ES_historical_r1i1p1'
            '_KNMI-RACMO22E_v2_mon_195101-196012.nc',
        ],
        [
            'rsutcs_CERES-EBAF_L3B_Ed2-8_200003-201404.nc',
        ],
    ]
    expected_datasets = [
        [
            'cmip3.CCCma.cccma_cgcm3_1.historical.mon.atmos.run1.tas.v1',
        ],
        [
            'cmip5.output1.INM.inmcm4.historical.mon.atmos.Amon.r1i1p1'
            '.v20130207',
            'cmip5.output1.INM.inmcm4.rcp85.mon.atmos.Amon.r1i1p1.v20130207',
        ],
        [
            'cmip5.output1.FIO.FIO-ESM.historical.mon.atmos.Amon.r1i1p1'
            '.v20121010',
        ],
        [
            'cmip5.output1.MOHC.HadGEM2-CC.rcp85.mon.atmos.Amon.r1i1p1'
            '.v20120531',
            'cmip5.output1.MOHC.HadGEM2-CC.rcp85.mon.atmos.Amon.r1i1p1'
            '.v20120531',
            'cmip5.output1.MOHC.HadGEM2-CC.rcp85.mon.atmos.Amon.r1i1p1'
            '.v20120531',
        ],
        [
            'cmip5.output1.ICHEC.EC-EARTH.historical.mon.atmos.Amon.r1i1p1'
            '.v20121115',
        ],
        [
            'CMIP6.CMIP.AWI.AWI-ESM-1-1-LR.historical.r1i1p1f1.Amon.tas.gn'
            '.v20200212',
            'CMIP6.CMIP.AWI.AWI-ESM-1-1-LR.historical.r1i1p1f1.Amon.tas.gn'
            '.v20200212',
        ],
        [
            'cordex.output.EUR-11.KNMI.MOHC-HadGEM2-ES.historical.r1i1p1'
            '.RACMO22E.v2.mon.tas.v20160620',
            'cordex.output.EUR-11.KNMI.MOHC-HadGEM2-ES.historical.r1i1p1'
            '.RACMO22E.v2.mon.tas.v20160620',
        ], [
            'obs4MIPs.CERES-EBAF.v20160610',
        ]
    ]

    for variable, files, datasets in zip(VARIABLES, expected_files,
                                         expected_datasets):
        result = find_files(**variable)
        found_files = [file.name for file in result]
        print(found_files)
        print(files)
        assert found_files == files
        found_datasets = [file.dataset for file in result]
        print(found_datasets)
        print(datasets)
        assert found_datasets == datasets


@pytest.mark.skip(reason="This will actually download the data")
def test_real_download():
    all_files = []
    for variable in VARIABLES:
        if variable.get('exp', '') == 'historical':
            variable['start_year'] = 2000
            variable['end_year'] = 2001
        files = find_files(**variable)
        assert files
        all_files.extend(files)

    dest_folder = Path.home() / 'esmvaltool_download_test'
    download(all_files, dest_folder)
    print(f"Download of variable={variable} successful")


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s [%(process)d] %(levelname)-8s "
                        "%(name)s,%(lineno)s\t%(message)s")
    logging.getLogger().setLevel('info'.upper())

    test_real_search_many()
    test_real_download()
