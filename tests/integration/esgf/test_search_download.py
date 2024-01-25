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
    'version': 'v1',
}, {
    'dataset': 'inmcm4',
    'ensemble': 'r1i1p1',
    'exp': ['historical', 'rcp85'],
    'mip': 'Amon',
    'project': 'CMIP5',
    'short_name': 'tas',
    'version': 'v20130207',
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
    'timerange': '2080/2100',
}, {
    'dataset': 'EC-EARTH',
    'ensemble': 'r1i1p1',
    'exp': 'historical',
    'mip': 'Amon',
    'project': 'CMIP5',
    'short_name': 'tas',
    'start_year': 1990,  # test legacy way of specifying timerange
    'end_year': 1999,
}, {
    'dataset': 'AWI-ESM-1-1-LR',
    'ensemble': 'r1i1p1f1',
    'exp': 'historical',
    'grid': 'gn',
    'mip': 'Amon',
    'project': 'CMIP6',
    'short_name': 'tas',
    'timerange': '2000/2001',
    'version': 'v20200212',
}, {
    'dataset': 'CESM2',
    'ensemble': 'r4i1p1f1',
    'exp': 'historical',
    'grid': 'gn',
    'mip': 'Amon',
    'project': 'CMIP6',
    'short_name': 'tas',
    'timerange': '2000/2001',
}, {
    'dataset': 'RACMO22E',
    'driver': 'MOHC-HadGEM2-ES',
    'domain': 'EUR-11',
    'ensemble': 'r1i1p1',
    'exp': 'historical',
    'frequency': 'mon',
    'project': 'CORDEX',
    'short_name': 'tas',
    'timerange': '1950/1952',
    'version': 'v20160620',
}, {
    'dataset': 'CERES-EBAF',
    'frequency': 'mon',
    'project': 'obs4MIPs',
    'short_name': 'rsutcs',
    'version': 'v20160610',
}, {
    'dataset': 'GPCP-V2.3',
    'project': 'obs4MIPs',
    'short_name': 'pr',
}]


def get_mock_connection(facets, results):
    """Create a mock pyesgf.search.SearchConnection instance."""
    class MockFileSearchContext:
        def search(self, **kwargs):
            return results

    class MockConnection:
        def new_context(self, *args, **kwargs):
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

    with raw_results.open('r', encoding='utf-8') as file:
        search_results = [
            FileResult(json=j, context=None) for j in json.load(file)
        ]
    conn = get_mock_connection(facets, search_results)
    mocker.patch.object(_search.pyesgf.search,
                        'SearchConnection',
                        autspec=True,
                        return_value=conn)

    files = find_files(**variable)

    expected_results_file = data_path / 'expected.yml'
    if expected_results_file.exists():
        with expected_results_file.open(encoding='utf-8') as file:
            expected_results = yaml.safe_load(file)
    else:
        expected_results = {}
    print(expected_results)
    if json_file in expected_results:
        expected_files = expected_results[json_file]
    else:
        expected_results[json_file] = [
            {
                'checksums': file._checksums,
                'dataset': file.dataset,
                'facets': file.facets,
                'local_file': str(file.local_file(Path())),
                'name': file.name,
                'size': file.size,
                'urls': file.urls,
            }
            for file in files
        ]
        with expected_results_file.open('w', encoding='utf-8') as file:
            yaml.safe_dump(expected_results, file)

        assert False, 'Wrote expected results, please check.'

    assert len(files) == len(expected_files)
    for found_file, expected in zip(files, expected_files):
        assert found_file.name == expected['name']
        assert found_file.local_file(Path()) == Path(expected['local_file'])
        assert found_file.dataset == expected['dataset']
        assert found_file.size == expected['size']
        assert found_file.facets == expected['facets']
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
            'tas_Amon_CESM2_historical_r4i1p1f1_gn_185001-201412.nc',
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
        [
            'pr_GPCP-SG_L3_v2.3_197901-201710.nc',
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
            '.v20131231',
        ],
        [
            'CMIP6.CMIP.AWI.AWI-ESM-1-1-LR.historical.r1i1p1f1.Amon.tas.gn'
            '.v20200212',
            'CMIP6.CMIP.AWI.AWI-ESM-1-1-LR.historical.r1i1p1f1.Amon.tas.gn'
            '.v20200212',
        ],
        [
            'CMIP6.CMIP.NCAR.CESM2.historical.r4i1p1f1.Amon.tas.gn.v20190308',
        ],
        [
            'cordex.output.EUR-11.KNMI.MOHC-HadGEM2-ES.historical.r1i1p1'
            '.RACMO22E.v2.mon.tas.v20160620',
            'cordex.output.EUR-11.KNMI.MOHC-HadGEM2-ES.historical.r1i1p1'
            '.RACMO22E.v2.mon.tas.v20160620',
        ],
        [
            'obs4MIPs.CERES-EBAF.v20160610',
        ],
        ['obs4MIPs.GPCP-V2.3.v20180519'],
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
        print(result[0].facets)
        for file in result:
            for key, value in variable.items():
                if key in ('start_year', 'end_year', 'timerange'):
                    continue
                if isinstance(value, list):
                    assert file.facets.get(key) in value
                else:
                    assert file.facets.get(key) == value


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
