import logging
from pathlib import Path

import pytest

from esmvalcore.esgf import search
from esmvalcore.preprocessor import download

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
    'time_frequency': 'mon',
    'project': 'obs4MIPs',
    'short_name': 'rsutcs',
}]


@pytest.mark.skip(reason="This will actually search the ESGF.")
def test_search():
    results = [
        ['tas_a1_20c3m_1_cgcm3.1_t47_1850_2000.nc'],
        [
            'tas_Amon_inmcm4_historical_r1i1p1_185001-200512.nc',
            'tas_Amon_inmcm4_rcp85_r1i1p1_200601-210012.nc',
        ],
        [
            'tas_Amon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_200001-200012.nc',
            'tas_Amon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_200101-200112.nc',
        ],
        [
            'tas_EUR-11_MOHC-HadGEM2-ES_historical_r1i1p1'
            '_KNMI-RACMO22E_v2_mon_195001-195012.nc',
            'tas_EUR-11_MOHC-HadGEM2-ES_historical_r1i1p1'
            '_KNMI-RACMO22E_v2_mon_195101-196012.nc',
        ],
        ['rsutcs_CERES-EBAF_L3B_Ed2-8_200003-201404.nc'],
    ]

    for variable, result in zip(VARIABLES, results):
        files = search(**variable)
        print(files)
        filenames = [file.name for file in files]
        assert filenames == result


@pytest.mark.skip(reason="This will try to download data")
def test_download():
    for variable in VARIABLES:
        variable['start_year'] = 2000
        variable['end_year'] = 2001
        files = search(**variable)
        dest_folder = Path.home() / 'esmvaltool_download_test'
        local_files = download(files, dest_folder)
        assert local_files
        for file in local_files:
            assert Path(file).is_file()
        print(f"Download successful, {local_files=}")


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s [%(process)d] %(levelname)-8s "
                        "%(name)s,%(lineno)s\t%(message)s")
    logging.getLogger().setLevel('info'.upper())

    test_search()
    # test_download()
