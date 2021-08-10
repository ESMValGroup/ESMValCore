import logging
from pathlib import Path

import pytest

from esmvalcore.esgf import find_files
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
    'exp': 'historical',
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


@pytest.mark.skip(reason="This will actually search the ESGF.")
def test_search():
    results = {
        'cmip3.CCCma.cccma_cgcm3_1.historical.mon.atmos.run1.tas.v1': [
            'tas_a1_20c3m_1_cgcm3.1_t47_1850_2000.nc',
        ],
        'cmip5.output1.INM.inmcm4.historical.mon.atmos.Amon.r1i1p1.v20130207':
        [
            'tas_Amon_inmcm4_historical_r1i1p1_185001-200512.nc',
        ],
        'cmip5.output1.FIO.FIO-ESM.historical.mon.atmos.Amon.r1i1p1.v20121010':
        [
            'tas_Amon_FIO-ESM_historical_r1i1p1_185001-200512.nc',
        ],
        'cmip5.output1.MOHC.HadGEM2-CC.rcp85.mon.atmos.Amon.r1i1p1.v20120531':
        [
            'tas_Amon_HadGEM2-CC_rcp85_r1i1p1_205512-208011.nc',
            'tas_Amon_HadGEM2-CC_rcp85_r1i1p1_208012-209912.nc',
            'tas_Amon_HadGEM2-CC_rcp85_r1i1p1_210001-210012.nc',
        ],
        'CMIP6.CMIP.AWI.AWI-ESM-1-1-LR.historical'
        '.r1i1p1f1.Amon.tas.gn.v20200212': [
            'tas_Amon_AWI-ESM-1-1-LR_historical_'
            'r1i1p1f1_gn_200001-200012.nc',
            'tas_Amon_AWI-ESM-1-1-LR_historical_'
            'r1i1p1f1_gn_200101-200112.nc',
        ],
        'cordex.output.EUR-11.KNMI.MOHC-HadGEM2-ES.historical'
        '.r1i1p1.RACMO22E.v2.mon.tas.v20160620': [
            'tas_EUR-11_MOHC-HadGEM2-ES_historical_r1i1p1'
            '_KNMI-RACMO22E_v2_mon_195001-195012.nc',
            'tas_EUR-11_MOHC-HadGEM2-ES_historical_r1i1p1'
            '_KNMI-RACMO22E_v2_mon_195101-196012.nc',
        ],
        'obs4MIPs.CERES-EBAF.v20160610': [
            'rsutcs_CERES-EBAF_L3B_Ed2-8_200003-201404.nc',
        ],
    }

    for variable, dataset in zip(VARIABLES, results):
        files = find_files(**variable)
        assert files
        for file in files:
            print(file.dataset, file.name)
            assert file.dataset == dataset
        filenames = [file.name for file in files]
        assert filenames == results[dataset]


@pytest.mark.skip(reason="This will try to download data")
def test_download():
    for variable in VARIABLES:
        if variable['exp'] == 'historical':
            variable['start_year'] = 2000
            variable['end_year'] = 2001
        files = find_files(**variable)
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
