from pathlib import Path

from esmvalcore.preprocessor._download import ESGFFile, download, esgf_search

VARIABLES = [
    {
        'dataset': 'cccma_cgcm3_1',
        'ensemble': 'run1',
        'exp': 'historical',
        'frequency': 'mon',
        'project': 'CMIP3',
        'short_name': 'tas',
        # 'version': '1',
    },
    {
        'dataset': 'CanESM2',
        'ensemble': 'r1i1p1',
        'exp': 'historical',
        'mip': 'Amon',
        'project': 'CMIP5',
        'short_name': 'tas',
        # 'version': '20120718',
    },
    {
        'activity': 'CMIP',
        'dataset': 'AWI-ESM-1-1-LR',
        'ensemble': 'r1i1p1f1',
        'exp': 'historical',
        'grid': 'gn',
        'mip': 'Amon',
        'project': 'CMIP6',
        'short_name': 'tas',
        # 'version': '20200212',
    },
    {
        'dataset': 'COSMO-crCLIM-v1-1',
        'driver': 'MPI-M-MPI-ESM-LR',
        'domain': 'EUR-11',
        'ensemble': 'r1i1p1',
        'exp': 'historical',
        'frequency': 'mon',
        'project': 'CORDEX',
        'short_name': 'tas',
        # 'version': '20160610',
    },
    {
        'source_id': 'CERES-EBAF',
        'time_frequency': 'mon',
        'project': 'obs4MIPs',
        'short_name': 'rsutcs',
        # 'version': '20160610',
    }
]


def test_esgf_search():
    for variable in VARIABLES:
        variable['start_year'] = 1980
        variable['end_year'] = 2000
        files = esgf_search(variable)
        print(files)


def test_esgf_download():
    dataset = 'cmip5.output1.CCCma.CanESM2.historical.mon.atmos.Amon.r1i1p1.v20120718'
    url = 'http://esgf2.dkrz.de/thredds/fileServer/lta_dataroot/cmip5/output1/CCCma/CanESM2/historical/mon/atmos/Amon/r1i1p1/v20120718/tas/tas_Amon_CanESM2_historical_r1i1p1_185001-200512.nc'
    files = [ESGFFile(url, dataset)]

    dest_folder = str(Path.home() / 'esmvaltool_download')
    local_file = download(files, dest_folder)
    print(local_file)


def test_download_all():
    for variable in VARIABLES:
        if variable['project'] == 'CORDEX':
            continue
        variable['start_year'] = 2000
        variable['end_year'] = 2001
        files = esgf_search(variable)
        dest_folder = str(Path.home() / 'esmvaltool_download')
        local_files = download(files, dest_folder)
        print(f"Download successful, {local_files=}")


if __name__ == '__main__':
    test_esgf_search()

    # test_esgf_download()

    test_download_all()
