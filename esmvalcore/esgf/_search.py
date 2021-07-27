"""Module for finding files on ESGF."""
# TODO: improve password saving by using keyring?
import itertools
import logging
import pprint

import pyesgf.logon
import pyesgf.search

from .._config._esgf_pyclient import _load_esgf_pyclient_config
from .._data_finder import get_start_end_year, select_files
from ._download import ESGFFile

logger = logging.getLogger(__name__)

FACETS = {
    'CMIP3': {
        'dataset': 'model',
        'ensemble': 'ensemble',
        'exp': 'experiment',
        'frequency': 'time_frequency',
        'short_name': 'variable',
    },
    'CMIP5': {
        'dataset': 'model',
        'ensemble': 'ensemble',
        'exp': 'experiment',
        'mip': 'cmor_table',
        'short_name': 'variable',
    },
    'CMIP6': {
        'activity': 'activity_id',
        'dataset': 'source_id',
        'ensemble': 'variant_label',
        'exp': 'experiment_id',
        'grid': 'grid_label',
        'mip': 'table_id',
        'short_name': 'variable',
    },
    'CORDEX': {
        'dataset': 'rcm_name',
        'driver': 'driving_model',
        'domain': 'domain',
        'ensemble': 'ensemble',
        'exp': 'experiment',
        'frequency': 'time_frequency',
        'short_name': 'variable',
    },
    'obs4MIPs': {
        'dataset': 'source_id',
        'frequency': 'time_frequency',
        'short_name': 'variable',
    }
}
"""Mapping between the recipe and ESGF facet names."""

DATASET_MAP = {
    'CMIP3': {},
    'CMIP5': {
        'ACCESS1-0': 'ACCESS1.0',
        'ACCESS1-3': 'ACCESS1.3',
        'bcc-csm1-1': 'BCC-CSM1.1',
        'bcc-csm1-1-m': 'BCC-CSM1.1(m)',
        'CESM1-BGC': 'CESM1(BGC)',
        'CESM1-CAM5': 'CESM1(CAM5)',
        'CESM1-CAM5-1-FV2': 'CESM1(CAM5.1,FV2)',
        'CESM1-FASTCHEM': 'CESM1(FASTCHEM)',
        'CESM1-WACCM': 'CESM1(WACCM)',
        'CSIRO-Mk3-6-0': 'CSIRO-Mk3.6.0',
        'fio-esm': 'FIO-ESM',
        'GFDL-CM2p1': 'GFDL-CM2.1',
        'inmcm4': 'INM-CM4',
        'MRI-AGCM3-2H': 'MRI-AGCM3.2H',
        'MRI-AGCM3-2S': 'MRI-AGCM3.2S'
    },
    'CMIP6': {},
    'CORDEX': {},
    'obs4MIPs': {},
}
"""Cache for the mapping between recipe/filesystem and ESGF dataset names."""


def create_dataset_map(connection=None):
    """Create the DATASET_MAP from recipe datasets to ESGF dataset names."""
    dataset_map = {}

    if connection is None:
        connection = get_connection()

    indices = {
        'CMIP3': 2,
        'CMIP5': 3,
        'CMIP6': 3,
        'CORDEX': 7,
        'obs4MIPs': 2,
    }

    for project in FACETS:
        dataset_map[project] = {}
        ctx = connection.new_context(project=project, latest=True)

        dataset_key = FACETS[project]['dataset']
        available_datasets = sorted(ctx.facet_counts[dataset_key])
        print(f"The following datasets are available for project {project}:")
        for dataset in available_datasets:
            print(dataset)

        # Figure out the ESGF name of the requested dataset
        n_available = len(available_datasets)
        for i, dataset in enumerate(available_datasets, 1):
            print(f"Looking for dataset name of facet name"
                  f" {dataset} ({i} of {n_available})")
            query = {dataset_key: dataset}
            dataset_result = next(iter(ctx.search(batch_size=1, **query)))
            print(f"Dataset id: {dataset_result.dataset_id}")
            dataset_id = dataset_result.dataset_id
            if dataset in dataset_id:
                print(f"Dataset facet is identical to "
                      f"dataset name for '{dataset}'")
            else:
                idx = indices[project]
                dataset_alias = dataset_id.split('.')[idx]
                print(f"Found dataset name '{dataset_alias}'"
                      f" for facet '{dataset}',")
                dataset_map[project][dataset_alias] = dataset

    return dataset_map


def get_esgf_facets(variable):
    """Translate variable to facets for searching on ESGF."""
    project = variable.get('project', '')
    if project not in FACETS:
        logger.info(
            "Unable to download from ESGF, because project '%s' is not on it.",
            project)
        return None

    facets = {'project': project}
    if project == 'CMIP5':
        facets['product'] = 'output1'
    for our_name, esgf_name in FACETS[project].items():
        if our_name in variable:
            value = variable[our_name]
            if our_name == 'dataset':
                # Replace dataset name by ESGF name for dataset
                value = DATASET_MAP[project].get(value, value)
            facets[esgf_name] = value

    return facets


def sort_hosts(hosts, preferred_hosts):
    """Select a list of suitable hosts from a list of hosts.

    Parameters
    ----------
    hosts : :obj:`list` of :obj:`str`
        List of all available hosts.
    preferred_hosts : :obj:`list` of :obj:`str`
        List of preferred hosts.

    Returns
    -------
    :obj:`list` of :obj:`str`
        The list of hosts, with preferred hosts first.
    """
    hosts = list(hosts)
    for host in preferred_hosts[::-1]:
        if host in hosts:
            # Move host to the beginning of the list
            hosts.insert(0, hosts.pop(hosts.index(host)))

    return hosts


def select_latest_versions(datasets: dict) -> dict:
    """Return a dict with only the latest version of each dataset.

    Parameters
    ----------
    datasets : dict
        A dict with dataset objects

    Returns
    -------
    most_recent_datasets : dict
        A dict containing only the most recent version of each dataset object,
        in case multiple versions have been passed.
    """
    def name(dataset_name):
        """Return the name of the dataset without the version."""
        return dataset_name.rsplit('.', 1)[0]

    latest_versions = {}
    for _, versions in itertools.groupby(sorted(datasets), key=name):
        latest = list(versions)[-1]
        latest_versions[latest] = datasets[latest]

    return latest_versions


def find_files(datasets, short_name):
    """Find files for each dataset in datasets."""
    cfg = _load_esgf_pyclient_config()
    preferred_hosts = cfg.get('preferred_hosts', [])

    files = {}
    for dataset_name in sorted(datasets):
        copies = datasets[dataset_name]
        hosts = sort_hosts(copies.keys(), preferred_hosts)
        logger.info(
            "Searching for files for dataset %s, available on hosts %s",
            dataset_name,
            hosts,
        )

        dataset_files = {}
        for host in hosts:
            dataset_result = copies[host]
            file_result = dataset_result.file_context().search(
                variable=short_name)
            for file in file_result:
                if file.filename in dataset_files:
                    dataset_files[file.filename].urls.append(file.download_url)
                else:
                    dataset_files[file.filename] = ESGFFile(
                        urls=[file.download_url],
                        dataset=dataset_name,
                        name=file.filename,
                        size=file.size,
                        checksum=file.checksum,
                        checksum_type=file.checksum_type,
                    )
        files[dataset_name] = list(dataset_files.values())

    return files


def merge_datasets(datasets):
    """Merge datasets that only differ in capitalization.

    Example of two datasets that will be merged:

    Dataset available on hosts 'esgf-data1.ceda.ac.uk', 'esgf.nci.org.au',
    'esgf2.dkrz.de':
    cmip5.output1.FIO.FIO-ESM.historical.mon.atmos.Amon.r1i1p1.v20121010
    Dataset available on host 'aims3.llnl.gov':
    cmip5.output1.FIO.fio-esm.historical.mon.atmos.Amon.r1i1p1.v20121010
    """
    merged = {}
    names = sorted(datasets, key=str.lower)
    for _, duplicates in itertools.groupby(names, key=str.lower):
        name = next(duplicates)
        files = {file.name: file for file in datasets[name]}
        for alt_name in duplicates:
            logger.info("Combining dataset %s with %s", alt_name, name)
            for file in datasets[alt_name]:
                if file.name not in files:
                    files[file.name] = file
                for url in file.urls:
                    if url not in files[file.name].urls:
                        files[file.name].urls.append(url)
        merged[name] = list(files.values())

    return merged


def esgf_search(recipe_facets):
    """Search for files on ESGF.

    Parameters
    ----------
    recipe_facets: :obj:`dict` of :obj:`str`
        Facets from the recipe to constrain the search.

    Returns
    -------
    :obj:`dict` of :obj:`list` of :obj:`ESGFFile`
        The found datasets, stored in a dict with dataset names as keys and
        lists of ESGFFile instances as values.
    """
    facets = get_esgf_facets(recipe_facets)
    if facets is None:
        return {}

    logger.info("Searching on ESGF using facets=%s", facets)
    connection = get_connection()
    context = connection.new_context(**facets, latest=True)
    logger.info(
        "Found %s datasets (any version, including copies)"
        " with facets=%s", context.hit_count, facets)

    # Obtain dataset results
    datasets = {}
    for dataset_result in context.search(**facets, latest=True):
        dataset_name, host = dataset_result.dataset_id.split('|')
        if dataset_name not in datasets:
            datasets[dataset_name] = {}
        datasets[dataset_name][host] = dataset_result

    logger.info("Found the following datasets matching facets %s:\n%s", facets,
                '\n'.join(datasets))

    if not datasets:
        project = facets['project']
        dataset_key = FACETS[project]['dataset']
        available_datasets = sorted(context.facet_counts[dataset_key])
        print(f"Available datasets for project {project}:"
              "\n" + "\n".join(available_datasets))
        raise ValueError(f"Dataset {facets[dataset_key]} not found")

    # Select only the latest versions
    datasets = select_latest_versions(datasets)

    # Find files for each dataset
    files = find_files(datasets, facets['variable'])

    # Merge datasets that only differ in capitalization
    files = merge_datasets(files)

    return files


def select_by_time(files, start_year, end_year):
    """Select files containing data between start_year and end_year."""
    filedict = {file.name: file for file in files}
    files = select_files(filedict, start_year, end_year)

    # filter partially overlapping files
    intervals = {get_start_end_year(name): name for name in files}
    files = []
    for (start, end), filename in intervals.items():
        for _start, _end in intervals:
            if start == _start and end == _end:
                continue
            if start >= _start and end <= _end:
                break
        else:
            files.append(filename)

    return [filedict[f] for f in files]


def get_connection():
    """Connect to ESGF."""
    cfg = _load_esgf_pyclient_config()

    manager = pyesgf.logon.LogonManager()
    if not manager.is_logged_on():
        manager.logon(**cfg['logon'])
        logger.info("Logged %s", "on" if manager.is_logged_on() else "off")

    connection = pyesgf.search.SearchConnection(**cfg["search_connection"])
    return connection


def search(*, project, short_name, dataset, **facets):
    """Search for files on ESGF.

    Parameters
    ----------
    project : str
        Choose from CMIP3, CMIP5, CMIP6, CORDEX, or obs4MIPs.
    short_name : str
        The name of the variable.
    dataset : str
        The name of the dataset.
    **facets:
        Any other search facets.

    Raises
    ------
    ValueError
        If more than one dataset is found.

    Examples
    --------
    Examples of how to use the search function for all supported projects.

    Search for a CMIP3 dataset:

    >>> search(
        project='CMIP3',
        frequency='mon',
        short_name='tas',
        dataset='cccma_cgcm3_1',
        exp='historical',
        ensemble='run1',
    )  # doctest: +SKIP
    [ESGFFile:cmip3/CCCma/cccma_cgcm3_1/historical/mon/atmos/run1/tas/v1/tas_a1_20c3m_1_cgcm3.1_t47_1850_2000.nc]

    Search for a CMIP5 dataset:

    >>> search(
        project='CMIP5',
        mip='Amon',
        short_name='tas',
        dataset='inmcm4',
        exp='historical',
        ensemble='r1i1p1',
    )  # doctest: +SKIP
    [ESGFFile:cmip5/output1/INM/inmcm4/historical/mon/atmos/Amon/r1i1p1/v20130207/tas_Amon_inmcm4_historical_r1i1p1_185001-200512.nc]

    Search for a CMIP6 dataset:

    >>> search(
        project='CMIP6',
        mip='Amon',
        short_name='tas',
        dataset='CanESM5',
        exp='historical',
        ensemble='r1i1p1f1',
    )  # doctest: +SKIP
    [ESGFFile:CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p1f1/Amon/tas/gn/v20190429/tas_Amon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc]

    Search for a CORDEX dataset and limit the search results to files
    containing data to the years in the range 1990-2000:

    >>> search(
        project='CORDEX',
        frequency='mon',
        dataset='COSMO-crCLIM-v1-1',
        short_name='tas',
        exp='historical',
        ensemble='r1i1p1',
        domain='EUR-11',
        driver='MPI-M-MPI-ESM-LR',
        start_year=1990,
        end_year=2000,
    )  # doctest: +SKIP
    [ESGFFile:cordex/output/EUR-11/CLMcom-ETH/MPI-M-MPI-ESM-LR/historical/r1i1p1/COSMO-crCLIM-v1-1/v1/mon/tas/v20191219/tas_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_mon_198101-199012.nc,
    ESGFFile:cordex/output/EUR-11/CLMcom-ETH/MPI-M-MPI-ESM-LR/historical/r1i1p1/COSMO-crCLIM-v1-1/v1/mon/tas/v20191219/tas_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_mon_199101-200012.nc]

    Search for a obs4MIPs dataset:

    >>> search(
        project='obs4MIPs',
        frequency='mon',
        dataset='CERES-EBAF',
        short_name='rsutcs',
    )  # doctest: +SKIP
    [ESGFFile:obs4MIPs/NASA-LaRC/CERES-EBAF/atmos/mon/v20160610/rsutcs_CERES-EBAF_L3B_Ed2-8_200003-201404.nc]

    Returns
    -------
    :obj:`list` of :obj:`ESGFFile`
        A list of files that have been found.
    """
    # The project and short_name functions are required for the function
    # to work.
    facets['project'] = project
    facets['short_name'] = short_name
    # The dataset facet is not strictly required, but without it it seems
    # likely that too many results will be found.
    facets['dataset'] = dataset

    datasets = esgf_search(facets)
    logger.info("Found files\n%s", pprint.pformat(datasets))
    if not datasets:
        return []

    if len(datasets) > 1:
        raise ValueError(
            "Expected to find a single dataset, found\n:{}".format(
                pprint.pformat(datasets)))

    dataset_name = next(iter(datasets))
    files = datasets[dataset_name]

    filter_timerange = (facets.get('frequency', '') != 'fx'
                        and 'start_year' in facets and 'end_year' in facets)
    if filter_timerange:
        files = select_by_time(files, facets['start_year'], facets['end_year'])
        logger.info("Selected files:\n%s", '\n'.join(str(f) for f in files))

    return files


if __name__ == '__main__':
    # Run this module to create an up to date DATASET_MAP
    print(create_dataset_map())
