"""Module for finding files on ESGF."""
import itertools
import logging
from functools import lru_cache

import pyesgf.search

from .._data_finder import get_start_end_year
from ._download import ESGFFile
from ._logon import get_connection
from .facets import DATASET_MAP, FACETS

logger = logging.getLogger(__name__)


def get_esgf_facets(variable):
    """Translate variable to facets for searching on ESGF."""
    project = variable.get('project', '')
    facets = {'project': project}
    for our_name, esgf_name in FACETS[project].items():
        if our_name in variable:
            values = variable[our_name]

            if isinstance(values, (tuple, list)):
                values = list(values)
            else:
                values = [values]

            for i, value in enumerate(values):
                if our_name == 'dataset':
                    # Replace dataset name by ESGF name for dataset
                    values[i] = DATASET_MAP[project].get(value, value)

            facets[esgf_name] = ','.join(values)

    return facets


def select_latest_versions(files):
    """Select only the latest version of files."""
    result = []

    def same_file(file):
        """Return a versionless identifier for a file."""
        # Dataset without the version number
        dataset = file.dataset.rsplit('.', 1)[0]
        return (dataset, file.name)

    files = sorted(files, key=same_file)
    for _, versions in itertools.groupby(files, key=same_file):
        versions = sorted(versions, reverse=True)
        latest_version = versions[0]
        result.append(latest_version)
        if len(versions) > 1:
            logger.debug("Only using the latest version %s, not %s",
                         latest_version, versions[1:])

    return result


def esgf_search_files(facets):
    """Search for files on ESGF.

    Parameters
    ----------
    facets: :obj:`dict` of :obj:`str`
        Facets to constrain the search.

    Returns
    -------
    list of :py:class:`~ESGFFile`
        The found files.
    """
    logger.debug("Searching for datasets on ESGF using facets=%s", facets)
    connection = get_connection()
    context = connection.new_context(
        pyesgf.search.context.FileSearchContext,
        **facets,
        latest=True,
    )

    results = context.search(
        batch_size=500,
        # enable ignore_facet_check once the following issue has been fixed:
        # https://github.com/ESGF/esgf-pyclient/issues/75
        # ignore_facet_check=True,
    )

    files = ESGFFile._from_results(results, facets)

    files = select_latest_versions(files)

    msg = 'none' if not files else '\n' + '\n'.join(str(f) for f in files)
    logger.debug("Found the following files matching facets %s: %s", facets,
                 msg)

    return files


def select_by_time(files, start_year, end_year):
    """Select files containing data between start_year and end_year."""
    selection = []
    for file in files:
        try:
            start, end = get_start_end_year(file.name)
        except ValueError:
            # If start and end year cannot be read from the filename
            # just select everything.
            selection.append(file)
        else:
            if start <= end_year and end >= start_year:
                selection.append(file)

    return selection


def find_files(*, project, short_name, dataset, **facets):
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
        Any other search facets. Values can be strings, list of strings, or
        'start_year' and 'end_year' with values of type :obj:`int`.

    Examples
    --------
    Examples of how to use the search function for all supported projects.

    Search for a CMIP3 dataset:

    >>> search(
    ...     project='CMIP3',
    ...     frequency='mon',
    ...     short_name='tas',
    ...     dataset='cccma_cgcm3_1',
    ...     exp='historical',
    ...     ensemble='run1',
    ... )  # doctest: +SKIP
    [ESGFFile:cmip3/CCCma/cccma_cgcm3_1/historical/mon/atmos/run1/tas/v1/tas_a1_20c3m_1_cgcm3.1_t47_1850_2000.nc]

    Search for a CMIP5 dataset:

    >>> search(
    ...     project='CMIP5',
    ...     mip='Amon',
    ...     short_name='tas',
    ...     dataset='inmcm4',
    ...     exp='historical',
    ...     ensemble='r1i1p1',
    ... )  # doctest: +SKIP
    [ESGFFile:cmip5/output1/INM/inmcm4/historical/mon/atmos/Amon/r1i1p1/v20130207/tas_Amon_inmcm4_historical_r1i1p1_185001-200512.nc]

    Search for a CMIP6 dataset:

    >>> search(
    ...     project='CMIP6',
    ...     mip='Amon',
    ...     short_name='tas',
    ...     dataset='CanESM5',
    ...     exp='historical',
    ...     ensemble='r1i1p1f1',
    ... )  # doctest: +SKIP
    [ESGFFile:CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p1f1/Amon/tas/gn/v20190429/tas_Amon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc]

    Search for a CORDEX dataset and limit the search results to files
    containing data to the years in the range 1990-2000:

    >>> search(
    ...     project='CORDEX',
    ...     frequency='mon',
    ...     dataset='COSMO-crCLIM-v1-1',
    ...     short_name='tas',
    ...     exp='historical',
    ...     ensemble='r1i1p1',
    ...     domain='EUR-11',
    ...     driver='MPI-M-MPI-ESM-LR',
    ...     start_year=1990,
    ...     end_year=2000,
    ... )  # doctest: +SKIP
    [ESGFFile:cordex/output/EUR-11/CLMcom-ETH/MPI-M-MPI-ESM-LR/historical/r1i1p1/COSMO-crCLIM-v1-1/v1/mon/tas/v20191219/tas_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_mon_198101-199012.nc,
    ESGFFile:cordex/output/EUR-11/CLMcom-ETH/MPI-M-MPI-ESM-LR/historical/r1i1p1/COSMO-crCLIM-v1-1/v1/mon/tas/v20191219/tas_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_mon_199101-200012.nc]

    Search for a obs4MIPs dataset:

    >>> search(
    ...     project='obs4MIPs',
    ...     frequency='mon',
    ...     dataset='CERES-EBAF',
    ...     short_name='rsutcs',
    ... )  # doctest: +SKIP
    [ESGFFile:obs4MIPs/NASA-LaRC/CERES-EBAF/atmos/mon/v20160610/rsutcs_CERES-EBAF_L3B_Ed2-8_200003-201404.nc]

    Returns
    -------
    :obj:`list` of :obj:`ESGFFile`
        A list of files that have been found.
    """  # pylint: disable=locally-disabled, line-too-long
    if project not in FACETS:
        raise ValueError(
            f"Unable to download from ESGF, because project {project} is not"
            " on it or is not supported by the esmvalcore.esgf module.")

    # The project is required for the function to work.
    facets['project'] = project
    # The dataset and short_name facet are not strictly required,
    # but without these it seems likely that the user is requesting
    # more results than they intended.
    facets['dataset'] = dataset
    facets['short_name'] = short_name

    # Convert lists to tuples to allow caching results
    for facet, value in facets.items():
        if isinstance(value, list):
            facets[facet] = tuple(value)

    return cached_search(**facets)


@lru_cache(10000)
def cached_search(**facets):
    """Search for files on ESGF.

    A cached search function will speed up recipes that use the same
    variable multiple times.
    """
    esgf_facets = get_esgf_facets(facets)
    files = esgf_search_files(esgf_facets)

    filter_timerange = (facets.get('frequency', '') != 'fx'
                        and 'start_year' in facets and 'end_year' in facets)
    if filter_timerange:
        files = select_by_time(files, facets['start_year'], facets['end_year'])
        logger.debug("Selected files:\n%s", '\n'.join(str(f) for f in files))

    return files
