"""Module for finding files on ESGF."""

import itertools
import logging
from functools import lru_cache

import pyesgf.search
import requests.exceptions

from esmvalcore.config._esgf_pyclient import get_esgf_config
from esmvalcore.local import (
    _get_start_end_date,
    _parse_period,
    _replace_years_with_timerange,
    _truncate_dates,
)

from ._download import ESGFFile
from .facets import DATASET_MAP, FACETS

logger = logging.getLogger(__name__)


def get_esgf_facets(variable):
    """Translate variable to facets for searching on ESGF."""
    project = variable.get("project", "")
    facets = {"project": project}
    for our_name, esgf_name in FACETS[project].items():
        if our_name in variable:
            values = variable[our_name]
            if values == "*":
                # Wildcards can be specified on ESGF by omitting the facet
                continue

            if isinstance(values, (tuple, list)):
                values = list(values)
            else:
                values = [values]

            for i, value in enumerate(values):
                if our_name == "dataset":
                    # Replace dataset name by ESGF name for dataset
                    values[i] = DATASET_MAP[project].get(value, value)

            facets[esgf_name] = ",".join(values)

    return facets


def select_latest_versions(files, versions):
    """Select only the latest version of files."""
    result = []

    def same_file(file):
        """Return a versionless identifier for a file."""
        # Dataset without the version number
        dataset = file.dataset.rsplit(".", 1)[0]
        return (dataset, file.name)

    if isinstance(versions, str):
        versions = (versions,)

    files = sorted(files, key=same_file)
    for _, group in itertools.groupby(files, key=same_file):
        group = sorted(group, reverse=True)
        if versions:
            selection = [f for f in group if f.facets["version"] in versions]
            if not selection:
                # Skip the file if it is not the requested version(s).
                continue
            group = selection
        latest_version = group[0]
        result.append(latest_version)
        if len(group) > 1:
            logger.debug(
                "Only using the latest version %s, not %s",
                latest_version,
                group[1:],
            )

    return result


FIRST_ONLINE_INDEX_NODE = None
"""Remember the first index node that is online."""


def _search_index_nodes(facets):
    """Search for files on ESGF.

    Parameters
    ----------
    facets: :obj:`dict` of :obj:`str`
        Facets to constrain the search.

    Raises
    ------
    FileNotFoundError
        If the function was unable to connect to ESGF.

    Returns
    -------
    pyesgf.search.results.ResultSet
        A ResultSet containing :obj:`pyesgf.search.results.FileResult`s.
    """
    cfg = get_esgf_config()
    search_args = dict(cfg["search_connection"])
    urls = search_args.pop("urls")

    global FIRST_ONLINE_INDEX_NODE  # noqa: PLW0603
    if FIRST_ONLINE_INDEX_NODE:
        urls.insert(0, urls.pop(urls.index(FIRST_ONLINE_INDEX_NODE)))

    errors = []
    for url in urls:
        connection = pyesgf.search.SearchConnection(url=url, **search_args)
        context = connection.new_context(
            pyesgf.search.context.FileSearchContext,
            **facets,
        )
        logger.debug("Searching %s for datasets using facets=%s", url, facets)
        try:
            results = context.search(
                batch_size=500,
                ignore_facet_check=True,
            )
            FIRST_ONLINE_INDEX_NODE = url
            return list(results)
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
            requests.exceptions.Timeout,
        ) as error:
            logger.debug("Unable to connect to %s due to %s", url, error)
            errors.append(error)

    raise FileNotFoundError(
        "Failed to search ESGF, unable to connect:\n"
        + "\n".join(f"- {e}" for e in errors),
    )


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
    results = _search_index_nodes(facets)

    files = ESGFFile._from_results(results, facets)  # noqa: SLF001

    msg = "none" if not files else "\n" + "\n".join(str(f) for f in files)
    logger.debug(
        "Found the following files matching facets %s: %s",
        facets,
        msg,
    )

    return files


def select_by_time(files, timerange):
    """Select files containing data between a timerange."""
    if "*" in timerange:
        # TODO: support * combined with a period
        return files

    selection = []

    for file in files:
        start_date, end_date = _parse_period(timerange)
        try:
            start, end = _get_start_end_date(file)
        except ValueError:
            # If start and end year cannot be read from the filename
            # just select everything.
            selection.append(file)
        else:
            start_date, end = _truncate_dates(start_date, end)
            end_date, start = _truncate_dates(end_date, start)
            if start <= end_date and end >= start_date:
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
    **facets : typing.Union[str, list[str]]
        Any other search facets. An ``'*'`` can be used to match
        any value. By default, only the latest version of a file will
        be returned. To select all versions use ``version='*'`` while other
        omitted facets will default to ``'*'``. It is also
        possible to specify multiple values for a facet, e.g.
        ``exp=['historical', 'ssp585']`` will match any file that belongs
        to either the historical or ssp585 experiment.
        The ``timerange`` facet can be specified in `ISO 8601 format
        <https://en.wikipedia.org/wiki/ISO_8601>`__.

    Note
    ----
    A value of ``timerange='*'`` is supported, but combining a ``'*'`` with
    a time or period :ref:`as supported in the recipe <datasets>` is currently
    not supported and will return all found files.

    Examples
    --------
    Examples of how to use this function for all supported projects.

    Search for a CMIP3 dataset:

    >>> find_files(
    ...     project='CMIP3',
    ...     frequency='mon',
    ...     short_name='tas',
    ...     dataset='cccma_cgcm3_1',
    ...     exp='historical',
    ...     ensemble='run1',
    ... )  # doctest: +SKIP
    [ESGFFile:cmip3/CCCma/cccma_cgcm3_1/historical/mon/atmos/run1/tas/v1/tas_a1_20c3m_1_cgcm3.1_t47_1850_2000.nc]

    Search for a CMIP5 dataset:

    >>> find_files(
    ...     project='CMIP5',
    ...     mip='Amon',
    ...     short_name='tas',
    ...     dataset='inmcm4',
    ...     exp='historical',
    ...     ensemble='r1i1p1',
    ... )  # doctest: +SKIP
    [ESGFFile:cmip5/output1/INM/inmcm4/historical/mon/atmos/Amon/r1i1p1/v20130207/tas_Amon_inmcm4_historical_r1i1p1_185001-200512.nc]

    Search for a CMIP6 dataset:

    >>> find_files(
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

    >>> find_files(
    ...     project='CORDEX',
    ...     frequency='mon',
    ...     dataset='COSMO-crCLIM-v1-1',
    ...     short_name='tas',
    ...     exp='historical',
    ...     ensemble='r1i1p1',
    ...     domain='EUR-11',
    ...     driver='MPI-M-MPI-ESM-LR',
    ...     timerange='1990/2000',
    ... )  # doctest: +SKIP
    [ESGFFile:cordex/output/EUR-11/CLMcom-ETH/MPI-M-MPI-ESM-LR/historical/r1i1p1/COSMO-crCLIM-v1-1/v1/mon/tas/v20191219/tas_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_mon_198101-199012.nc,
     ESGFFile:cordex/output/EUR-11/CLMcom-ETH/MPI-M-MPI-ESM-LR/historical/r1i1p1/COSMO-crCLIM-v1-1/v1/mon/tas/v20191219/tas_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_mon_199101-200012.nc]

    Search for an obs4MIPs dataset:

    >>> find_files(
    ...     project='obs4MIPs',
    ...     frequency='mon',
    ...     dataset='CERES-EBAF',
    ...     short_name='rsutcs',
    ... )  # doctest: +SKIP
    [ESGFFile:obs4MIPs/NASA-LaRC/CERES-EBAF/atmos/mon/v20160610/rsutcs_CERES-EBAF_L3B_Ed2-8_200003-201404.nc]

    Search for any ensemble member:

    >>> find_files(
    ...     project='CMIP6',
    ...     mip='Amon',
    ...     short_name='tas',
    ...     dataset='BCC-CSM2-MR',
    ...     exp='historical',
    ...     ensemble='*',
    ... )  # doctest: +SKIP
    [ESGFFile:CMIP6/CMIP/BCC/BCC-CSM2-MR/historical/r1i1p1f1/Amon/tas/gn/v20181126/tas_Amon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc,
     ESGFFile:CMIP6/CMIP/BCC/BCC-CSM2-MR/historical/r2i1p1f1/Amon/tas/gn/v20181115/tas_Amon_BCC-CSM2-MR_historical_r2i1p1f1_gn_185001-201412.nc,
     ESGFFile:CMIP6/CMIP/BCC/BCC-CSM2-MR/historical/r3i1p1f1/Amon/tas/gn/v20181119/tas_Amon_BCC-CSM2-MR_historical_r3i1p1f1_gn_185001-201412.nc]

    Search for all available versions of a file:

    >>> find_files(
    ...     project='CMIP5',
    ...     mip='Amon',
    ...     short_name='tas',
    ...     dataset='CCSM4',
    ...     exp='historical',
    ...     ensemble='r1i1p1',
    ...     version='*',
    ... )  # doctest: +SKIP
    [ESGFFile:cmip5/output1/NCAR/CCSM4/historical/mon/atmos/Amon/r1i1p1/v20121031/tas_Amon_CCSM4_historical_r1i1p1_185001-200512.nc,
     ESGFFile:cmip5/output1/NCAR/CCSM4/historical/mon/atmos/Amon/r1i1p1/v20130425/tas_Amon_CCSM4_historical_r1i1p1_185001-200512.nc,
     ESGFFile:cmip5/output1/NCAR/CCSM4/historical/mon/atmos/Amon/r1i1p1/v20160829/tas_Amon_CCSM4_historical_r1i1p1_185001-200512.nc]

    Search for a specific version of a file:

    >>> find_files(
    ...     project='CMIP5',
    ...     mip='Amon',
    ...     short_name='tas',
    ...     dataset='CCSM4',
    ...     exp='historical',
    ...     ensemble='r1i1p1',
    ...     version='v20130425',
    ... )  # doctest: +SKIP
    [ESGFFile:cmip5/output1/NCAR/CCSM4/historical/mon/atmos/Amon/r1i1p1/v20130425/tas_Amon_CCSM4_historical_r1i1p1_185001-200512.nc]

    Returns
    -------
    :obj:`list` of :obj:`ESGFFile`
        A list of files that have been found.
    """  # pylint: disable=locally-disabled, line-too-long
    if project not in FACETS:
        msg = (
            f"Unable to download from ESGF, because project {project} is not"
            " on it or is not supported by the esmvalcore.esgf module."
        )
        raise ValueError(
            msg,
        )

    # The project is required for the function to work.
    facets["project"] = project
    # The dataset and short_name facet are not strictly required,
    # but without these it seems likely that the user is requesting
    # more results than they intended.
    facets["dataset"] = dataset
    facets["short_name"] = short_name

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

    if "version" not in facets or facets["version"] != "*":
        files = select_latest_versions(files, facets.get("version"))

    _replace_years_with_timerange(facets)
    if "timerange" in facets:
        files = select_by_time(files, facets["timerange"])
        logger.debug("Selected files:\n%s", "\n".join(str(f) for f in files))

    return files
