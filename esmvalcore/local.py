"""Find files on the local filesystem.

.. deprecated:: 2.14.0
    This module has been moved to :mod:`esmvalcore.io.local`. Importing it as
    :mod:`esmvalcore.local` is deprecated and will be removed in version 2.16.0.
"""

from __future__ import annotations

import logging
import os.path
import textwrap
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from esmvalcore.config import CFG
from esmvalcore.config._config import CFG as CONFIG_DEVELOPER
from esmvalcore.config._config import (
    get_ignored_warnings,
    get_project_config,
    load_config_developer,
)
from esmvalcore.io.local import (
    LocalDataSource,
    LocalFile,
    _filter_versions_called_latest,
    _replace_tags,
    _select_latest_version,
)

if TYPE_CHECKING:
    from esmvalcore.typing import FacetValue

__all__ = [
    "DataSource",
    "LocalDataSource",
    "LocalFile",
    "find_files",
]

logger = logging.getLogger(__name__)


def _ensure_config_developer_drs() -> Path:
    """Ensure that directory structure from config-developer.yml is loaded."""
    config_developer_file = CFG.get("config_developer_file")
    if not config_developer_file:
        config_developer_file = Path(__file__).parent / "config-developer.yml"
    if not CONFIG_DEVELOPER:
        # Load the config-developer.yml file, but do not update the CMOR tables.
        load_config_developer(config_developer_file, set_cmor_tables=False)
    return config_developer_file


def _select_drs(input_type: str, project: str, structure: str) -> list[str]:
    """Select the directory structure of input path."""
    cfg = get_project_config(project)
    input_path_patterns = cfg[input_type]
    if isinstance(input_path_patterns, str):
        return [input_path_patterns]

    if structure in input_path_patterns:
        value = input_path_patterns[structure]
        if isinstance(value, str):
            value = [value]
        return value

    msg = f"drs {structure} for {project} project not specified in config-developer file"
    raise KeyError(msg)


_ROOTPATH_WARNED: set[tuple[str, tuple[str]]] = set()

_LEGACY_DATA_SOURCES_WARNED: set[str] = set()


def _get_data_sources(project: str) -> list[LocalDataSource]:
    """Get a list of data sources."""
    config_developer_file = _ensure_config_developer_drs()
    rootpaths = CFG["rootpath"]
    default_drs = {
        "CMIP3": "ESGF",
        "CMIP5": "ESGF",
        "CMIP6": "ESGF",
        "CORDEX": "ESGF",
        "obs4MIPs": "ESGF",
    }
    for key in (project, "default"):
        if key in rootpaths:
            paths = rootpaths[key]
            nonexistent = tuple(p for p in paths if not os.path.exists(p))
            if nonexistent and (key, nonexistent) not in _ROOTPATH_WARNED:
                logger.warning(
                    "Configured '%s' rootpaths '%s' do not exist",
                    key,
                    ", ".join(str(p) for p in nonexistent),
                )
                _ROOTPATH_WARNED.add((key, nonexistent))
            if isinstance(paths, list):
                structure = CFG.get("drs", {}).get(
                    project,
                    default_drs.get(project, "default"),
                )
                paths = dict.fromkeys(paths, structure)
            sources: list[LocalDataSource] = []
            for path, structure in paths.items():
                dir_templates = _select_drs("input_dir", project, structure)
                file_templates = _select_drs("input_file", project, structure)
                sources.extend(
                    LocalDataSource(
                        name="legacy-local",
                        project=project,
                        priority=1,
                        rootpath=Path(path),
                        dirname_template=d,
                        filename_template=f,
                        ignore_warnings=get_ignored_warnings(project, "load"),
                    )
                    for d in dir_templates
                    for f in file_templates
                )
            if project not in _LEGACY_DATA_SOURCES_WARNED:
                logger.warning(
                    (
                        "Using legacy data sources for project '%s' using 'rootpath' "
                        "and 'drs' settings and the path templates from '%s'"
                    ),
                    project,
                    config_developer_file,
                )
                _LEGACY_DATA_SOURCES_WARNED.add(project)
            return sources

    msg = (
        f"No '{project}' or 'default' path specified under 'rootpath' in "
        "the configuration."
    )
    raise KeyError(msg)


class DataSource(LocalDataSource):
    """Data source for finding files on a local filesystem.

    .. deprecated:: 2.14.0
         This class is deprecated and will be removed in version 2.16.0.
         Please use :class:`esmvalcore.local.LocalDataSource` instead.
    """

    def __init__(self, *args, **kwargs):
        msg = (
            "The 'esmvalcore.local.LocalDataSource' class is deprecated and will be "
            "removed in version 2.16.0. Please use 'esmvalcore.local.LocalDataSource'"
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

    @property
    def regex_pattern(self) -> str:
        """Get regex pattern that can be used to extract facets from paths."""
        return self._regex_pattern

    def get_glob_patterns(self, **facets: FacetValue) -> list[Path]:
        """Compose the globs that will be used to look for files."""
        return self._get_glob_patterns(**facets)

    def path2facets(self, path: Path, add_timerange: bool) -> dict[str, str]:
        """Extract facets from path."""
        return self._path2facets(path, add_timerange)

    def find_files(self, **facets: FacetValue) -> list[LocalFile]:
        """Find files."""
        return self.find_data(**facets)


def find_files(
    *,
    debug: bool = False,
    **facets: FacetValue,
) -> list[LocalFile] | tuple[list[LocalFile], list[Path]]:
    """Find files on the local filesystem.

    .. deprecated:: 2.14.0
         This function is deprecated and will be removed in version 2.16.0.
         Please use :meth:`esmvalcore.local.LocalDataSource.find_data` instead.

    The directories that are searched for files are defined in
    :data:`esmvalcore.config.CFG` under the ``'rootpath'`` key using the
    directory structure defined under the ``'drs'`` key.
    If ``esmvalcore.config.CFG['rootpath']`` contains a key that matches the
    value of the ``project`` facet, those paths will be used. If there is no
    project specific key, the directories in
    ``esmvalcore.config.CFG['rootpath']['default']`` will be searched.

    See :ref:`findingdata` for extensive instructions on configuring ESMValCore
    so it can find files locally.

    Parameters
    ----------
    debug
        When debug is set to :obj:`True`, the function will return a tuple
        with the first element containing the files that were found
        and the second element containing the :func:`glob.glob` patterns that
        were used to search for files.
    **facets
        Facets used to search for files. An ``'*'`` can be used to match
        any value. By default, only the latest version of a file will
        be returned. To select all versions use ``version='*'``. It is also
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
    Search for files containing surface air temperature from any CMIP6 model
    for the historical experiment:

    >>> esmvalcore.local.find_files(
    ...     project='CMIP6',
    ...     activity='CMIP',
    ...     mip='Amon',
    ...     short_name='tas',
    ...     exp='historical',
    ...     dataset='*',
    ...     ensemble='*',
    ...     grid='*',
    ...     institute='*',
    ... )  # doctest: +SKIP
    [LocalFile('/home/bandela/climate_data/CMIP6/CMIP/BCC/BCC-ESM1/historical/r1i1p1f1/Amon/tas/gn/v20181214/tas_Amon_BCC-ESM1_historical_r1i1p1f1_gn_185001-201412.nc')]

    Returns
    -------
    list[LocalFile]
        The files that were found.
    """
    msg = (
        "The function 'esmvalcore.local.find_files' is deprecated and will be removed "
        "in version 2.16.0. Please use 'esmvalcore.local.LocalDataSource.find_data'"
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)

    facets = dict(facets)
    if "original_short_name" in facets:
        facets["short_name"] = facets["original_short_name"]

    files = []
    filter_latest = False
    data_sources = _get_data_sources(facets["project"])  # type: ignore
    for data_source in data_sources:
        for file in data_source.find_data(**facets):
            if file.facets.get("version") == "latest":
                filter_latest = True
            files.append(file)

    if filter_latest:
        files = _filter_versions_called_latest(files)

    if "version" not in facets:
        files = _select_latest_version(files)

    files.sort()  # sorting makes it easier to see what was found

    if debug:
        globs = []
        for data_source in data_sources:
            globs.extend(data_source._get_glob_patterns(**facets))  # noqa: SLF001
        return files, sorted(globs)
    return files


_GET_OUTPUT_FILE_WARNED: set[str] = set()


def _get_output_file(variable: dict[str, Any], preproc_dir: Path) -> Path:
    """Return the full path to the output (preprocessed) file."""
    _ensure_config_developer_drs()
    project = variable["project"]
    cfg = get_project_config(project)
    if project not in _GET_OUTPUT_FILE_WARNED:
        _GET_OUTPUT_FILE_WARNED.add(project)
        msg = textwrap.dedent(
            f"""
            Defining 'output_file' in config-developer.yml is deprecated and will be removed in version 2.16.0. Please use the following configuration instead:
            projects:
              {variable["project"]}:
                preprocessor_filename_template: "{cfg["output_file"]}"
            """,
        ).rstrip()
        logger.warning(msg)

    # Join different experiment names
    if isinstance(variable.get("exp"), (list, tuple)):
        variable = dict(variable)
        variable["exp"] = "-".join(variable["exp"])
    outfile = _replace_tags(cfg["output_file"], variable)[0]
    if "timerange" in variable:
        timerange = variable["timerange"].replace("/", "-")
        outfile = Path(f"{outfile}_{timerange}")
    outfile = Path(f"{outfile}.nc")
    return Path(
        preproc_dir,
        variable.get("diagnostic", ""),
        variable.get("variable_group", ""),
        outfile,
    )
