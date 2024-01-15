"""Classes and functions for defining, finding, and loading data."""
from __future__ import annotations

import logging
import pprint
import re
import textwrap
import uuid
from copy import deepcopy
from fnmatch import fnmatchcase
from itertools import groupby
from pathlib import Path
from typing import Any, Iterator, Sequence, Union

from iris.cube import Cube

from esmvalcore import esgf, local
from esmvalcore._recipe import check
from esmvalcore._recipe.from_datasets import datasets_to_recipe
from esmvalcore.cmor.table import _get_mips, _update_cmor_facets
from esmvalcore.config import CFG, Session
from esmvalcore.config._config import (
    get_activity,
    get_extra_facets,
    get_ignored_warnings,
    get_institutes,
)
from esmvalcore.exceptions import InputFilesNotFound, RecipeError
from esmvalcore.local import (
    _dates_to_timerange,
    _get_output_file,
    _get_start_end_date,
)
from esmvalcore.preprocessor import preprocess
from esmvalcore.typing import Facets, FacetValue

__all__ = [
    'Dataset',
    'INHERITED_FACETS',
    'datasets_to_recipe',
]

logger = logging.getLogger(__name__)

File = Union[esgf.ESGFFile, local.LocalFile]

INHERITED_FACETS: list[str] = [
    'dataset',
    'domain',
    'driver',
    'grid',
    'project',
    'timerange',
]
"""Inherited facets.

Supplementary datasets created based on the available files using the
:func:`Dataset.from_files` method will inherit the values of these facets from
the main dataset.
"""


def _augment(base: dict, update: dict):
    """Update dict `base` with values from dict `update`."""
    for key in update:
        if key not in base:
            base[key] = update[key]


def _isglob(facet_value: FacetValue | None) -> bool:
    """Check if a facet value is a glob pattern."""
    return (isinstance(facet_value, str)
            and bool(re.match(r'.*[\*\?]+.*|.*\[.*\].*', facet_value)))


def _ismatch(facet_value: FacetValue, pattern: FacetValue) -> bool:
    """Check if a facet value matches a glob pattern."""
    return (isinstance(pattern, str) and isinstance(facet_value, str)
            and fnmatchcase(facet_value, pattern))


class Dataset:
    """Define datasets, find the related files, and load them.

    Parameters
    ----------
    **facets
        Facets describing the dataset. See
        :obj:`esmvalcore.esgf.facets.FACETS` for the mapping between
        the facet names used by ESMValCore and those used on ESGF.

    Attributes
    ----------
    supplementaries : list[Dataset]
        List of supplementary datasets.
    facets: :obj:`esmvalcore.typing.Facets`
        Facets describing the dataset.
    """

    _SUMMARY_FACETS = (
        'short_name',
        'mip',
        'project',
        'dataset',
        'rcm_version',
        'driver',
        'domain',
        'activity',
        'exp',
        'ensemble',
        'grid',
        'version',
    )
    """Facets used to create a summary of a Dataset instance."""

    def __init__(self, **facets: FacetValue):

        self.facets: Facets = {}
        self.supplementaries: list['Dataset'] = []

        self._persist: set[str] = set()
        self._session: Session | None = None
        self._files: Sequence[File] | None = None
        self._file_globs: Sequence[Path] | None = None

        for key, value in facets.items():
            self.set_facet(key, deepcopy(value), persist=True)

    @staticmethod
    def from_recipe(
        recipe: Path | str | dict,
        session: Session,
    ) -> list['Dataset']:
        """Read datasets from a recipe.

        Parameters
        ----------
        recipe
            :ref:`Recipe <recipe>` to load the datasets from. The value
            provided here should be either a path to a file, a recipe file
            that has been loaded using e.g. :func:`yaml.safe_load`, or an
            :obj:`str` that can be loaded using :func:`yaml.safe_load`.
        session
            Datasets to use in the recipe.

        Returns
        -------
        list[Dataset]
            A list of datasets.
        """
        from esmvalcore._recipe.to_datasets import datasets_from_recipe
        return datasets_from_recipe(recipe, session)

    def _file_to_dataset(
        self,
        file: esgf.ESGFFile | local.LocalFile,
    ) -> Dataset:
        """Create a dataset from a file with a `facets` attribute."""
        facets = dict(file.facets)
        if 'version' not in self.facets:
            # Remove version facet if no specific version requested
            facets.pop('version', None)

        updated_facets = {
            f: v
            for f, v in facets.items() if f in self.facets
            and _isglob(self.facets[f]) and _ismatch(v, self.facets[f])
        }
        dataset = self.copy()
        dataset.facets.update(updated_facets)

        # If possible, remove unexpanded facets that can be automatically
        # populated.
        unexpanded = {f for f, v in dataset.facets.items() if _isglob(v)}
        required_for_augment = {'project', 'mip', 'short_name', 'dataset'}
        if unexpanded and not unexpanded & required_for_augment:
            copy = dataset.copy()
            copy.supplementaries = []
            for facet in unexpanded:
                copy.facets.pop(facet)
            copy.augment_facets()
            for facet in unexpanded:
                if facet in copy.facets:
                    dataset.facets.pop(facet)

        return dataset

    def _get_available_datasets(self) -> Iterator[Dataset]:
        """Yield datasets based on the available files.

        This function requires that self.facets['mip'] is not a glob pattern.
        """
        dataset_template = self.copy()
        dataset_template.supplementaries = []
        if _isglob(dataset_template.facets.get('timerange')):
            # Remove wildcard `timerange` facet, because data finding cannot
            # handle it
            dataset_template.facets.pop('timerange')

        seen = set()
        partially_defined = []
        expanded = False
        for file in dataset_template.files:
            dataset = self._file_to_dataset(file)

            # Filter out identical datasets
            facetset = frozenset(
                (f, frozenset(v) if isinstance(v, list) else v)
                for f, v in dataset.facets.items())
            if facetset not in seen:
                seen.add(facetset)
                if any(_isglob(v) for f, v in dataset.facets.items()
                       if f != 'timerange'):
                    partially_defined.append((dataset, file))
                else:
                    dataset._update_timerange()
                    dataset._supplementaries_from_files()
                    expanded = True
                    yield dataset

        # Only yield datasets with globs if there is no better alternative
        for dataset, file in partially_defined:
            msg = (f"{dataset} with unexpanded wildcards, created from file "
                   f"{file} with facets {file.facets}. Are the missing facets "
                   "in the path to the file?" if isinstance(
                       file, local.LocalFile) else "available on ESGF?")
            if expanded:
                logger.info("Ignoring %s", msg)
            else:
                logger.debug(
                    "Not updating timerange and supplementaries for %s "
                    "because it still contains wildcards.", msg)
                yield dataset

    def from_files(self) -> Iterator['Dataset']:
        """Create datasets based on the available files.

        The facet values for local files are retrieved from the directory tree
        where the directories represent the facets values.
        Reading facet values from file names is not yet supported.
        See :ref:`CMOR-DRS` for more information on this kind of file
        organization.

        :func:`glob.glob` patterns can be used as facet values to select
        multiple datasets.
        If for some of the datasets not all glob patterns can be expanded
        (e.g. because the required facet values cannot be inferred from the
        directory names), these datasets will be ignored, unless this happens
        to be all datasets.

        If :func:`glob.glob` patterns are used in supplementary variables and
        multiple matching datasets are found, only the supplementary dataset
        that has most facets in common with the main dataset will be attached.

        Supplementary datasets will in inherit the facet values from the main
        dataset for those facets listed in :obj:`INHERITED_FACETS`.

        Examples
        --------
        See :ref:`/notebooks/discovering-data.ipynb` for example use cases.

        Yields
        ------
        Dataset
            Datasets representing the available files.
        """
        expanded = False
        if any(_isglob(v) for v in self.facets.values()):
            if _isglob(self.facets['mip']):
                available_mips = _get_mips(
                    self.facets['project'],  # type: ignore
                    self.facets['short_name'],  # type: ignore
                )
                mips = [
                    mip for mip in available_mips
                    if _ismatch(mip, self.facets['mip'])
                ]
            else:
                mips = [self.facets['mip']]  # type: ignore

            for mip in mips:
                dataset_template = self.copy(mip=mip)
                for dataset in dataset_template._get_available_datasets():
                    expanded = True
                    yield dataset

        if not expanded:
            # If the definition contains no wildcards, no files were found,
            # or the file facets didn't match the specification, yield the
            # original, but do expand any supplementary globs.
            self._supplementaries_from_files()
            yield self

    def _supplementaries_from_files(self) -> None:
        """Expand wildcards in supplementary datasets."""
        supplementaries: list[Dataset] = []
        for supplementary_ds in self.supplementaries:
            for facet in INHERITED_FACETS:
                if facet in self.facets:
                    supplementary_ds.facets[facet] = self.facets[facet]
            supplementaries.extend(supplementary_ds.from_files())
        self.supplementaries = supplementaries
        self._remove_unexpanded_supplementaries()
        self._remove_duplicate_supplementaries()
        self._fix_fx_exp()

    def _remove_unexpanded_supplementaries(self) -> None:
        """Remove supplementaries where wildcards could not be expanded."""
        supplementaries = []
        for supplementary_ds in self.supplementaries:
            unexpanded = [
                f for f, v in supplementary_ds.facets.items() if _isglob(v)
            ]
            if unexpanded:
                logger.info(
                    "For %s: ignoring supplementary variable '%s', "
                    "unable to expand wildcards %s.",
                    self.summary(shorten=True),
                    supplementary_ds.facets['short_name'],
                    ", ".join(f"'{f}'" for f in unexpanded),
                )
            else:
                supplementaries.append(supplementary_ds)
        self.supplementaries = supplementaries

    def _match(self, other: Dataset) -> int:
        """Compute the match between two datasets."""
        score = 0
        for facet, value2 in self.facets.items():
            if facet in other.facets:
                value1 = other.facets[facet]
                if isinstance(value1, (list, tuple)):
                    if isinstance(value2, (list, tuple)):
                        score += any(elem in value2 for elem in value1)
                    else:
                        score += value2 in value1
                else:
                    if isinstance(value2, (list, tuple)):
                        score += value1 in value2
                    else:
                        score += value1 == value2
        return score

    def _remove_duplicate_supplementaries(self) -> None:
        """Remove supplementaries that are duplicates."""
        not_used = []
        supplementaries = list(self.supplementaries)
        self.supplementaries.clear()
        for _, duplicates in groupby(supplementaries,
                                     key=lambda ds: ds['short_name']):
            group = sorted(duplicates, key=self._match, reverse=True)
            self.supplementaries.append(group[0])
            not_used.extend(group[1:])

        if not_used:
            logger.debug(
                "List of all supplementary datasets found for %s:\n%s",
                self.summary(shorten=True),
                "\n".join(
                    sorted(ds.summary(shorten=True)
                           for ds in supplementaries)),
            )

    def _fix_fx_exp(self) -> None:
        for supplementary_ds in self.supplementaries:
            exps = supplementary_ds.facets.get('exp')
            frequency = supplementary_ds.facets.get('frequency')
            if isinstance(exps, list) and len(exps) > 1 and frequency == 'fx':
                for exp in exps:
                    dataset = supplementary_ds.copy(exp=exp)
                    if dataset.files:
                        supplementary_ds.facets['exp'] = exp
                        logger.info(
                            "Corrected wrong 'exp' from '%s' to '%s' for "
                            "supplementary variable '%s' of %s", exps, exp,
                            supplementary_ds.facets['short_name'],
                            self.summary(shorten=True))
                        break

    def copy(self, **facets: FacetValue) -> 'Dataset':
        """Create a copy.

        Parameters
        ----------
        **facets
            Update these facets in the copy. Note that for supplementary
            datasets attached to the dataset, the ``'short_name'`` and
            ``'mip'`` facets will not be updated with these values.

        Returns
        -------
        Dataset
            A copy of the dataset.
        """
        new = self.__class__()
        new._session = self._session
        for key, value in self.facets.items():
            new.set_facet(key, deepcopy(value), key in self._persist)
        for key, value in facets.items():
            new.set_facet(key, deepcopy(value))
        for supplementary in self.supplementaries:
            # The short_name and mip of the supplementary variable are probably
            # different from the main variable, so don't copy those facets.
            skip = ('short_name', 'mip')
            supplementary_facets = {
                k: v
                for k, v in facets.items() if k not in skip
            }
            new_supplementary = supplementary.copy(**supplementary_facets)
            new.supplementaries.append(new_supplementary)
        return new

    def __eq__(self, other) -> bool:
        """Compare with another dataset."""
        return (isinstance(other, self.__class__)
                and self._session == other._session
                and self.facets == other.facets
                and self.supplementaries == other.supplementaries)

    def __repr__(self) -> str:
        """Create a string representation."""
        first_keys = (
            'diagnostic',
            'variable_group',
            'dataset',
            'project',
            'mip',
            'short_name',
        )

        def facets2str(facets):

            view = {k: facets[k] for k in first_keys if k in facets}
            for key, value in sorted(facets.items()):
                if key not in first_keys:
                    view[key] = value

            return pprint.pformat(view, sort_dicts=False)

        txt = [
            f"{self.__class__.__name__}:",
            facets2str(self.facets),
        ]
        if self.supplementaries:
            txt.append("supplementaries:")
            txt.extend(
                textwrap.indent(facets2str(a.facets), "  ")
                for a in self.supplementaries)
        if self._session:
            txt.append(f"session: '{self.session.session_name}'")
        return "\n".join(txt)

    def _get_joined_summary_facets(
        self,
        separator: str,
        join_lists: bool = False,
    ) -> str:
        """Get string consisting of joined summary facets."""
        summary_facets_vals = []
        for key in self._SUMMARY_FACETS:
            if key not in self.facets:
                continue
            val = self.facets[key]
            if join_lists and isinstance(val, (tuple, list)):
                val = '-'.join(str(elem) for elem in val)
            else:
                val = str(val)
            summary_facets_vals.append(val)
        return separator.join(summary_facets_vals)

    def summary(self, shorten: bool = False) -> str:
        """Summarize the content of dataset.

        Parameters
        ----------
        shorten
            Shorten the summary.

        Returns
        -------
        str
            A summary describing the dataset.
        """
        if not shorten:
            return repr(self)

        title = self.__class__.__name__
        txt = f"{title}: " + self._get_joined_summary_facets(', ')

        def supplementary_summary(dataset):
            return ", ".join(
                str(dataset.facets[k]) for k in self._SUMMARY_FACETS
                if k in dataset.facets and dataset[k] != self.facets.get(k))

        if self.supplementaries:
            txt += (", supplementaries: " + "; ".join(
                supplementary_summary(a) for a in self.supplementaries) + "")
        return txt

    def __getitem__(self, key):
        """Get a facet value."""
        return self.facets[key]

    def __setitem__(self, key, value):
        """Set a facet value."""
        self.facets[key] = value

    def set_facet(self, key: str, value: FacetValue, persist: bool = True):
        """Set facet.

        Parameters
        ----------
        key
            The name of the facet.
        value
            The value of the facet.
        persist
            When writing a dataset to a recipe, only persistent facets
            will get written.
        """
        self.facets[key] = value
        if persist:
            self._persist.add(key)

    @property
    def minimal_facets(self) -> Facets:
        """Return a dictionary with the persistent facets."""
        return {k: v for k, v in self.facets.items() if k in self._persist}

    def set_version(self) -> None:
        """Set the ``'version'`` facet based on the available data."""
        versions: set[str] = set()
        for file in self.files:
            if 'version' in file.facets:
                versions.add(file.facets['version'])  # type: ignore
        version = versions.pop() if len(versions) == 1 else sorted(versions)
        if version:
            self.set_facet('version', version)
        for supplementary_ds in self.supplementaries:
            supplementary_ds.set_version()

    @property
    def session(self) -> Session:
        """A :obj:`esmvalcore.config.Session` associated with the dataset."""
        if self._session is None:
            session_name = f"session-{uuid.uuid4()}"
            self._session = CFG.start_session(session_name)
        return self._session

    @session.setter
    def session(self, session: Session | None) -> None:
        self._session = session
        for supplementary in self.supplementaries:
            supplementary._session = session

    def add_supplementary(self, **facets: FacetValue) -> None:
        """Add an supplementary dataset.

        This is a convenience function that will create a copy of the current
        dataset, update its facets with the values specified in ``**facets``,
        and append it to :obj:`Dataset.supplementaries`. For more control
        over the creation of the supplementary dataset, first create a new
        :class:`Dataset` describing the supplementary dataset and then append
        it to :obj:`Dataset.supplementaries`.

        Parameters
        ----------
        **facets
            Facets describing the supplementary variable.
        """
        supplementary = self.copy(**facets)
        supplementary.supplementaries = []
        self.supplementaries.append(supplementary)

    def augment_facets(self) -> None:
        """Add extra facets.

        This function will update the dataset with additional facets
        from various sources.
        """
        self._augment_facets()
        for supplementary in self.supplementaries:
            supplementary._augment_facets()

    def _augment_facets(self):
        extra_facets = get_extra_facets(self, self.session['extra_facets_dir'])
        _augment(self.facets, extra_facets)
        if 'institute' not in self.facets:
            institute = get_institutes(self.facets)
            if institute:
                self.facets['institute'] = institute
        if 'activity' not in self.facets:
            activity = get_activity(self.facets)
            if activity:
                self.facets['activity'] = activity
        _update_cmor_facets(self.facets)
        if self.facets.get('frequency') == 'fx':
            self.facets.pop('timerange', None)

    def find_files(self) -> None:
        """Find files.

        Look for files and populate the :obj:`Dataset.files` property of
        the dataset and its supplementary datasets.
        """
        self.augment_facets()

        if _isglob(self.facets.get('timerange')):
            self._update_timerange()

        self._find_files()
        for supplementary in self.supplementaries:
            supplementary.find_files()

    def _find_files(self) -> None:
        self.files, self._file_globs = local.find_files(
            debug=True,
            **self.facets,
        )

        # If project does not support automatic downloads from ESGF, stop here
        if self.facets['project'] not in esgf.facets.FACETS:
            return

        # 'never' mode: never download files from ESGF and stop here
        if self.session['search_esgf'] == 'never':
            return

        # 'when_missing' mode: if files are available locally, do not check
        # ESGF
        if self.session['search_esgf'] == 'when_missing':
            try:
                check.data_availability(self, log=False)
            except InputFilesNotFound:
                pass  # search ESGF for files
            else:
                return  # use local files

        # Local files are not available in 'when_missing' mode or 'always' mode
        # is used: check ESGF
        local_files = {f.name: f for f in self.files}
        search_result = esgf.find_files(**self.facets)
        for file in search_result:
            if file.name not in local_files:
                # Use ESGF files that are not available locally.
                self.files.append(file)
            else:
                # Use ESGF files that are newer than the locally available
                # files.
                local_file = local_files[file.name]
                if 'version' in local_file.facets:
                    if file.facets['version'] > local_file.facets['version']:
                        idx = self.files.index(local_file)
                        self.files[idx] = file

    @property
    def files(self) -> Sequence[File]:
        """The files associated with this dataset."""
        if self._files is None:
            self.find_files()
        return self._files  # type: ignore

    @files.setter
    def files(self, value):
        self._files = value

    def load(self) -> Cube:
        """Load dataset.

        Raises
        ------
        InputFilesNotFound
            When no files were found.

        Returns
        -------
        iris.cube.Cube
            An :mod:`iris` cube with the data corresponding the the dataset.
        """
        input_files = list(self.files)
        for supplementary_dataset in self.supplementaries:
            input_files.extend(supplementary_dataset.files)
        esgf.download(input_files, self.session['download_dir'])

        cube = self._load()
        supplementary_cubes = []
        for supplementary_dataset in self.supplementaries:
            supplementary_cube = supplementary_dataset._load()
            supplementary_cubes.append(supplementary_cube)

        output_file = _get_output_file(self.facets, self.session.preproc_dir)
        cubes = preprocess(
            [cube],
            'add_supplementary_variables',
            input_files=input_files,
            output_file=output_file,
            debug=self.session['save_intermediary_cubes'],
            supplementary_cubes=supplementary_cubes,
        )

        return cubes[0]

    def _load(self) -> Cube:
        """Load self.files into an iris cube and return it."""
        if not self.files:
            lines = [
                f"No files were found for {self}",
                "locally using glob patterns:",
                "\n".join(str(f) for f in self._file_globs or []),
            ]
            if self.session['search_esgf'] != 'never':
                lines.append('or on ESGF.')
            msg = "\n".join(lines)
            raise InputFilesNotFound(msg)

        output_file = _get_output_file(self.facets, self.session.preproc_dir)
        fix_dir_prefix = Path(
            self.session._fixed_file_dir,
            self._get_joined_summary_facets('_', join_lists=True) + '_',
        )

        settings: dict[str, dict[str, Any]] = {}
        settings['fix_file'] = {
            'output_dir': fix_dir_prefix,
            'add_unique_suffix': True,
            'session': self.session,
            **self.facets,
        }
        settings['load'] = {
            'ignore_warnings': get_ignored_warnings(
                self.facets['project'], 'load'
            ),
        }
        settings['fix_metadata'] = {
            'check_level': self.session['check_level'],
            'session': self.session,
            **self.facets,
        }
        settings['concatenate'] = {
            'check_level': self.session['check_level']
        }
        settings['cmor_check_metadata'] = {
            'check_level': self.session['check_level'],
            'cmor_table': self.facets['project'],
            'mip': self.facets['mip'],
            'frequency': self.facets['frequency'],
            'short_name': self.facets['short_name'],
        }
        if 'timerange' in self.facets:
            settings['clip_timerange'] = {
                'timerange': self.facets['timerange'],
            }
        settings['fix_data'] = {
            'check_level': self.session['check_level'],
            'session': self.session,
            **self.facets,
        }
        settings['cmor_check_data'] = {
            'check_level': self.session['check_level'],
            'cmor_table': self.facets['project'],
            'mip': self.facets['mip'],
            'frequency': self.facets['frequency'],
            'short_name': self.facets['short_name'],
        }

        result = [
            file.local_file(self.session['download_dir']) if isinstance(
                file, esgf.ESGFFile) else file for file in self.files
        ]
        for step, kwargs in settings.items():
            result = preprocess(
                result,
                step,
                input_files=self.files,
                output_file=output_file,
                debug=self.session['save_intermediary_cubes'],
                **kwargs,
            )

        cube = result[0]
        return cube

    def from_ranges(self) -> list['Dataset']:
        """Create a list of datasets from short notations.

        This expands the ``'ensemble'`` and ``'sub_experiment'`` facets in the
        dataset definition if they are ranges.

        For example ``'ensemble'='r(1:3)i1p1f1'`` will be expanded to
        three datasets, with ``'ensemble'`` values ``'r1i1p1f1'``,
        ``'r2i1p1f1'``, ``'r3i1p1f1'``.

        Returns
        -------
        list[Dataset]
            The datasets.
        """
        datasets = [self]
        for key in 'ensemble', 'sub_experiment':
            if key in self.facets:
                datasets = [
                    ds.copy(**{key: value}) for ds in datasets
                    for value in ds._expand_range(key)
                ]
        return datasets

    def _expand_range(self, input_tag):
        """Expand ranges such as ensemble members or start dates.

        Expansion only supports ensembles defined as strings, not lists.
        """
        expanded = []
        regex = re.compile(r'\(\d+:\d+\)')

        def expand_range(input_range):
            match = regex.search(input_range)
            if match:
                start, end = match.group(0)[1:-1].split(':')
                for i in range(int(start), int(end) + 1):
                    range_ = regex.sub(str(i), input_range, 1)
                    expand_range(range_)
            else:
                expanded.append(input_range)

        tag = self.facets.get(input_tag, "")
        if isinstance(tag, (list, tuple)):
            for elem in tag:
                if regex.search(elem):
                    raise RecipeError(
                        f"In {self}: {input_tag} expansion "
                        f"cannot be combined with {input_tag} lists")
            expanded.append(tag)
        else:
            expand_range(tag)

        return expanded

    def _update_timerange(self):
        """Update wildcards in timerange with found datetime values.

        If the timerange is given as a year, it ensures it's formatted
        as a 4-digit value (YYYY).
        """
        dataset = self.copy()
        dataset.supplementaries = []
        dataset.augment_facets()
        if 'timerange' not in dataset.facets:
            self.facets.pop('timerange', None)
            return

        timerange = self.facets['timerange']
        if not isinstance(timerange, str):
            raise TypeError(
                f"timerange should be a string, got '{timerange!r}'")
        check.valid_time_selection(timerange)

        if '*' in timerange:
            dataset = self.copy()
            dataset.facets.pop('timerange')
            dataset.supplementaries = []
            check.data_availability(dataset)
            intervals = [_get_start_end_date(f) for f in dataset.files]

            min_date = min(interval[0] for interval in intervals)
            max_date = max(interval[1] for interval in intervals)

            if timerange == '*':
                timerange = f'{min_date}/{max_date}'
            if '*' in timerange.split('/')[0]:
                timerange = timerange.replace('*', min_date)
            if '*' in timerange.split('/')[1]:
                timerange = timerange.replace('*', max_date)

        # Make sure that years are in format YYYY
        start_date, end_date = timerange.split('/')
        timerange = _dates_to_timerange(start_date, end_date)
        check.valid_time_selection(timerange)

        self.set_facet('timerange', timerange)
