"""Classes and functions for defining, finding, and loading data."""
from __future__ import annotations

import logging
import pprint
import re
import textwrap
import uuid
from copy import deepcopy
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union

from iris.cube import Cube

from . import _recipe_checks as check
from . import esgf, local
from .cmor.table import _update_cmor_facets
from .config import CFG, Session
from .config._config import get_activity, get_extra_facets, get_institutes
from .exceptions import InputFilesNotFound, RecipeError
from .local import _dates_to_timerange, _get_output_file, _get_start_end_date
from .preprocessor import preprocess
from .typing import Facets, FacetValue

__all__ = [
    'Dataset',
]

logger = logging.getLogger(__name__)

File = Union[esgf.ESGFFile, local.LocalFile]


def _augment(base, update):
    """Update dict base with values from dict update."""
    for key in update:
        if key not in base:
            base[key] = update[key]


def _isglob(facet_value: Union[FacetValue, None]) -> bool:
    if isinstance(facet_value, str):
        return bool(re.match(r'.*[\*\?]+.*|.*\[.*\].*', facet_value))
    return False


def _ismatch(facet_value: FacetValue, pattern: FacetValue) -> bool:
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
    ancillaries : list[Dataset]
        List of ancillary datasets.
    facets: :obj:`esmvalcore.typing.Facets`
        Facets describing the dataset.
    """

    def __init__(self, **facets: FacetValue):

        self.facets: Facets = {}
        self.ancillaries: list['Dataset'] = []

        self._persist: set[str] = set()
        self._session: Optional[Session] = None
        self._files: Optional[Sequence[File]] = None
        self._files_debug: Optional[Sequence[Path]] = None

        for key, value in facets.items():
            self.set_facet(key, deepcopy(value), persist=True)

    def from_files(self) -> Iterator['Dataset']:
        """Create a list of datasets from the available files.

        Yields
        ------
        Dataset
            Datasets representing the available files.
        """

        def same(facets_a, facets_b):
            """Define when two sets of facets are the same."""
            return facets_a.issubset(facets_b) or facets_b.issubset(facets_a)

        dataset = self.copy()
        dataset.ancillaries = []
        timerange = dataset.facets.get('timerange')
        if _isglob(timerange):
            # Remove wildcard `timerange` facet, because data finding cannot
            # handle it
            dataset.facets.pop('timerange')

        expanded = False
        if any(_isglob(v) for v in self.facets.values()):
            available_facets: list[frozenset[tuple[str, FacetValue]]] = []
            for file in dataset.files:
                facets = dict(file.facets)
                if 'version' not in self.facets:
                    # Remove version facet if no specific version requested
                    facets.pop('version', None)

                facetset = frozenset(facets.items())

                # Filter out identical facetsets
                for prev_facetset in available_facets:
                    if same(facetset, prev_facetset):
                        break
                else:
                    available_facets.append(facetset)

            for facetset in sorted(available_facets):
                updated_facets = {
                    k: v
                    for k, v in facetset
                    if k in self.facets and _isglob(self.facets[k])
                    and _ismatch(v, self.facets[k])
                }
                new_ds = self.copy()
                new_ds.facets.update(updated_facets)

                if timerange is not None:
                    new_ds['timerange'] = timerange
                    new_ds._update_timerange()

                ancillaries: list['Dataset'] = []
                for ancillary_ds in new_ds.ancillaries:
                    afacets = ancillary_ds.facets
                    for key, value in updated_facets.items():
                        if (key in afacets and _isglob(afacets[key])):
                            # Only overwrite ancillary facets that were globs.
                            afacets[key] = value
                    ancillaries.extend(ancillary_ds.from_files())
                new_ds.ancillaries = ancillaries

                expanded = True
                yield new_ds

        if not expanded:
            # If the definition contains no wildcards or no files were found,
            # yield the original (but do expand any ancillary globs).
            ancillaries = []
            for ancillary_ds in self.ancillaries:
                ancillaries.extend(ancillary_ds.from_files())
            self.ancillaries = ancillaries
            yield self

    def copy(self, **facets: FacetValue) -> 'Dataset':
        """Create a copy.

        Parameters
        ----------
        **facets
            Update these facets in the copy. Note that for ancillary datasets
            attached to the dataset, the 'short_name' and 'mip' facets will
            not be updated with these values.

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
        for ancillary in self.ancillaries:
            # The short_name and mip of the ancillary variable are probably
            # different from the main variable, so don't copy those facets.
            skip = ('short_name', 'mip')
            ancillary_facets = {k: facets[k] for k in facets if k not in skip}
            new_ancillary = ancillary.copy(**ancillary_facets)
            new.ancillaries.append(new_ancillary)
        return new

    def __eq__(self, other) -> bool:
        try:
            other_session = other.session
        except ValueError:
            other_session = None
        return (isinstance(other, self.__class__)
                and self._session == other_session
                and self.facets == other.facets
                and self.ancillaries == other.ancillaries)

    def __repr__(self) -> str:

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
        if self.ancillaries:
            txt.append("ancillaries:")
            txt.extend(
                textwrap.indent(facets2str(a.facets), "  ")
                for a in self.ancillaries)
        return "\n".join(txt)

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

        keys = (
            'project',
            'mip',
            'short_name',
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
        return (
            f"{self.__class__.__name__}: " +
            ", ".join(str(self.facets[k]) for k in keys if k in self.facets))

    def __getitem__(self, key):
        return self.facets[key]

    def __setitem__(self, key, value):
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
        """A dictionary with the persistent facets."""
        return {k: v for k, v in self.facets.items() if k in self._persist}

    def set_version(self) -> None:
        """Set the 'version' facet based on the available data."""
        versions: set[str] = set()
        for file in self.files:
            if 'version' in file.facets:
                versions.add(file.facets['version'])  # type: ignore
        version = versions.pop() if len(versions) == 1 else sorted(versions)
        if version:
            self.set_facet('version', version)
        for ancillary_ds in self.ancillaries:
            ancillary_ds.set_version()

    @property
    def session(self) -> Session:
        """A :obj:`esmvalcore.config.Session` associated with the dataset."""
        if self._session is None:
            session_name = f"session-{uuid.uuid4()}"
            self._session = CFG.start_session(session_name)
        return self._session

    @session.setter
    def session(self, session: Optional[Session]) -> None:
        self._session = session
        for ancillary in self.ancillaries:
            ancillary._session = session

    def add_ancillary(self, **facets: FacetValue) -> None:
        """Add an ancillary dataset.

        Parameters
        ----------
        **facets
            Facets describing the ancillary variable.
        """
        ancillary = self.copy(**facets)
        ancillary.ancillaries = []
        self.ancillaries.append(ancillary)

    def augment_facets(self) -> None:
        """Add extra facets.

        This function will update the dataset with additional facets
        from various sources.
        """
        self._augment_facets()
        for ancillary in self.ancillaries:
            ancillary._augment_facets()

    def _augment_facets(self):
        extra_facets = get_extra_facets(
            project=self.facets['project'],
            dataset=self.facets['dataset'],
            mip=self.facets['mip'],
            short_name=self.facets['short_name'],
            extra_facets_dir=self.session['extra_facets_dir'],
        )
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
        the dataset and its ancillary datasets.
        """
        self.augment_facets()

        if _isglob(self.facets.get('timerange')):
            self._update_timerange()

        self._find_files()
        for ancillary in self.ancillaries:
            ancillary.find_files()

    def _find_files(self) -> None:
        self.files, self._files_debug = local.find_files(
            debug=True,
            **self.facets,
        )
        project = self.facets['project']

        # Set up downloading from ESGF if requested.
        search_esgf = False
        if not self.session['offline'] and project in esgf.facets.FACETS:
            try:
                check.data_availability(
                    self.files,
                    self.facets,
                    self._files_debug,
                    log=False,
                )
            except InputFilesNotFound:
                search_esgf = True

        if search_esgf:
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
                        if file.facets['version'] > local_file.facets[
                                'version']:
                            idx = self.files.index(local_file)
                            self.files[idx] = file

    @property
    def files(self) -> Sequence[File]:
        """A list of files associated with this dataset."""
        if self._files is None:
            self.find_files()
        return self._files  # type: ignore

    @files.setter
    def files(self, value):
        self._files = value

    def load(self) -> Cube:
        """Load dataset.

        Returns
        -------
        iris.cube.Cube
            An :mod:`iris` cube with the data corresponding the the dataset.
        """
        cube = self._load()

        input_files = list(self.files)
        ancillary_cubes = []
        for ancillary_dataset in self.ancillaries:
            if ancillary_dataset.files:
                input_files.extend(ancillary_dataset.files)
                ancillary_cube = ancillary_dataset._load()
                ancillary_cubes.append(ancillary_cube)

        cubes = preprocess(
            [cube],
            'add_ancillary_variables',
            input_files=input_files,
            ancillary_cubes=ancillary_cubes,
        )
        return cubes[0]

    def _load(self) -> Cube:
        """Load self.files into an iris cube and return it."""
        output_file = _get_output_file(self.facets, self.session.preproc_dir)

        settings: dict[str, dict[str, Any]] = {}
        settings['fix_file'] = {
            'output_dir': Path(f"{output_file.with_suffix('')}_fixed"),
            **self.facets,
        }
        settings['load'] = {}
        settings['fix_metadata'] = {
            'check_level': self.session['check_level'],
            **self.facets,
        }
        settings['concatenate'] = {}
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
            **self.facets,
        }
        settings['cmor_check_data'] = {
            'check_level': self.session['check_level'],
            'cmor_table': self.facets['project'],
            'mip': self.facets['mip'],
            'frequency': self.facets['frequency'],
            'short_name': self.facets['short_name'],
        }

        result = self.files
        for step, kwargs in settings.items():
            result = preprocess(result, step, input_files=self.files, **kwargs)
        cube = result[0]
        return cube

    def from_ranges(self) -> list['Dataset']:
        """Factory function that expands shorthands to generate datasets.

        This expands the 'ensemble' and 'sub_experiment' facets in the
        dataset definition if they are ranges.

        For example 'ensemble: r(1:3)i1p1f1' will be expanded to
        three datasets, with 'ensemble' values 'r1i1p1f1', 'r2i1p1f1',
        'r3i1p1f1'.

        Returns
        -------
        list[Dataset]
            A list of datasets.
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
        """Expand ranges such as ensemble members or stardates.

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
        if 'timerange' not in self.facets:
            return

        timerange = self.facets.pop('timerange')
        if not isinstance(timerange, str):
            raise TypeError(f"timerange should be a string, got {timerange!r}")
        check.valid_time_selection(timerange)

        if '*' in timerange:
            self.find_files()
            check.data_availability(self.files, self.facets, self._files_debug)
            intervals = [_get_start_end_date(f.name) for f in self.files]
            self._files = None

            min_date = min(interval[0] for interval in intervals)
            max_date = max(interval[1] for interval in intervals)

            if timerange == '*':
                timerange = f'{min_date}/{max_date}'
            if '*' in timerange.split('/')[0]:
                timerange = timerange.replace('*', min_date)
            if '*' in timerange.split('/')[1]:
                timerange = timerange.replace('*', max_date)

        # Make sure that years are in format YYYY
        (start_date, end_date) = timerange.split('/')
        timerange = _dates_to_timerange(start_date, end_date)
        check.valid_time_selection(timerange)

        self.set_facet('timerange', timerange)