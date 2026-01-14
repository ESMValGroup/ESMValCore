"""Classes and functions for defining, finding, and loading data."""

from __future__ import annotations

import fnmatch
import logging
import os
import pprint
import re
import textwrap
import uuid
from copy import deepcopy
from fnmatch import fnmatchcase
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any

from esmvalcore import esgf
from esmvalcore._recipe import check
from esmvalcore._recipe.from_datasets import datasets_to_recipe
from esmvalcore.cmor.table import _get_mips, _update_cmor_facets
from esmvalcore.config import CFG
from esmvalcore.config._config import (
    get_activity,
    get_institutes,
    load_extra_facets,
)
from esmvalcore.config._data_sources import _get_data_sources
from esmvalcore.exceptions import InputFilesNotFound, RecipeError
from esmvalcore.io.local import _dates_to_timerange
from esmvalcore.preprocessor import _get_preprocessor_filename, preprocess
from esmvalcore.preprocessor._derive import get_required

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from iris.cube import Cube

    from esmvalcore.config import Session
    from esmvalcore.io.protocol import DataElement, DataSource
    from esmvalcore.preprocessor import PreprocessorItem
    from esmvalcore.typing import Facets, FacetValue

__all__ = [
    "INHERITED_FACETS",
    "Dataset",
    "datasets_to_recipe",
]

logger = logging.getLogger(__name__)

INHERITED_FACETS: list[str] = [
    "dataset",
    "domain",
    "driver",
    "grid",
    "project",
    "timerange",
]
"""Inherited facets.

Supplementary datasets created based on the available files using the
:func:`Dataset.from_files` method will inherit the values of these facets from
the main dataset.
"""


def _augment(base: dict, update: dict) -> None:
    """Update dict `base` with values from dict `update`."""
    for key, value in update.items():
        if key not in base:
            base[key] = value


def _isglob(facet_value: FacetValue | None) -> bool:
    """Check if a facet value is a glob pattern."""
    return isinstance(facet_value, str) and bool(
        re.match(r".*[\*\?]+.*|.*\[.*\].*", facet_value),
    )


def _ismatch(facet_value: FacetValue, pattern: FacetValue) -> bool:
    """Check if a facet value matches a glob pattern."""
    return (
        isinstance(pattern, str)
        and isinstance(facet_value, str)
        and fnmatchcase(facet_value, pattern)
    )


class Dataset:
    """Define datasets, find the related files, and load them.

    Parameters
    ----------
    **facets
        Facets describing the dataset. See
        :obj:`esmvalcore.io.esgf.facets.FACETS` for the mapping between
        the facet names used by ESMValCore and those used on ESGF.

    Attributes
    ----------
    supplementaries: list[Dataset]
        List of supplementary datasets.
    facets: :obj:`esmvalcore.typing.Facets`
        Facets describing the dataset.
    """

    _SUMMARY_FACETS: tuple[str, ...] = (
        "short_name",
        "mip",
        "project",
        "dataset",
        "rcm_version",
        "driver",
        "domain",
        "activity",
        "exp",
        "ensemble",
        "grid",
        "version",
    )
    """Facets used to create a summary of a Dataset instance."""

    def __init__(self, **facets: FacetValue) -> None:
        self.facets: Facets = {}
        self.supplementaries: list[Dataset] = []

        self._persist: set[str] = set()
        self._session: Session | None = None
        self._files: Sequence[DataElement] | None = None
        self._used_data_sources: Sequence[DataSource] = []
        self._required_datasets: list[Dataset] | None = None

        for key, value in facets.items():
            self.set_facet(key, deepcopy(value), persist=True)

    @staticmethod
    def from_recipe(
        recipe: Path | str | dict,
        session: Session,
    ) -> list[Dataset]:
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
        from esmvalcore._recipe.to_datasets import (  # noqa: PLC0415
            datasets_from_recipe,
        )

        return datasets_from_recipe(recipe, session)

    def _is_derived(self) -> bool:
        """Return ``True`` for derived variables, ``False`` otherwise."""
        return bool(self.facets.get("derive", False))

    def _is_force_derived(self) -> bool:
        """Return ``True`` for force-derived variables, ``False`` otherwise."""
        return self._is_derived() and bool(
            self.facets.get("force_derivation", False),
        )

    def _derivation_necessary(self) -> bool:
        """Return ``True`` if derivation is necessary, ``False`` otherwise."""
        # If variable cannot be derived, derivation is not necessary
        if not self._is_derived():
            return False

        # If forced derivation is requested, derivation is necessary
        if self._is_force_derived():
            return True

        # Otherwise, derivation is necessary if no files for the self dataset
        # are found
        ds_copy = self.copy()
        ds_copy.supplementaries = []

        # Avoid potential errors from missing data during timerange glob
        # expansion
        if _isglob(ds_copy.facets.get("timerange", "")):
            ds_copy.facets.pop("timerange", None)

        return not ds_copy.files

    def _get_required_datasets(self) -> list[Dataset]:
        """Get required datasets for derivation."""
        required_datasets: list[Dataset] = []
        required_vars_facets = get_required(
            self.facets["short_name"],  # type: ignore
            self.facets["project"],  # type: ignore
        )

        for required_facets in required_vars_facets:
            required_dataset = self._copy(derive=False, force_derivation=False)
            keep = {"alias", "recipe_dataset_index", *self.minimal_facets}
            required_dataset.facets = {
                k: v for k, v in required_dataset.facets.items() if k in keep
            }
            required_dataset.facets.update(required_facets)
            required_dataset.augment_facets()
            required_datasets.append(required_dataset)

        return required_datasets

    @property
    def required_datasets(self) -> list[Dataset]:
        """Get required datasets.

        For non-derived variables (i.e., those with facet ``derive=False``),
        this will simply return the dataset itself in a list.

        For derived variables (i.e., those with facet ``derive=True``), this
        will return the datasets required for derivation if derivation is
        necessary, and the dataset itself if derivation is not necessary.
        Derivation is necessary if the facet ``force_derivation=True`` is set
        or no files for the dataset itself are available.

        See also :func:`esmvalcore.preprocessor.derive` for an example usage.

        """
        if self._required_datasets is not None:
            return self._required_datasets

        if not self._derivation_necessary():
            self._required_datasets = [self]
        else:
            self._required_datasets = self._get_required_datasets()

        return self._required_datasets

    @staticmethod
    def _file_to_dataset(
        dataset: Dataset,
        file: DataElement,
    ) -> Dataset:
        """Create a dataset from a file with a `facets` attribute."""
        facets = dict(file.facets)
        if "version" not in dataset.facets:
            # Remove version facet if no specific version requested
            facets.pop("version", None)

        updated_facets = {
            f: v
            for f, v in facets.items()
            if f in dataset.facets
            and _isglob(dataset.facets[f])
            and _ismatch(v, dataset.facets[f])
        }
        new_dataset = dataset.copy()
        new_dataset.facets.update(updated_facets)

        # If possible, remove unexpanded facets that can be automatically
        # populated.
        unexpanded = {f for f, v in new_dataset.facets.items() if _isglob(v)}
        required_for_augment = {"project", "mip", "short_name", "dataset"}
        if unexpanded and not unexpanded & required_for_augment:
            copy = new_dataset.copy()
            copy.supplementaries = []
            for facet in unexpanded:
                copy.facets.pop(facet)
            copy.augment_facets()
            for facet in unexpanded:
                if facet in copy.facets:
                    new_dataset.facets.pop(facet)

        return new_dataset

    @staticmethod
    def _get_expanded_globs(
        dataset_with_globs: Dataset,
        dataset_with_expanded_globs: Dataset,
    ) -> tuple[tuple[str, FacetValue], ...]:
        """Get facets that have been updated by expanding globs."""
        expanded_globs: dict[str, FacetValue] = {}
        for key, value in dataset_with_globs.facets.items():
            if (
                _isglob(value)
                and key in dataset_with_expanded_globs.facets
                and not _isglob(dataset_with_expanded_globs[key])
            ):
                expanded_globs[key] = dataset_with_expanded_globs[key]
        return tuple(expanded_globs.items())

    @staticmethod
    def _get_all_available_datasets(dataset: Dataset) -> Iterator[Dataset]:
        """Yield datasets based on the available files.

        This function requires that dataset.facets['mip'] is not a glob
        pattern.

        Does take variable derivation into account, i.e., datasets available
        through variable derivation are returned.

        """
        if not dataset._derivation_necessary():
            yield from Dataset._get_available_datasets(dataset)
            return

        # Since we are in full control of the derived variables (the module is
        # private; no custom derivation functions are possible), we can be sure
        # that the following list is never empty
        non_optional_datasets = [
            d
            for d in dataset.required_datasets
            if not d.facets.get("optional", False)
        ]
        if not non_optional_datasets:
            msg = (
                f"Using wildcards to derive {dataset.summary(shorten=True)} "
                f"is not possible, derivation function only requires optional "
                f"variables"
            )
            raise RecipeError(msg)

        # Record all expanded globs from first non-optional required dataset
        # (called "reference_dataset" hereafter)
        reference_dataset = non_optional_datasets[0]
        reference_expanded_globs = {
            Dataset._get_expanded_globs(dataset, ds)
            for ds in Dataset._get_available_datasets(reference_dataset)
        }

        # Iterate through all other non-optional required datasets and only
        # keep those expanded globs which are present for all other
        # non-optional required datasets
        for required_dataset in non_optional_datasets:
            if required_dataset is reference_dataset:
                continue
            new_expanded_globs = {
                Dataset._get_expanded_globs(dataset, ds)
                for ds in Dataset._get_available_datasets(required_dataset)
            }
            reference_expanded_globs &= new_expanded_globs

        # Use the final expanded globs to create new dataset(s)
        for expanded_globs in reference_expanded_globs:
            new_ds = dataset.copy()
            new_ds.facets.update(dict(expanded_globs))
            yield new_ds

    @staticmethod
    def _get_available_datasets(dataset: Dataset) -> Iterator[Dataset]:
        """Yield datasets based on the available files.

        This function requires that self.facets['mip'] is not a glob pattern.

        Does not take variable derivation into account, i.e., datasets
        potentially available through variable derivation are ignored. To
        consider derived variables properly, use the function
        :func:`_get_all_available_datasets`.

        """
        dataset_template = dataset.copy()
        dataset_template.supplementaries = []

        seen = set()
        partially_defined = []
        expanded = False
        for file in dataset_template.files:
            new_dataset = Dataset._file_to_dataset(dataset, file)
            # Do not use the timerange facet from the file because there may be
            # multiple files per dataset.
            new_dataset.facets.pop("timerange", None)
            # Restore the original timerange facet if it was specified.
            if "timerange" in dataset.facets:
                new_dataset.facets["timerange"] = dataset.facets["timerange"]

            # Filter out identical datasets
            facetset = frozenset(
                (f, frozenset(v) if isinstance(v, list) else v)
                for f, v in new_dataset.facets.items()
            )
            if facetset not in seen:
                seen.add(facetset)
                if any(
                    _isglob(v)
                    for f, v in new_dataset.facets.items()
                    if f != "timerange"
                ):
                    partially_defined.append((new_dataset, file))
                else:
                    new_dataset._update_timerange()  # noqa: SLF001
                    expanded = True
                    yield new_dataset

        # Only yield datasets with globs if there is no better alternative
        for new_dataset, file in partially_defined:
            msg = (
                f"{new_dataset} with unexpanded wildcards, created from file "
                f"{file} with facets {file.facets}. Please check why "
                "the missing facets are not available for the file."
                "This will depend on the data source they come from, e.g. can "
                "they be extracted from the path for local files, or are they "
                "available from ESGF when when searching ESGF for files?"
            )
            if expanded:
                logger.info("Ignoring %s", msg)
            else:
                logger.debug(
                    "Not updating timerange and supplementaries for %s "
                    "because it still contains wildcards.",
                    msg,
                )
                yield new_dataset

    def from_files(self) -> Iterator[Dataset]:
        """Create datasets based on the available files.

        The facet values for local files are retrieved from the directory tree
        where the directories represent the facets values.
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

        This also works for :ref:`derived variables <Variable derivation>`. The
        datasets required for derivation can be accessed via
        :attr:`Dataset.required_datasets`.

        Examples
        --------
        See :doc:`/notebooks/discovering-data` notebook for example use cases.

        Yields
        ------
        Dataset
            Datasets representing the available files.
        """
        # No wildcards present -> simply return self with expanded
        # supplementaries
        if not any(_isglob(v) for v in self.facets.values()):
            self._supplementaries_from_files()
            yield self
            return

        # Wildcards present -> expand them
        expanded = False
        if _isglob(self.facets["mip"]):
            available_mips = _get_mips(
                self.facets["project"],  # type: ignore
                self.facets["short_name"],  # type: ignore
            )
            mips = [
                mip
                for mip in available_mips
                if _ismatch(mip, self.facets["mip"])
            ]
        else:
            mips = [self.facets["mip"]]  # type: ignore

        for mip in mips:
            dataset_template = self.copy(mip=mip)
            for dataset in self._get_all_available_datasets(
                dataset_template,
            ):
                dataset._supplementaries_from_files()  # noqa: SLF001
                expanded = True
                yield dataset

        # If files were found, or the file facets didn't match the
        # specification, yield the original, but do expand any supplementary
        # globs. For derived variables, make sure to purge any files found for
        # required variables; those won't match in their facets.
        if not expanded:
            self._supplementaries_from_files()
            if self._derivation_necessary():
                for required_dataset in self.required_datasets:
                    required_dataset.files = []
            yield self

    def _supplementaries_from_files(self) -> None:
        """Expand wildcards in supplementary datasets."""
        supplementaries: list[Dataset] = []
        for supplementary_ds in self.supplementaries:
            for facet in INHERITED_FACETS:
                # allow use of facets from supplementary variable dict
                if (
                    facet in self.facets
                    and facet not in supplementary_ds.facets
                ):
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
                    supplementary_ds.facets["short_name"],
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
                elif isinstance(value2, (list, tuple)):
                    score += value1 in value2
                else:
                    score += value1 == value2
        return score

    def _remove_duplicate_supplementaries(self) -> None:
        """Remove supplementaries that are duplicates."""
        not_used = []
        supplementaries = list(self.supplementaries)
        self.supplementaries.clear()
        for _, duplicates in groupby(
            supplementaries,
            key=lambda ds: ds["short_name"],
        ):
            group = sorted(duplicates, key=self._match, reverse=True)
            self.supplementaries.append(group[0])
            not_used.extend(group[1:])

        if not_used:
            logger.debug(
                "List of all supplementary datasets found for %s:\n%s",
                self.summary(shorten=True),
                "\n".join(
                    sorted(ds.summary(shorten=True) for ds in supplementaries),
                ),
            )

    def _fix_fx_exp(self) -> None:
        for supplementary_ds in self.supplementaries:
            exps = supplementary_ds.facets.get("exp")
            frequency = supplementary_ds.facets.get("frequency")
            if isinstance(exps, list) and len(exps) > 1 and frequency == "fx":
                for exp in exps:
                    dataset = supplementary_ds.copy(exp=exp)
                    if dataset.files:
                        supplementary_ds.facets["exp"] = exp
                        logger.info(
                            "Corrected wrong 'exp' from '%s' to '%s' for "
                            "supplementary variable '%s' of %s",
                            exps,
                            exp,
                            supplementary_ds.facets["short_name"],
                            self.summary(shorten=True),
                        )
                        break

    def _copy(self, **facets: FacetValue) -> Dataset:
        """Create a copy of the parent dataset without supplementaries."""
        new = self.__class__()
        new._session = self._session  # noqa: SLF001
        for key, value in self.facets.items():
            new.set_facet(key, deepcopy(value), key in self._persist)
        for key, value in facets.items():
            new.set_facet(key, deepcopy(value))
        return new

    def copy(self, **facets: FacetValue) -> Dataset:
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
        new = self._copy(**facets)
        for supplementary in self.supplementaries:
            # The short_name and mip of the supplementary variable are probably
            # different from the main variable, so don't copy those facets.
            skip = ("short_name", "mip")
            supplementary_facets = {
                k: v for k, v in facets.items() if k not in skip
            }
            new_supplementary = supplementary.copy(**supplementary_facets)
            new.supplementaries.append(new_supplementary)

        return new

    def __eq__(self, other: object) -> bool:
        """Compare with another dataset."""
        return (
            isinstance(other, self.__class__)
            and self._session == other._session
            and self.facets == other.facets
            and self.supplementaries == other.supplementaries
        )

    def __repr__(self) -> str:
        """Create a string representation."""
        first_keys = (
            "diagnostic",
            "variable_group",
            "dataset",
            "project",
            "mip",
            "short_name",
        )

        def facets2str(facets: Facets) -> str:
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
                textwrap.indent(facets2str(s.facets), "  ")
                for s in self.supplementaries
            )
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
                val = "-".join(str(elem) for elem in val)
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
        txt = f"{title}: " + self._get_joined_summary_facets(", ")

        def supplementary_summary(dataset: Dataset) -> str:
            return ", ".join(
                str(dataset.facets[k])
                for k in self._SUMMARY_FACETS
                if k in dataset.facets and dataset[k] != self.facets.get(k)
            )

        if self.supplementaries:
            txt += (
                ", supplementaries: "
                + "; ".join(
                    supplementary_summary(s) for s in self.supplementaries
                )
                + ""
            )

        return txt

    def __getitem__(self, key: str) -> FacetValue:
        """Get a facet value."""
        return self.facets[key]

    def __setitem__(self, key: str, value: FacetValue) -> None:
        """Set a facet value."""
        self.facets[key] = value

    def set_facet(
        self,
        key: str,
        value: FacetValue,
        persist: bool = True,
    ) -> None:
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

    @staticmethod
    def _get_version(dataset: Dataset) -> str | list[str]:
        """Get available version(s) of dataset."""
        versions: set[str] = set()
        for file in dataset.files:
            if "version" in file.facets:
                versions.add(str(file.facets["version"]))
        return versions.pop() if len(versions) == 1 else sorted(versions)

    def set_version(self) -> None:
        """Set the ``'version'`` facet based on the available data."""
        versions: set[str] = set()
        for required_dataset in self.required_datasets:
            version = self._get_version(required_dataset)
            if version:
                if isinstance(version, list):
                    versions.update(version)
                else:
                    versions.add(version)
        version = versions.pop() if len(versions) == 1 else sorted(versions)
        if version:
            self.set_facet("version", version)

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
            supplementary._session = session  # noqa: SLF001

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
        if self._is_derived():
            facets.setdefault("derive", False)
        if self._is_force_derived():
            facets.setdefault("force_derivation", False)
        supplementary = self.copy(**facets)
        supplementary.supplementaries = []
        self.supplementaries.append(supplementary)

    def augment_facets(self) -> None:
        """Add additional facets.

        This function will update the dataset with additional facets from
        various sources. These include :ref:`config-extra-facets` as well as
        facets read from the controlled voculary included in the CMOR tables
        if applicable.
        """
        self._augment_facets()
        for supplementary in self.supplementaries:
            supplementary._augment_facets()  # noqa: SLF001

    @staticmethod
    def _pattern_filter(patterns: Iterable[str], name: str) -> list[str]:
        """Get the subset of the list `patterns` that `name` matches."""
        return [pat for pat in patterns if fnmatch.fnmatchcase(name, pat)]

    def _get_extra_facets(self) -> dict[str, Any]:
        """Get extra facets of dataset."""
        extra_facets: dict[str, Any] = {}

        raw_extra_facets = (
            self.session["projects"]
            .get(self["project"], {})
            .get("extra_facets", {})
        )
        dataset_names = self._pattern_filter(raw_extra_facets, self["dataset"])  # type: ignore[arg-type]
        for dataset_name in dataset_names:
            mips = self._pattern_filter(
                raw_extra_facets[dataset_name],
                self["mip"],  # type: ignore[arg-type]
            )
            for mip in mips:
                variables = self._pattern_filter(
                    raw_extra_facets[dataset_name][mip],
                    self["short_name"],  # type: ignore[arg-type]
                )
                for var in variables:
                    facets = raw_extra_facets[dataset_name][mip][var]
                    extra_facets.update(facets)

        # Add deprecated user-defined extra facets
        # TODO: remove in v2.15.0
        if os.environ.get("ESMVALTOOL_USE_NEW_EXTRA_FACETS_CONFIG"):
            return extra_facets
        project_details = load_extra_facets(
            self.facets["project"],
            tuple(self.session["extra_facets_dir"]),
        )
        dataset_names = self._pattern_filter(project_details, self["dataset"])  # type: ignore[arg-type]
        for dataset_name in dataset_names:
            mips = self._pattern_filter(
                project_details[dataset_name],
                self["mip"],  # type: ignore[arg-type]
            )
            for mip in mips:
                variables = self._pattern_filter(
                    project_details[dataset_name][mip],
                    self["short_name"],  # type: ignore[arg-type]
                )
                for var in variables:
                    facets = project_details[dataset_name][mip][var]
                    extra_facets.update(facets)

        return extra_facets

    def _augment_facets(self) -> None:
        extra_facets = self._get_extra_facets()
        _augment(self.facets, extra_facets)
        if "institute" not in self.facets:
            institute = get_institutes(self.facets)
            if institute:
                self.facets["institute"] = institute
        if "activity" not in self.facets:
            activity = get_activity(self.facets)
            if activity:
                self.facets["activity"] = activity
        _update_cmor_facets(self.facets)
        if self.facets.get("frequency") == "fx":
            self.facets.pop("timerange", None)

    def find_files(self) -> None:
        """Find files.

        Look for files and populate the :obj:`Dataset.files` property of
        the dataset and its supplementary datasets.
        """
        self.augment_facets()

        if _isglob(self.facets.get("timerange")):
            self._update_timerange()

        self._find_files()
        for supplementary in self.supplementaries:
            supplementary.find_files()

    def _find_files(self) -> None:
        def version(file: DataElement) -> str:
            return str(file.facets.get("version", ""))

        self._used_data_sources = []
        files: dict[str, DataElement] = {}
        for data_source in sorted(
            _get_data_sources(self.session, self.facets["project"]),  # type: ignore[arg-type]
            key=lambda ds: ds.priority,
        ):
            result = data_source.find_data(**self.facets)
            for file in result:
                if file.name not in files:
                    files[file.name] = file
                if version(files[file.name]) < version(file):
                    files[file.name] = file
            self.files = list(files.values())
            self._used_data_sources.append(data_source)
            # 'quick' mode: if files are available from a higher
            # priority source, do not search lower priority sources.
            if self.session["search_data"] == "complete":
                try:
                    check.data_availability(self, log=False)
                except InputFilesNotFound:
                    pass  # continue search for data
                else:
                    return  # use what has been found so far

    @property
    def files(self) -> list[DataElement]:
        """The files associated with this dataset."""
        if self._files is None:
            self.find_files()
        return self._files  # type: ignore

    @files.setter
    def files(self, value: Sequence[DataElement]) -> None:
        self._files = list(value)

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
        esgf.download(input_files)
        for file in input_files:
            file.prepare()

        cube = self._load()
        supplementary_cubes = []
        for supplementary_dataset in self.supplementaries:
            supplementary_cube = supplementary_dataset._load()  # noqa: SLF001
            supplementary_cubes.append(supplementary_cube)

        output_file = _get_preprocessor_filename(self)
        cubes = preprocess(
            [cube],
            "add_supplementary_variables",
            input_files=input_files,
            output_file=output_file,
            debug=self.session["save_intermediary_cubes"],
            supplementary_cubes=supplementary_cubes,
        )

        return cubes[0]

    def _load(self) -> Cube:
        """Load self.files into an iris cube and return it."""
        if not self.files:
            msg = check.get_no_data_message(self)
            raise InputFilesNotFound(msg)

        output_file = _get_preprocessor_filename(self)
        fix_dir_prefix = Path(
            self.session._fixed_file_dir,  # noqa: SLF001
            self._get_joined_summary_facets("_", join_lists=True) + "_",
        )

        settings: dict[str, dict[str, Any]] = {}
        settings["fix_file"] = {
            "output_dir": fix_dir_prefix,
            "add_unique_suffix": True,
            "session": self.session,
            **self.facets,
        }
        settings["load"] = {}
        settings["fix_metadata"] = {
            "session": self.session,
            **self.facets,
        }
        settings["concatenate"] = {"check_level": self.session["check_level"]}
        settings["cmor_check_metadata"] = {
            "check_level": self.session["check_level"],
            "cmor_table": self.facets["project"],
            "mip": self.facets["mip"],
            "frequency": self.facets["frequency"],
            "short_name": self.facets["short_name"],
        }
        if "timerange" in self.facets:
            settings["clip_timerange"] = {
                "timerange": self.facets["timerange"],
            }
        settings["fix_data"] = {
            "session": self.session,
            **self.facets,
        }
        settings["cmor_check_data"] = {
            "check_level": self.session["check_level"],
            "cmor_table": self.facets["project"],
            "mip": self.facets["mip"],
            "frequency": self.facets["frequency"],
            "short_name": self.facets["short_name"],
        }

        result: Sequence[PreprocessorItem] = self.files
        for step, kwargs in settings.items():
            result = preprocess(
                result,
                step,
                input_files=self.files,
                output_file=output_file,
                debug=self.session["save_intermediary_cubes"],
                **kwargs,
            )

        return result[0]

    def from_ranges(self) -> list[Dataset]:
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
        for key in "ensemble", "sub_experiment":
            if key in self.facets:
                datasets = [
                    ds.copy(**{key: value})
                    for ds in datasets
                    for value in ds._expand_range(key)  # noqa: SLF001
                ]
        return datasets

    def _expand_range(self, input_tag: str) -> list[FacetValue]:
        """Expand ranges such as ensemble members or start dates.

        Expansion only supports ensembles defined as strings, not lists.
        """
        expanded: list[FacetValue] = []
        regex = re.compile(r"\(\d+:\d+\)")

        def expand_range(input_range: str) -> None:
            match = regex.search(input_range)
            if match:
                start, end = match.group(0)[1:-1].split(":")
                for i in range(int(start), int(end) + 1):
                    range_ = regex.sub(str(i), input_range, 1)
                    expand_range(range_)
            else:
                expanded.append(input_range)

        tag = self.facets.get(input_tag, "")
        if isinstance(tag, (list, tuple)):
            for elem in tag:
                if regex.search(elem):
                    msg = (
                        f"In {self}: {input_tag} expansion "
                        f"cannot be combined with {input_tag} lists"
                    )
                    raise RecipeError(msg)
            expanded.append(tag)
        else:
            expand_range(tag)  # type: ignore[arg-type]

        return expanded

    def _update_timerange(self) -> None:
        """Update wildcards in timerange with found datetime values.

        If the timerange is given as a year, it ensures it's formatted
        as a 4-digit value (YYYY).
        """
        dataset = self.copy()
        dataset.supplementaries = []
        dataset.augment_facets()
        if "timerange" not in dataset.facets:
            # timerange facet may be removed in augment_facets for time-independent data.
            self.facets.pop("timerange", None)
            return

        timerange = self.facets["timerange"]
        if not isinstance(timerange, str):
            msg = f"timerange should be a string, got '{timerange!r}'"
            raise TypeError(msg)
        check.valid_time_selection(timerange)

        if "*" in timerange:
            # Replace wildcards in timerange with "timerange" from DataElements,
            # but only if all DataElements have the "timerange" facet.
            dataset = self.copy()
            dataset.facets.pop("timerange")
            dataset.supplementaries = []
            if dataset.files and all(
                "timerange" in f.facets for f in dataset.files
            ):
                # "timerange" can only be reliably computed when all DataElements
                # provide it.
                intervals = [
                    f.facets["timerange"].split("/")  # type: ignore[union-attr]
                    for f in dataset.files
                ]

                min_date = min(interval[0] for interval in intervals)
                max_date = max(interval[1] for interval in intervals)

                if timerange == "*":
                    timerange = f"{min_date}/{max_date}"
                if "*" in timerange.split("/")[0]:
                    timerange = timerange.replace("*", min_date)
                if "*" in timerange.split("/")[1]:
                    timerange = timerange.replace("*", max_date)

        if "*" in timerange:
            # Drop the timerange facet if it still contains wildcards.
            self.facets.pop("timerange")
        else:
            # Make sure that years are in format YYYY
            start_date, end_date = timerange.split("/")
            timerange = _dates_to_timerange(start_date, end_date)
            # Update the timerange
            check.valid_time_selection(timerange)
            self.set_facet("timerange", timerange)
