"""Module for navigating a file's ancestry in the ESMValTool."""
from abc import ABCMeta, abstractmethod


class NoLocalParentError(ValueError):
    """Exception raised when the parent for a dataset cannot be found
    locally."""


class NoESGFParentError(ValueError):
    """Exception raised when the parent for a dataset cannot be found on
    ESGF."""


class LocalDataset(metaclass=ABCMeta):

    @property
    @abstractmethod
    def _project(self):
        """str: Name of the project to which this dataset belongs"""

    @property
    @abstractmethod
    def _parent_id_keys(self):
        """List[str] Attributes which can be used to identify the parent of a
        dataset."""

    def find_local_parent(self):
        """Find parent files locally.

        Returns
        -------
        list[:obj:`pathlib.Path`]
            Path to files in the parent dataset

        Raises
        ------
        NoLocalParentError
            No parent files could be found locally
        """

    def find_esgf_parent(self):
        """Find parent files on ESGF.

        Returns
        -------
        list[:obj:`ESGFFile`]
            ESGF files in the parent dataset

        Raises
        ------
        NoESGFParentError
            No parent files could be found on ESGF

        Notes
        -----
        Sub-classes of :obj:`LocalDataset` which relate to datasets that cannot be
        downloaded from ESGF should simply implement this method with
        ``raise NotImplementedError``.
        """


class LocalDatasetCMIP5(LocalDataset):
    _project = "CMIP5"
    _parent_id_keys = (
        "parent_experiment",
        "parent_experiment_rip",
    )


class LocalDatasetCMIP6(LocalDataset):
    _project = "CMIP6"
    _parent_id_keys = (
        "parent_activity_id",
        "parent_experiment_id",
        "parent_mip_era",
        "parent_source_id",
        "parent_variant_label",
    )


def _get_local_dataset(project):
    """Get an instance of :class:`LocalDataset` for a given project.

    Parameters
    ----------
    project : str
        Project

    Returns
    -------
    :obj:`LocalDataset`

    Raises
    ------
    NotImplementedError
        There is no implementation of :obj:`LocalDataset` for the requested
        project
    """
    if project == "CMIP5":
        return LocalDatasetCMIP5()

    if project == "CMIP6":
        return LocalDatasetCMIP6()

    raise NotImplementedError(f"No LocalDataset for {project}")
