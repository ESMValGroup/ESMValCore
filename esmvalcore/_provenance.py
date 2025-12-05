"""Provenance module."""

from __future__ import annotations

import copy
import logging
import os
from functools import total_ordering
from pathlib import Path
from typing import TYPE_CHECKING, Any

from netCDF4 import Dataset
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from prov.model import ProvDerivation, ProvDocument

from esmvalcore._version import __version__
from esmvalcore.io.protocol import DataElement

if TYPE_CHECKING:
    from collections.abc import Iterable

    import prov.model

    from esmvalcore._task import BaseTask

logger = logging.getLogger(__name__)

ESMVALTOOL_URI_PREFIX = "https://www.esmvaltool.org/"


def create_namespace(
    provenance: prov.model.ProvBundle,
    namespace: str,
) -> None:
    """Create an esmvaltool namespace."""
    provenance.add_namespace(namespace, uri=ESMVALTOOL_URI_PREFIX + namespace)


def get_esmvaltool_provenance() -> prov.model.ProvActivity:
    """Create an esmvaltool run activity."""
    provenance = ProvDocument()
    namespace = "software"
    create_namespace(provenance, namespace)
    attributes: dict = {}  # TODO: add dependencies with versions here
    return provenance.activity(
        namespace + ":esmvaltool==" + __version__,
        other_attributes=attributes,
    )


ESMVALTOOL_PROVENANCE = get_esmvaltool_provenance()


def attribute_to_authors(
    entity: prov.model.ProvEntity,
    authors: list[dict[str, str]],
) -> None:
    """Attribute entity to authors."""
    namespace = "author"
    create_namespace(entity.bundle, namespace)

    for author in authors:
        if isinstance(author, str):
            # This happens if the config-references.yml file is not available
            author = {"name": author}  # noqa: PLW2901
        agent = entity.bundle.agent(
            namespace + ":" + author["name"],
            {"attribute:" + k: author[k] for k in author if k != "name"},
        )
        entity.wasAttributedTo(agent)


def attribute_to_projects(
    entity: prov.model.ProvEntity,
    projects: list[str],
) -> None:
    """Attribute entity to projects."""
    namespace = "project"
    create_namespace(entity.bundle, namespace)

    for project in projects:
        agent = entity.bundle.agent(namespace + ":" + project)
        entity.wasAttributedTo(agent)


def get_recipe_provenance(
    documentation: dict[str, Any],
    filename: Path,
) -> prov.model.ProvEntity:
    """Create a provenance entity describing a recipe."""
    provenance = ProvDocument()

    for namespace in ("recipe", "attribute"):
        create_namespace(provenance, namespace)

    entity = provenance.entity(
        f"recipe:{filename}",
        {
            "attribute:description": documentation.get("description", ""),
            "attribute:references": str(documentation.get("references", [])),
        },
    )

    attribute_to_authors(entity, documentation.get("authors", []))
    attribute_to_projects(entity, documentation.get("projects", []))

    return entity


def get_task_provenance(
    task: BaseTask,
    recipe_entity: prov.model.ProvEntity,
) -> prov.model.ProvActivity:
    """Create a provenance activity describing a task."""
    provenance = ProvDocument()
    create_namespace(provenance, "task")

    activity = provenance.activity("task:" + task.name)

    trigger = recipe_entity
    provenance.update(recipe_entity.bundle)

    starter = ESMVALTOOL_PROVENANCE
    provenance.update(starter.bundle)

    activity.wasStartedBy(trigger, starter)

    return activity


@total_ordering
class TrackedFile:
    """File with provenance tracking."""

    def __init__(
        self,
        filename: DataElement | Path,
        attributes: dict[str, Any] | None = None,
        ancestors: Iterable[TrackedFile] | None = None,
        prov_filename: str | None = None,
    ) -> None:
        """Create an instance of a file with provenance tracking.

        Arguments
        ---------
        filename:
            Path or data element containing the data described by the provenance.

        Attributes
        ----------
            Dictionary with facets describing the file. If set to None, this
            will be read from the file when provenance is initialized.
        ancestors:
            Ancestor files.
        prov_filename:
            The path this file has in the provenance record. This can
            differ from `filename` if the file was moved before resuming
            processing.
        """
        self._filename = filename
        if prov_filename is None:
            self.prov_filename = (
                str(filename) if isinstance(filename, Path) else filename.name
            )
        else:
            self.prov_filename = prov_filename

        self.attributes = copy.deepcopy(attributes)

        self.provenance = None
        self.entity = None
        self.activity = None
        self._ancestors = [] if ancestors is None else list(ancestors)

    @property
    def attributes(self) -> dict[str, Any]:
        """Attributes describing the file."""
        if self._attributes is None:
            msg = f"Call {self.__class__.__name__}.initialize_provenance before accessing attributes"
            raise ValueError(msg)
        return self._attributes

    @attributes.setter
    def attributes(self, value: dict[str, Any] | None) -> None:
        """Set attributes describing the file."""
        self._attributes = value

    def __str__(self) -> str:
        """Return summary string."""
        return f"{self.__class__.__name__}: {self.filename}"

    def __repr__(self) -> str:
        """Return representation string (e.g., used by ``pformat``)."""
        return f"{self.__class__.__name__}: {self.filename}"

    def __eq__(self, other: object) -> bool:
        """Check if `other` equals `self`."""
        return hasattr(other, "filename") and self.filename == other.filename

    def __lt__(self, other: object) -> bool:
        """Check if `other` should be sorted before `self`."""
        return hasattr(other, "filename") and self.filename < other.filename

    def __hash__(self) -> int:
        """Return a unique hash for the file."""
        return hash(self.filename)

    def copy_provenance(self) -> TrackedFile:
        """Create a copy with identical provenance information."""
        if self.provenance is None:
            msg = f"Provenance of {self} not initialized"
            raise ValueError(msg)
        new = TrackedFile(Path(self.filename), self.attributes)
        new.provenance = copy.deepcopy(self.provenance)
        new.entity = new.provenance.get_record(self.entity.identifier)[0]
        new.activity = new.provenance.get_record(self.activity.identifier)[0]
        return new

    @property
    def filename(self) -> DataElement | Path:
        """Name of data described by this provenance document."""
        return self._filename

    @property
    def provenance_file(self) -> Path:
        """Filename of provenance file."""
        if not isinstance(self.filename, Path):
            msg = f"Saving provenance is only supported for pathlib.Path, not {type(self.filename)}"
            raise NotImplementedError(msg)
        return self.filename.with_name(f"{self.filename.stem}_provenance.xml")

    def initialize_provenance(self, activity: prov.model.ProvActivity) -> None:
        """Initialize the provenance document.

        Note: this also copies the ancestor provenance. Therefore, changes
        made to ancestor provenance after calling this function will not
        propagate into the provenance of this file.
        """
        if self.provenance is not None:
            msg = f"Provenance of {self} already initialized"
            raise ValueError(msg)
        self.provenance = ProvDocument()
        self._initialize_namespaces()
        self._initialize_activity(activity)
        self._initialize_entity()
        self._initialize_ancestors(activity)

    def _initialize_namespaces(self) -> None:
        """Initialize the namespaces."""
        for namespace in ("file", "attribute", "preprocessor", "task"):
            create_namespace(self.provenance, namespace)

    def _initialize_activity(self, activity: prov.model.ProvActivity) -> None:
        """Copy the preprocessor task activity."""
        self.activity = activity
        self.provenance.update(activity.bundle)  # type: ignore[attr-defined]

    def _initialize_entity(self) -> None:
        """Initialize the entity representing the file."""
        if self._attributes is None:
            if not isinstance(self.filename, DataElement):
                msg = "Delayed reading of attributes is only supported for `DataElement`s"
                raise TypeError(msg)
            # This is used to delay reading the attributes of ancestor files of
            # preprocessor files as created in
            # esmvalcore.preprocessor.Processorfile.__init__ until after the data
            # has been loaded.
            self.attributes = copy.deepcopy(self.filename.attributes)

        attributes = {
            "attribute:" + str(k).replace(" ", "_"): str(v)
            for k, v in self.attributes.items()
            if k not in ("authors", "projects")
        }
        self.entity = self.provenance.entity(  # type: ignore[attr-defined]
            f"file:{self.prov_filename}",
            attributes,
        )

        attribute_to_authors(self.entity, self.attributes.get("authors", []))
        attribute_to_projects(self.entity, self.attributes.get("projects", []))

    def _initialize_ancestors(self, activity: prov.model.ProvActivity) -> None:
        """Register ancestor files for provenance tracking."""
        for ancestor in self._ancestors:
            if ancestor.provenance is None:
                if (
                    isinstance(ancestor.filename, Path)
                    and ancestor.provenance_file.exists()
                ):
                    ancestor.restore_provenance()
                else:
                    ancestor.initialize_provenance(activity)
            self.provenance.update(ancestor.provenance)  # type: ignore[attr-defined]
            self.wasderivedfrom(ancestor)

    def wasderivedfrom(
        self,
        other: TrackedFile | prov.model.ProvEntity,
    ) -> None:
        """Let the file know that it was derived from other."""
        if isinstance(other, TrackedFile):
            other_entity = other.entity
        else:
            other_entity = other
        if not self.activity:
            msg = f"Provenance of {self} not initialized"
            raise ValueError(msg)
        self.provenance.update(other_entity.bundle)  # type: ignore[attr-defined, union-attr]
        self.entity.wasDerivedFrom(other_entity, self.activity)

    def _select_for_include(self) -> dict[str, str]:
        attributes = {
            "software": f"Created with ESMValTool v{__version__}",
        }
        if "caption" in self.attributes:
            attributes["caption"] = self.attributes["caption"]
        return attributes

    @staticmethod
    def _include_provenance_nc(
        filename: Path,
        attributes: dict[str, str],
    ) -> None:
        with Dataset(filename, "a") as dataset:
            for key, value in attributes.items():
                setattr(dataset, key, value)

    @staticmethod
    def _include_provenance_png(
        filename: Path,
        attributes: dict[str, str],
    ) -> None:
        pnginfo = PngInfo()
        exif_tags = {
            "caption": "ImageDescription",
            "software": "Software",
        }
        for key, value in attributes.items():
            pnginfo.add_text(exif_tags.get(key, key), value, zip=True)
        with Image.open(filename) as image:
            image.save(filename, pnginfo=pnginfo)

    def _include_provenance(self) -> None:
        """Include provenance information as metadata."""
        if not isinstance(self.filename, Path):
            msg = f"Writing attributes is only supported for pathlib.Path, not {type(self.filename)}"
            raise NotImplementedError(msg)
        attributes = self._select_for_include()

        # Attach provenance to supported file types
        ext = os.path.splitext(self.filename)[1].lstrip(".").lower()
        write = getattr(self, "_include_provenance_" + ext, None)
        if write:
            write(self.filename, attributes)

    def save_provenance(self) -> None:
        """Export provenance information."""
        self.provenance = ProvDocument(
            records=set(self.provenance.records),  # type: ignore[attr-defined]
            namespaces=self.provenance.namespaces,  # type: ignore[attr-defined]
        )
        self._include_provenance()
        with open(self.provenance_file, "wb") as file:
            # Create file with correct permissions before saving.
            self.provenance.serialize(file, format="xml")  # type: ignore[attr-defined]
        self.activity = None
        self.entity = None
        self.provenance = None

    def restore_provenance(self) -> None:
        """Import provenance information from a previously saved file."""
        self.provenance = ProvDocument.deserialize(
            self.provenance_file,
            format="xml",
        )
        entity_uri = f"{ESMVALTOOL_URI_PREFIX}file{self.prov_filename}"
        self.entity = self.provenance.get_record(entity_uri)[0]  # type: ignore[attr-defined]
        # Find the associated activity
        for rec in self.provenance.records:  # type: ignore[attr-defined]
            if isinstance(rec, ProvDerivation):
                if rec.args[0] == self.entity.identifier:  # type: ignore[attr-defined]
                    activity_id = rec.args[2]
                    self.activity = self.provenance.get_record(activity_id)[0]  # type: ignore[attr-defined]
                    break
