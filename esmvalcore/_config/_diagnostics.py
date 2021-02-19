"""Diagnostics and tags management."""
import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class Diagnostics:
    """Manages the location of the recipes and diagnostics.

    The directory structure is defined below:

    .. code-block

        <path>
        ├── config-references.yml [-> tags_config]
        ├── diag_scripts
        ├── recipes
        └── references
    """

    def __init__(self, path):
        self.path = Path(path)

    def __repr__(self):
        """Return canonical class representation."""
        return f"{self.__class__.__name__}({self.path!s})"

    @property
    def recipes(self):
        """Return the location of the recipes."""
        return self.path / 'recipes'

    @property
    def references(self):
        """Return location of the references (bibtex files)."""
        return self.path / 'references'

    @property
    def tags_config(self):
        """Return location of the tags config."""
        return self.path / 'config-references.yml'

    @property
    def scripts(self):
        """Return location of diagnostic scripts."""
        return self.path / 'diag_scripts'

    def load_tags(self):
        """Load the tags config into an instance of ``TagsManager``."""
        return TagsManager.from_file(self.tags_config)

    @classmethod
    def find(cls):
        """Try to find installed diagnostic scripts."""
        try:
            import esmvaltool
        except ImportError:
            path = Path.cwd()
        else:
            # avoid a crash when there is a directory called
            # 'esmvaltool' that is not a Python package
            if esmvaltool.__file__ is None:
                path = Path.cwd()
            else:
                path = Path(esmvaltool.__file__).absolute().parent
        logger.debug('Using diagnostics from %s', path)
        return cls(path)


class TagsManager(dict):
    """Tag manager."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_file = None

    @classmethod
    def from_file(cls, filename: str):
        """Load the reference tags used for provenance recording."""
        if os.path.exists(filename):
            logger.debug("Loading tags from %s", filename)
            with open(filename) as file:
                tags = cls(yaml.safe_load(file))
                tags.source_file = filename
                return tags
        else:
            # This happens if no diagnostics are installed
            logger.debug("No tags loaded, file %s not present", filename)
            return cls()

    def set_tag_value(self, section: str, tag: str, value):
        """Set the value of a tag in a section.

        Parameters
        ----------
        section : str
            Name of the subsection
        tag : str
            Name of the tag
        value : str
            The value to set
        """
        if section not in self:
            self[section] = {}

        self[section][tag] = value

    def set_tag_values(self, tag_values: dict):
        """Update tags from dict.

        Parameters
        ----------
        tag_values : dict or TagsManager
            Mapping following the structure of Tags.
        """
        for section, tags in tag_values.items():
            for tag, value in tags.items():
                self.set_tag_value(section, tag, value)

    def get_tag_value(self, section: str, tag: str):
        """Retrieve the value of a tag from a section.

        Parameters
        ----------
        section : str
            Name of the subsection
        tag : str
            Name of the tag
        """
        if section not in self:
            postfix = f' in {self.source_file}' if self.source_file else ''
            raise ValueError(f"Section '{section}' does not exist{postfix}")

        if tag not in self[section]:
            postfix = f' of {self.source_file}' if self.source_file else ''
            raise ValueError(
                f"Tag '{tag}' does not exist in section '{section}'{postfix}")

        return self[section][tag]

    def get_tag_values(self, section: str, tags: tuple):
        """Retrieve the values for a list of tags from a section.

        Parameters
        ----------
        section : str
            Name of the subsection
        tags : tuple[str] or list[str]
            List or tuple with tag names
        """
        return tuple(self.get_tag_value(section, tag) for tag in tags)

    def replace_tags_in_dict(self, dct: dict):
        """Resolve tags and updates the given dict in-place.

        Tags are updated one level deep, and only if the corresponding
        section exists in the ``TagsManager``.
        """
        for key in dct:
            if key in self:
                tags = dct[key]
                dct[key] = self.get_tag_values(key, tags)


DIAGNOSTICS = Diagnostics.find()
TAGS = DIAGNOSTICS.load_tags()
