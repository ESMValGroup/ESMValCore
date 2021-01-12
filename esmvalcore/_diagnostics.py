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
            return Path.cwd()
        # avoid a crash when there is a directory called
        # 'esmvaltool' that is not a Python package
        if esmvaltool.__file__ is None:
            return Path.cwd()
        return cls(Path(esmvaltool.__file__).absolute().parent)


class TagsManager(dict):
    """Tag manager."""
    @classmethod
    def from_file(cls, filename: str):
        """Load the reference tags used for provenance recording."""
        if os.path.exists(filename):
            logger.debug("Loading tags from %s", filename)
            with open(filename) as file:
                return cls(yaml.safe_load(file))
        else:
            # This happens if no diagnostics are installed
            logger.debug("No tags loaded, file %s not present", filename)
            return cls()

    def get_tag_value(self, section, tag):
        """Retrieve the value of a tag."""
        if section not in self:
            raise ValueError(f"Section '{section}' does not exist in {self}")
        if tag not in self[section]:
            raise ValueError(
                f"Tag '{tag}' does not exist in section '{section}' of {self}")
        return self[section][tag]

    def get_tag_values(self, section, tags):
        """Retrieve a list of tags with their values."""
        return tuple(self.get_tag_value(section, tag) for tag in tags)

    def replace_tags_in_dict(self, dct: dict):
        """Resolves tags and updates the given dict in-place.

        Tags are updated one level deep, and only if the corresponding
        section exists in the ``TagsManager``.
        """
        for key in dct:
            if key in self:
                tags = dct[key]
                dct[key] = self.get_tag_values(key, tags)


DIAGNOSTICS = Diagnostics.find()
TAGS = DIAGNOSTICS.load_tags()
