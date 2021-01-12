"""Diagnostics and tags management."""
import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _find_diagnostics():
    """Try to find installed diagnostic scripts."""
    try:
        import esmvaltool
    except ImportError:
        return Path.cwd()
    # avoid a crash when there is a directory called
    # 'esmvaltool' that is not a Python package
    if esmvaltool.__file__ is None:
        return Path.cwd()
    return Path(esmvaltool.__file__).absolute().parent


DIAGNOSTICS_PATH = _find_diagnostics()

TAGS_CONFIG_FILE = os.path.join(DIAGNOSTICS_PATH, 'config-references.yml')


def _load_tags(filename=TAGS_CONFIG_FILE):
    """Load the reference tags used for provenance recording."""
    if os.path.exists(filename):
        logger.debug("Loading tags from %s", filename)
        with open(filename) as file:
            return yaml.safe_load(file)
    else:
        # This happens if no diagnostics are installed
        logger.debug("No tags loaded, file %s not present", filename)
        return {}


TAGS = _load_tags()


def get_tag_value(section, tag):
    """Retrieve the value of a tag."""
    if section not in TAGS:
        raise ValueError("Section '{}' does not exist in {}".format(
            section, TAGS_CONFIG_FILE))
    if tag not in TAGS[section]:
        raise ValueError(
            "Tag '{}' does not exist in section '{}' of {}".format(
                tag, section, TAGS_CONFIG_FILE))
    return TAGS[section][tag]


def replace_tags(section, tags):
    """Replace a list of tags with their values."""
    return tuple(get_tag_value(section, tag) for tag in tags)
