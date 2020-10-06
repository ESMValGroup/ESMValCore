"""ESMValTool utilities."""
import logging
import os

import yaml

from ._diagnostics import DIAGNOSTICS_PATH
from .cmor.table import CMOR_TABLES

logger = logging.getLogger(__name__)

TAGS_CONFIG_FILE = DIAGNOSTICS_PATH / 'config-references.yml'


def get_institutes(variable):
    """Return the institutes given the dataset name in CMIP5 and CMIP6."""
    dataset = variable['dataset']
    project = variable['project']
    logger.debug("Retrieving institutes for dataset %s", dataset)
    from .cmor.institutes import get_institute
    return get_institute(project, dataset)


def get_activity(variable):
    """Return the activity given the experiment name in CMIP6."""
    project = variable['project']
    try:
        exp = variable['exp']
        logger.debug("Retrieving activity_id for experiment %s", exp)
        if isinstance(exp, list):
            return [CMOR_TABLES[project].activities[value][0] for value in exp]
        return CMOR_TABLES[project].activities[exp][0]
    except (KeyError, AttributeError):
        return None


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


TAGS = _load_tags()
