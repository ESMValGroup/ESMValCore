"""ESMValTool configuration."""
import logging
import logging.config
import os
import time
from pathlib import Path

import yaml

from esmvalcore import drs_config

from .cmor.table import CMOR_TABLES

logger = logging.getLogger(__name__)


def find_diagnostics():
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


DIAGNOSTICS_PATH = find_diagnostics()


def configure_logging(cfg_file=None, output_dir=None, console_log_level=None):
    """Set up logging."""
    if cfg_file is None:
        cfg_file = os.path.join(os.path.dirname(__file__),
                                'config-logging.yml')

    cfg_file = os.path.abspath(cfg_file)
    with open(cfg_file) as file_handler:
        cfg = yaml.safe_load(file_handler)

    if output_dir is None:
        cfg['handlers'] = {
            name: handler
            for name, handler in cfg['handlers'].items()
            if 'filename' not in handler
        }
        prev_root = cfg['root']['handlers']
        cfg['root']['handlers'] = [
            name for name in prev_root if name in cfg['handlers']
        ]

    log_files = []
    for handler in cfg['handlers'].values():
        if 'filename' in handler:
            if not os.path.isabs(handler['filename']):
                handler['filename'] = os.path.join(output_dir,
                                                   handler['filename'])
            log_files.append(handler['filename'])
        if console_log_level is not None and 'stream' in handler:
            if handler['stream'] in ('ext://sys.stdout', 'ext://sys.stderr'):
                handler['level'] = console_log_level.upper()

    logging.config.dictConfig(cfg)
    logging.Formatter.converter = time.gmtime
    logging.captureWarnings(True)

    return log_files


def get_project_config(project):
    """Get developer-configuration for project."""
    logger.debug("Retrieving %s configuration", project)
    if project in drs_config:
        return drs_config[project]
    raise ValueError(f"Project '{project}' not in config-developer.yml")


def get_institutes(variable):
    """Return the institutes given the dataset name in CMIP5 and CMIP6."""
    dataset = variable['dataset']
    project = variable['project']
    logger.debug("Retrieving institutes for dataset %s", dataset)
    try:
        return CMOR_TABLES[project].institutes[dataset]
    except (KeyError, AttributeError):
        pass
    return drs_config.get(project, {}).get('institutes', {}).get(dataset, [])


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
