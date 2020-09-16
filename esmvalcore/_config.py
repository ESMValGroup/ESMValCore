"""ESMValTool configuration."""
import datetime
import logging
import logging.config
import os
import pprint
import time
from pathlib import Path

import yaml

from .cmor.table import CMOR_TABLES, read_cmor_tables

logger = logging.getLogger(__name__)

CFG = {}

ESMVALCORE_DIR = Path(__file__).parent
CONFIG_NAME = 'config-user.yml'
DEFAULT_SETTINGS = yaml.safe_load(open(str(ESMVALCORE_DIR / CONFIG_NAME)))
USER_CONFIG_FILE = Path('~/.esmvaltool/') / CONFIG_NAME


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


def read_config_user_file(config_file, folder_name=None, options=None):
    """Read config user file and store settings in a dictionary."""
    config_file = os.path.abspath(
        os.path.expandvars(os.path.expanduser(config_file)))
    # Read user config file
    if not os.path.exists(config_file):
        print(f"ERROR: Config file {config_file} does not exist")

    with open(config_file, 'r') as file:
        cfg = yaml.safe_load(file)

    if options is None:
        options = dict()
    for key, value in options.items():
        cfg[key] = value

    for key in DEFAULT_SETTINGS:
        if key not in cfg:
            logger.info(
                "No %s specification in config file, "
                "defaulting to %s", key, DEFAULT_SETTINGS[key])
            cfg[key] = DEFAULT_SETTINGS[key]

    cfg['output_dir'] = _normalize_path(cfg['output_dir'])
    cfg['auxiliary_data_dir'] = _normalize_path(cfg['auxiliary_data_dir'])

    cfg['config_developer_file'] = _normalize_path(
        cfg['config_developer_file'])

    for key in cfg['rootpath']:
        root = cfg['rootpath'][key]
        if isinstance(root, str):
            cfg['rootpath'][key] = [_normalize_path(root)]
        else:
            cfg['rootpath'][key] = [_normalize_path(path) for path in root]

    if folder_name:
        # insert a directory date_time_recipe_usertag in the output paths
        now = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        new_subdir = '_'.join((folder_name, now))
        cfg['output_dir'] = os.path.join(cfg['output_dir'], new_subdir)

        # create subdirectories
        cfg['preproc_dir'] = os.path.join(cfg['output_dir'], 'preproc')
        cfg['work_dir'] = os.path.join(cfg['output_dir'], 'work')
        cfg['plot_dir'] = os.path.join(cfg['output_dir'], 'plots')
        cfg['run_dir'] = os.path.join(cfg['output_dir'], 'run')

    # Read developer configuration file
    cfg_developer = read_config_developer_file(cfg['config_developer_file'])
    for key, value in cfg_developer.items():
        CFG[key] = value
    read_cmor_tables(CFG)

    return cfg


def _normalize_path(path):
    """Normalize paths.

    Expand ~ character and environment variables and convert path to absolute.

    Parameters
    ----------
    path: str
        Original path

    Returns
    -------
    str:
        Normalized path
    """
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def read_config_developer_file(cfg_file=None):
    """Read the developer's configuration file."""
    if cfg_file is None:
        cfg_file = os.path.join(
            os.path.dirname(__file__),
            'config-developer.yml',
        )

    with open(cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    return cfg


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
    if project in CFG:
        return CFG[project]
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
    return CFG.get(project, {}).get('institutes', {}).get(dataset, [])


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


class Config:
    """Importable config object."""
    def __init__(self):
        super().__init__()
        self.config_file = None
        self.default_mapping = DEFAULT_SETTINGS
        self.load()

    def load(self, config_file: str = USER_CONFIG_FILE):
        """Load config from config user file."""
        self.mapping = read_config_user_file(config_file)
        self.init_session_dir()
        self.config_file = config_file

    def load_from_dict(self, mapping):
        """Load config from dictionary."""
        self.mapping = mapping

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.config_file}')"

    def __str__(self):
        return pprint.pformat(self.mapping)

    def __getitem__(self, item):
        try:
            return self.mapping[item]
        except KeyError:
            return self.default_mapping[item]

    def init_session_dir(self, name: str = 'session'):
        """Initialize session.

        The `name` is used to name the working directory, e.g.
        recipe_example_20200916/ If no name is given, such as in an
        interactive session, defaults to `session`.
        """
        now = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        session_name = f"{name}_{now}"
        self._session_dir = Path(self.mapping['output_dir']) / session_name

    @property
    def session_dir(self):
        return self._session_dir

    @property
    def preproc_dir(self):
        return self.session_dir / 'preproc'

    @property
    def work_dir(self):
        return self.session_dir / 'work'

    @property
    def plot_dir(self):
        return self.session_dir / 'plots'

    @property
    def run_dir(self):
        return self.session_dir / 'run'


# initialize config object here, so it can be re-used across modules
config = Config()
