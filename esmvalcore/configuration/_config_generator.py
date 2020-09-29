import logging
import shutil
import sys
from pathlib import Path

import yaml

from .._session import session

logger = logging.getLogger(__name__)


# Work-around for yaml formatting bug: https://stackoverflow.com/a/39681672
class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def update_projects_dict(base_projects, site_projects):
    """Update the missing keys from the site_projects with the values from the
    base_projects."""
    projects = {}
    for key, site_project in site_projects.items():
        base_project = base_projects[key]
        d = {}
        d['output_file'] = site_project.get('output_file',
                                            base_project['output_file'])
        d['data'] = []
        for site_drs in site_project['data']:
            drs = base_project['data'][0].copy()
            drs.update(site_drs)
            d['data'].append(drs)

        projects[key] = d

    return projects


def load_yaml(filename):
    """Load a yaml file."""
    with open(filename) as f:
        mapping = yaml.safe_load(f)
    return mapping


def get_user_path(filename, destination=None, overwrite=False):
    """Get the user path, and exit if it already exists (depends on
    `overwrite`).

    Defaults to `~/.esmvalcore/config-user.yml`. If that exists, use the
    specified filename instead.
    """
    if not destination:
        destination = session.config_dir / 'config-user.yml'
        if destination.exists():
            destination = session.config_dir / filename
    else:
        destination = Path(destination)

    if destination.exists():
        if overwrite:
            logger.info('Overwriting file %s.', destination)
        else:
            logger.error('Get config aborted. File %s already exists.',
                         destination)
            sys.exit()

    return destination


def generate_config(name, overwrite, path):
    """Combine the default config with the default/site-specific DRS
    specifications."""
    base_config_dir = Path(__file__).parent

    default_config = base_config_dir / 'config-default.yml'

    base_projects = load_yaml(base_config_dir / 'drs-default.yml')

    if name != 'default':
        site_projects = load_yaml(base_config_dir / f'drs-{name}.yml')
        projects = update_projects_dict(base_projects, site_projects)
    else:
        projects = base_projects

    filename = f'config-{name}.yml'

    path = get_user_path(filename, path, overwrite=overwrite)

    shutil.copy(default_config, path)
    with open(path, 'a') as f:
        print('\n', file=f)
        print(f'# Data reference syntax ({name})', file=f)
        print('# Modify/add your own data specifications', file=f)
        yaml.dump(projects,
                  stream=f,
                  Dumper=MyDumper,
                  default_flow_style=False)

    logger.info('Writing config file to %s.', path)


def list_available_configs():
    """List available default/site-specific configs."""
    pre = '- '

    user_config_folder = session.config_dir
    base_config_dir = Path(__file__).parent

    base_configs = sorted(base_config_dir.glob('drs-*.yml'))
    user_configs = sorted(user_config_folder.glob('*.yml'))

    print('# Available site-specific configs')
    print('# Use `esmvaltool config get CONFIG_NAME` to copy them')
    for path in base_configs:
        name = path.stem.split('-', 1)[1]
        print(f'{pre}{name}')
    print()

    if not user_configs:
        return

    print(f'# Available configs in {user_config_folder}')
    print('# Select config with '
          '`esmvaltool run <recipe> --config_file=CONFIG_FILE`')
    for path in user_configs:
        if path.stem == 'config-user':
            print(f'{pre}{path.name} [default]')
        else:
            print(f'{pre}{path.name}')
