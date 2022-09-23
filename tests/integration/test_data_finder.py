"""Tests for _data_finder.py."""
import os
import shutil
import tempfile

import pytest
import yaml

import esmvalcore._config
from esmvalcore._data_finder import (
    _find_input_files,
    get_input_filelist,
    get_output_file,
)
from esmvalcore.cmor.table import read_cmor_tables

# Initialize with standard config developer file
CFG_DEVELOPER = esmvalcore._config.read_config_developer_file()
esmvalcore._config._config.CFG = CFG_DEVELOPER
# Initialize CMOR tables
read_cmor_tables(CFG_DEVELOPER)

# Load test configuration
with open(os.path.join(os.path.dirname(__file__), 'data_finder.yml')) as file:
    CONFIG = yaml.safe_load(file)


def _augment_with_extra_facets(variable):
    """Augment variable dict with extra facets."""
    extra_facets = esmvalcore._config.get_extra_facets(
        variable['project'],
        variable['dataset'],
        variable['mip'],
        variable['short_name'],
        (),
    )
    for (key, val) in extra_facets.items():
        if key not in variable:
            variable[key] = val


def print_path(path):
    """Print path."""
    txt = path
    if os.path.isdir(path):
        txt += '/'
    if os.path.islink(path):
        txt += ' -> ' + os.readlink(path)
    print(txt)


def tree(path):
    """Print path, similar to the the `tree` command."""
    print_path(path)
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            print_path(os.path.join(dirpath, dirname))
        for filename in filenames:
            print_path(os.path.join(dirpath, filename))


def create_file(filename):
    """Create an empty file."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filename, 'a'):
        pass


def create_tree(path, filenames=None, symlinks=None):
    """Create directory structure and files."""
    for filename in filenames or []:
        create_file(os.path.join(path, filename))

    for symlink in symlinks or []:
        link_name = os.path.join(path, symlink['link_name'])
        os.symlink(symlink['target'], link_name)


@pytest.mark.parametrize('cfg', CONFIG['get_output_file'])
def test_get_output_file(cfg):
    """Test getting output name for preprocessed files."""
    _augment_with_extra_facets(cfg['variable'])
    output_file = get_output_file(cfg['variable'], cfg['preproc_dir'])
    assert output_file == cfg['output_file']


@pytest.fixture
def root():
    """Root function for tests."""
    dirname = tempfile.mkdtemp()
    yield os.path.join(dirname, 'output1')
    print("Directory structure was:")
    tree(dirname)
    shutil.rmtree(dirname)


@pytest.mark.parametrize('cfg', CONFIG['get_input_filelist'])
def test_get_input_filelist(root, cfg):
    """Test retrieving input filelist."""
    create_tree(root, cfg.get('available_files'),
                cfg.get('available_symlinks'))

    # Augment variable dict with extra facets
    _augment_with_extra_facets(cfg['variable'])

    # Find files
    rootpath = {cfg['variable']['project']: [root]}
    drs = {cfg['variable']['project']: cfg['drs']}
    timerange = cfg['variable'].get('timerange')
    if timerange and '*' in timerange:
        (files, _, _) = _find_input_files(cfg['variable'], rootpath, drs)
        ref_files = [
            os.path.join(root, file) for file in cfg['found_files']]
        # Test result
        assert sorted(files) == sorted(ref_files)
    else:
        (input_filelist, dirnames,
         filenames) = get_input_filelist(cfg['variable'], rootpath, drs)
        # Test result
        ref_files = [os.path.join(root, file) for file in cfg['found_files']]
        if cfg['dirs'] is None:
            ref_dirs = []
        else:
            ref_dirs = [os.path.join(root, dir) for dir in cfg['dirs']]
        ref_patterns = cfg['file_patterns']

        assert sorted(input_filelist) == sorted(ref_files)
        assert sorted(dirnames) == sorted(ref_dirs)
        assert sorted(filenames) == sorted(ref_patterns)
