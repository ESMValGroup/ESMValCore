"""Tests for `esmvalcore.local`."""
import os
import pprint
from pathlib import Path

import pytest
import yaml

from esmvalcore.config import CFG
from esmvalcore.local import LocalFile, _get_output_file, find_files

# Load test configuration
with open(os.path.join(os.path.dirname(__file__),
                       'data_finder.yml'),
          encoding='utf-8') as file:
    CONFIG = yaml.safe_load(file)


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

    with open(filename, 'a', encoding='utf-8'):
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
    output_file = _get_output_file(cfg['variable'], cfg['preproc_dir'])
    expected = Path(cfg['output_file'])
    assert output_file == expected


@pytest.fixture
def root(tmp_path):
    """Root function for tests."""
    dirname = str(tmp_path)
    yield dirname
    print("Directory structure was:")
    tree(dirname)


@pytest.mark.parametrize('cfg', CONFIG['get_input_filelist'])
def test_find_files(monkeypatch, root, cfg):
    """Test retrieving input filelist."""
    print(f"Testing DRS {cfg['drs']} with variable:\n",
          pprint.pformat(cfg['variable']))
    project = cfg['variable']['project']
    monkeypatch.setitem(CFG, 'drs', {project: cfg['drs']})
    monkeypatch.setitem(CFG, 'rootpath', {project: root})
    create_tree(root, cfg.get('available_files'),
                cfg.get('available_symlinks'))

    # Find files
    input_filelist, globs = find_files(debug=True, **cfg['variable'])
    # Test result
    ref_files = [Path(root, file) for file in cfg['found_files']]
    ref_globs = [
        Path(root, d, f) for d in cfg['dirs'] for f in cfg['file_patterns']
    ]
    assert sorted([Path(f) for f in input_filelist]) == sorted(ref_files)
    assert sorted([Path(g) for g in globs]) == sorted(ref_globs)


def test_find_files_with_facets(monkeypatch, root):
    """Test that a LocalFile with populated `facets` is returned."""
    for cfg in CONFIG['get_input_filelist']:
        if cfg['drs'] != 'default':
            break

    project = cfg['variable']['project']
    monkeypatch.setitem(CFG, 'drs', {project: cfg['drs']})
    monkeypatch.setitem(CFG, 'rootpath', {project: root})

    create_tree(root, cfg.get('available_files'),
                cfg.get('available_symlinks'))

    # Find files
    input_filelist = find_files(**cfg['variable'])
    ref_files = [Path(root, file) for file in cfg['found_files']]
    assert sorted([Path(f) for f in input_filelist]) == sorted(ref_files)
    assert isinstance(input_filelist[0], LocalFile)
    assert input_filelist[0].facets
