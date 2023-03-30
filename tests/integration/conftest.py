import os
from pathlib import Path

import iris
import pytest

import esmvalcore.local
from esmvalcore.config import CFG, _config
from esmvalcore.config._config_object import CFG_DEFAULT


@pytest.fixture
def session(tmp_path, monkeypatch):
    session = CFG.start_session('recipe_test')
    session.clear()
    session.update(CFG_DEFAULT)
    session['output_dir'] = tmp_path / 'esmvaltool_output'

    # The patched_datafinder fixture does not return the correct input
    # directory structure, so make sure it is set to flat for every project
    monkeypatch.setitem(CFG, 'drs', {})
    for project in _config.CFG:
        monkeypatch.setitem(_config.CFG[project]['input_dir'], 'default', '/')
    # The patched datafinder fixture does not return any facets, so automatic
    # supplementary definition does not work with it.
    session['use_legacy_supplementaries'] = True
    return session


def create_test_file(filename, tracking_id=None):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    attributes = {}
    if tracking_id is not None:
        attributes['tracking_id'] = tracking_id
    cube = iris.cube.Cube([], attributes=attributes)

    iris.save(cube, filename)


def _get_filenames(root_path, filename, tracking_id):
    filename = Path(filename).name
    filename = str(root_path / 'input' / filename)
    filenames = []
    if filename.endswith('[_.]*nc'):
        # Restore when we support filenames with no dates
        # filenames.append(filename.replace('[_.]*nc', '.nc'))
        filename = filename.replace('[_.]*nc', '_*.nc')
    if filename.endswith('*.nc'):
        filename = filename[:-len('*.nc')] + '_'
        intervals = [
            '1990_1999',
            '2000_2009',
            '2010_2019',
        ]
        for interval in intervals:
            filenames.append(filename + interval + '.nc')
    else:
        filenames.append(filename)

    for filename in filenames:
        create_test_file(filename, next(tracking_id))
    return filenames


@pytest.fixture
def patched_datafinder(tmp_path, monkeypatch):
    def tracking_ids(i=0):
        while True:
            yield i
            i += 1

    tracking_id = tracking_ids()

    def glob(file_glob):
        return _get_filenames(tmp_path, file_glob, tracking_id)

    monkeypatch.setattr(esmvalcore.local, 'glob', glob)


@pytest.fixture
def patched_failing_datafinder(tmp_path, monkeypatch):
    def tracking_ids(i=0):
        while True:
            yield i
            i += 1

    tracking_id = tracking_ids()

    def glob(filename):
        # Fail for specified fx variables
        if 'fx_' in filename:
            return []
        if 'sftlf' in filename:
            return []
        if 'IyrAnt_' in filename:
            return []
        if 'IyrGre_' in filename:
            return []
        return _get_filenames(tmp_path, filename, tracking_id)

    monkeypatch.setattr(esmvalcore.local, 'glob', glob)
