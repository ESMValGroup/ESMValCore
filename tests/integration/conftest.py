import os
from pathlib import Path

import iris
import pytest

import esmvalcore.local
from esmvalcore.config import CFG
from esmvalcore.config._config_object import CFG_DEFAULT
from esmvalcore.local import (
    LocalFile,
    _replace_tags,
    _select_drs,
    _select_files,
)


@pytest.fixture
def session(tmp_path: Path, monkeypatch):
    CFG.clear()
    CFG.update(CFG_DEFAULT)
    monkeypatch.setitem(CFG, 'rootpath', {'default': str(tmp_path)})

    session = CFG.start_session('recipe_test')
    session['output_dir'] = tmp_path / 'esmvaltool_output'
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


def _get_files(root_path, facets, tracking_id):
    file_template = _select_drs('input_file', facets['project'])
    file_globs = _replace_tags(file_template, facets)
    filename = Path(file_globs[0]).name
    filename = str(root_path / 'input' / filename)
    filenames = []
    if filename.endswith('[_.]*nc'):
        # Restore when we support filenames with no dates
        # filenames.append(filename.replace('[_.]*nc', '.nc'))
        filename = filename.replace('[_.]*nc', '_*.nc')
    if filename.endswith('*.nc'):
        filename = filename[:-len('*.nc')] + '_'
        if facets['frequency'] == 'fx':
            intervals = ['']
        else:
            intervals = [
                '1990_1999',
                '2000_2009',
                '2010_2019',
            ]
        for interval in intervals:
            filenames.append(filename + interval + '.nc')
    else:
        filenames.append(filename)

    if 'timerange' in facets:
        filenames = _select_files(filenames, facets['timerange'])

    for filename in filenames:
        create_test_file(filename, next(tracking_id))

    files = []
    for filename in filenames:
        file = LocalFile(filename)
        file.facets = facets
        files.append(file)

    return files, file_globs


@pytest.fixture
def patched_datafinder(tmp_path, monkeypatch):

    def tracking_ids(i=0):
        while True:
            yield i
            i += 1

    tracking_id = tracking_ids()

    def find_files(*, debug: bool = False, **facets):
        files, file_globs = _get_files(tmp_path, facets, tracking_id)
        if debug:
            return files, file_globs
        return files

    monkeypatch.setattr(esmvalcore.local, 'find_files', find_files)


@pytest.fixture
def patched_failing_datafinder(tmp_path, monkeypatch):

    def tracking_ids(i=0):
        while True:
            yield i
            i += 1

    tracking_id = tracking_ids()

    def find_files(*, debug: bool = False, **facets):
        files, file_globs = _get_files(tmp_path, facets, tracking_id)
        if 'fx' == facets['frequency']:
            files = []
        if debug:
            return files, file_globs
        return files

    monkeypatch.setattr(esmvalcore.local, 'find_files', find_files)
