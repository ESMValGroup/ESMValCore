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
    cube = iris.cube.Cube([])
    cube.attributes.globals = attributes

    iris.save(cube, filename)


def _get_files(root_path, facets, tracking_id):
    """Return dummy files.

    Wildcards are only supported for `dataset` and `institute`; in this case
    return files for the two "models" AAA and BBB.

    """
    if facets['dataset'] == '*':
        all_facets = [
            {**facets, 'dataset': 'AAA', 'institute': 'A'},
            {**facets, 'dataset': 'BBB', 'institute': 'B'},
        ]
    else:
        all_facets = [facets]

    # Globs without expanded facets
    dir_template = _select_drs('input_dir', facets['project'])
    file_template = _select_drs('input_file', facets['project'])
    dir_globs = _replace_tags(dir_template, facets)
    file_globs = _replace_tags(file_template, facets)
    globs = sorted(
        root_path / 'input' / d / f for d in dir_globs for f in file_globs
    )

    files = []
    for expanded_facets in all_facets:
        filenames = []
        dir_template = _select_drs('input_dir', expanded_facets['project'])
        file_template = _select_drs('input_file', expanded_facets['project'])
        dir_globs = _replace_tags(dir_template, expanded_facets)
        file_globs = _replace_tags(file_template, expanded_facets)
        filename = (
            str(root_path / 'input' / dir_globs[0] / Path(file_globs[0]).name)
        )

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

        for filename in filenames:
            file = LocalFile(filename)
            file.facets = expanded_facets
            files.append(file)

    return files, globs


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
    """Failing data finder.

    Do not return files for:
    - fx files
    - Variable rsutcs for model AAA

    Otherwise, return files just like `patched_datafinder`.

    """

    def tracking_ids(i=0):
        while True:
            yield i
            i += 1

    tracking_id = tracking_ids()

    def find_files(*, debug: bool = False, **facets):
        files, file_globs = _get_files(tmp_path, facets, tracking_id)
        if 'fx' == facets['frequency']:
            files = []
        returned_files = []
        for file in files:
            if not ('AAA' in file.name and 'rsutcs' in file.name):
                returned_files.append(file)
        if debug:
            return returned_files, file_globs
        return returned_files

    monkeypatch.setattr(esmvalcore.local, 'find_files', find_files)
