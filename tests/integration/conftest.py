import os

import iris
import pytest

from esmvalcore import _data_finder


def create_test_file(filename, tracking_id=None):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    attributes = {}
    if tracking_id is not None:
        attributes['tracking_id'] = tracking_id
    cube = iris.cube.Cube([], attributes=attributes)

    iris.save(cube, filename)


def _get_filenames(root_path, filenames, tracking_id):
    filename = filenames[0]
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

    def find_files(_, filenames):
        # Any occurrence of [something] in filename should have
        # been replaced before this function is called.
        for filename in filenames:
            assert '{' not in filename
        return _get_filenames(tmp_path, filenames, tracking_id)

    monkeypatch.setattr(_data_finder, 'find_files', find_files)


@pytest.fixture
def patched_failing_datafinder(tmp_path, monkeypatch):

    def tracking_ids(i=0):
        while True:
            yield i
            i += 1

    tracking_id = tracking_ids()

    def find_files(_, filenames):
        # Any occurrence of [something] in filename should have
        # been replaced before this function is called.
        for filename in filenames:
            assert '{' not in filename

        # Fail for specified fx variables
        for filename in filenames:
            if 'fx_' in filename:
                return []
            if 'sftlf' in filename:
                return []
            if 'IyrAnt_' in filename:
                return []
            if 'IyrGre_' in filename:
                return []
        return _get_filenames(tmp_path, filenames, tracking_id)

    monkeypatch.setattr(_data_finder, 'find_files', find_files)
