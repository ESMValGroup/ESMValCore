import glob
import itertools
import logging
# from ._data_finder import find_files
# from ._data_finder import _replace_tags
# from ._data_finder import _resolve_latestversion
import os
from pathlib import Path
import itertools

from . import _data_finder

logger = logging.getLogger(__name__)


class SearchLocation(object):
    def __init__(self, rootpath, input_dir, input_file):
        self.input_dir = input_dir
        self.input_file = input_file
        self.rootpath = [Path(path) for path in rootpath]
        # self.rootpath = Path(rootpath)

    def __repr__(self):
        s = f'rootpath: {self.rootpath!r}'
        s += f'\ninput_dir: {self.input_dir!r}'
        s += f'\ninput_file: {self.input_file!r}'
        return s

    def to_dict(self):
        return self.__dict__

    def _find_input_dirs(self, variable):
        """Return a the full paths to input directories."""
        dirnames = []

        base_paths = self.rootpath
        path_template = self.input_dir

        for dirname_template in _data_finder._replace_tags(
                path_template, variable):
            for base_path in base_paths:
                dirname = _data_finder._resolve_latestversion(
                    dirname_template)  # why is this not part of _replace_tags?
                dirname = os.path.join(base_path, dirname)
                matches = glob.glob(dirname)
                matches = [match for match in matches if os.path.isdir(match)]
                if matches:
                    for match in matches:
                        logger.debug("Found %s", match)
                        dirnames.append(match)
                else:
                    logger.debug("Skipping non-existent %s", dirname)

        return dirnames

    def _get_filenames_glob(self, variable):
        """Return patterns that can be used to look for input files."""
        path_template = self.input_file
        filenames_glob = _data_finder._replace_tags(path_template, variable)
        return filenames_glob

    def _find_input_files(self, variable):
        input_dirs = self._find_input_dirs(variable)
        filenames_glob = self._get_filenames_glob(variable)
        files = _data_finder.find_files(input_dirs, filenames_glob)

        return (files, input_dirs, filenames_glob)


class ProjectData(object):
    def __init__(self, name, output_file, drs_list):
        self.name = name

        # TODO: Make SearchLocation have a singular rootpath
        # locations = []
        # for drs in drs_list:
        #     input_dir = drs['input_dir']
        #     input_file = drs['input_file']
        #     for rootpath in drs['rootpath']:
        #         locations.append(SearchLocation(input_file=input_file, input_dir=input_dir, rootpath=rootpath))

        self._search_locations = [SearchLocation(**item) for item in drs_list]
        self._output_file = output_file

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.name)})'

    def search_locations(self):
        return self._search_locations

    @property
    def output_file(self):
        return self._output_file

    def get_cmor_table(self):
        from .cmor.table import CMOR_TABLES
        return CMOR_TABLES[self.name]

    def get_input_filelist(self, variable):
        filelist = []
        for search_location in self.search_locations:
            new_files = search_location.find_input_files(variable)
            filelist.append(new_files)

        flatten = itertools.chain.from_iterable

        files = list(flatten([lst['files'] for lst in filelist]))
        dirnames = list(flatten([lst['input_dirs'] for lst in filelist]))
        filenames = list(flatten([lst['filenames_glob'] for lst in filelist]))

        return files, dirnames, filenames
