import glob
import itertools
import logging
import os
from pathlib import Path

from . import _data_finder

logger = logging.getLogger(__name__)


class SearchLocation(object):
    def __init__(self, rootpath: str, input_dir: str, input_file: str):
        self.input_dir = input_dir
        self.input_file = input_file
        self.rootpath = Path(rootpath)

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

        base_path = self.rootpath
        path_template = self.input_dir

        for dirname_template in _data_finder._replace_tags(
                path_template, variable):
            dirname = _data_finder._resolve_latestversion(dirname_template)
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

    def find_input_files(self, variable):
        input_dirs = self._find_input_dirs(variable)
        filenames_glob = self._get_filenames_glob(variable)
        files = _data_finder.find_files(input_dirs, filenames_glob)

        return {
            'files': files,
            'input_dirs': input_dirs,
            'filenames_glob': filenames_glob
        }


class ProjectData(object):
    def __init__(self, name, output_file, drs_list):
        self.name = name

        search_locations = []
        for drs in drs_list:
            input_dir = drs['input_dir']
            input_file = drs['input_file']
            for rootpath in drs['rootpath']:
                search_locations.append(
                    SearchLocation(input_file=input_file,
                                   input_dir=input_dir,
                                   rootpath=rootpath))

        self._search_locations = search_locations
        self._output_file = output_file

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.name)})'

    @property
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
            result = search_location.find_input_files(variable)
            filelist.append(result)

        flatten = itertools.chain.from_iterable

        files = list(flatten([dct['files'] for dct in filelist]))
        dirnames = list(flatten([dct['input_dirs'] for dct in filelist]))
        filenames = list(flatten([dct['filenames_glob'] for dct in filelist]))

        return files, dirnames, filenames
