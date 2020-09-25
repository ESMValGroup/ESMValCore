class ProjectData(object):
    def __init__(self, name, output_file, data):
        self.name = name
        self._data = data
        self._output_file = output_file

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.name)})'

    def search_locations(self):
        return self._data

    @property
    def output_file(self):
        return self._output_file

    def get_cmor_table(self):
        from .cmor.table import CMOR_TABLES
        return CMOR_TABLES[self.name]

    def _find_input_dirs(variable):
        pass

    def _get_filenames_glob(variable):
        pass

    def _find_input_files(variable):
        pass

    def get_input_filelist(variable):
        pass
