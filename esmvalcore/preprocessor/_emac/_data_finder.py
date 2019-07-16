"""Extensions to the data finder module for EMAC data."""
from esmvalcore._data_finder import (_find_input_files, _replace_tags,
                                     load_var_mapping)


def get_input_filelist(variable, rootpath, drs):
    """Extension for the data finder function `get_input_filelist` for EMAC."""
    mapping = load_var_mapping(variable['short_name'], variable['project'],
                               variable['var_mapping'])
    files = []
    for channel in mapping.values():
        var = dict(variable)
        var['channel'] = channel
        files.extend(_find_input_files(var, rootpath, drs))
        files = list(set(files))
    return files


def preprocess_filename_for_years(filename, variable):
    """Preprocess filename to extract start and end year of data."""
    data_prefix = '[dataset]'
    pattern = _replace_tags(data_prefix, variable)[0]
    filename = filename.replace(pattern, '')
    filename = filename.lstrip('_')
    return filename
