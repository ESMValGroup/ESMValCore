"""Preprocessor download function."""
from ..esgf import ESGFFile


def download(files, dest_folder):
    """Download files that are not available locally.

    Parameters
    ----------
    files: :obj:`list` of :obj:`str` or :obj:`ESGFFile`
        List of local or ESGF files.
    dest_folder: str
        Directory where downloaded files will be stored.
    """
    local_files = []
    for file in files:
        if isinstance(file, ESGFFile):
            local_file = file.download(dest_folder)
            local_files.append(str(local_file))
        else:
            local_files.append(file)

    return local_files
