import re
from pathlib import Path
from typing import Any

from ._data_finder import _select_drs, get_input_filelist


def _path2facets(path: Path, drs: str) -> dict[str, str]:
    """Extract facets from a path using a DRS like '{facet1}/{facet2}'."""
    keys = []
    for key in re.findall(r"{(.*?)}", drs):
        key = key.split('.')[0]  # Remove trailing .lower and .upper
        keys.append(key)
    start, end = -len(keys) - 1, -1
    values = path.parts[start:end]
    facets = {key: values[idx] for idx, key in enumerate(keys)}
    return facets


def find_files(session, *, project: str, debug: bool = False, **facets):

    facets['project'] = project
    result = get_input_filelist(
        facets,
        rootpath=session['rootpath'],
        drs=session['drs'],
    )
    (filenames, dirnames, fileglobs) = result
    drs = _select_drs('input_dir', session['drs'], project)
    files = []
    for filename in filenames:
        file = LocalFile(filename)
        file.facets = _path2facets(file, drs)
        files.append(file)

    if debug:
        return files, (dirnames, fileglobs)
    return files


class LocalFile(type(Path())):  # type: ignore

    @property
    def facets(self) -> dict:
        if not hasattr(self, '_facets'):
            self._facets: dict[str, Any] = {}
        return self._facets

    @facets.setter
    def facets(self, value: dict):
        self._facets = value
