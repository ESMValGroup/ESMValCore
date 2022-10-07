import re
from pathlib import Path

from ._config import Session
from ._data_finder import _select_drs, get_input_filelist
from .types import FacetValue


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


def find_files(
    session: Session,
    *,
    debug: bool = False,
    **facets: FacetValue,
):
    (filenames, dirnames, fileglobs) = get_input_filelist(
        facets,
        rootpath=session['rootpath'],
        drs=session['drs'],
    )
    drs = _select_drs('input_dir', session['drs'], facets['project'])
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
    def facets(self) -> dict[str, str]:
        if not hasattr(self, '_facets'):
            self._facets: dict[str, str] = {}
        return self._facets

    @facets.setter
    def facets(self, value: dict[str, str]):
        self._facets = value
