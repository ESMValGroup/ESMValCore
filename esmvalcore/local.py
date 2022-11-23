from __future__ import annotations

import itertools
import re
from pathlib import Path

from ._data_finder import _select_drs, get_input_filelist
from .config import Session
from .types import Facets, FacetValue


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


def _select_latest_version(files: list['LocalFile']) -> list['LocalFile']:
    """Select only the latest version of files."""

    def filename(file):
        return file.name

    def version(file):
        return file.facets.get('version', '')

    result = []
    for _, group in itertools.groupby(sorted(files, key=filename),
                                      key=filename):
        duplicates = sorted(group, key=version)
        latest = duplicates[-1]
        result.append(latest)
    return result


def find_files(
    session: Session,
    *,
    debug: bool = False,
    **facets: FacetValue,
):
    """Find files on the local filesystem.

    Parameters
    ----------
    session
        The session.
    debug
        When debug is set to `True`, the function will return a tuple
        with the first element containing the files that were found
        and the second element containing the globs patterns that
        were used to search for files.
    **facets
        Facets used to search for files.

    Returns
    -------
    list[LocalFile]
        The files that were found.
    """
    filenames, globs = get_input_filelist(
        facets,
        rootpath=session['rootpath'],
        drs=session['drs'],
    )
    drs = _select_drs('input_dir', session['drs'], facets['project'])
    if isinstance(drs, list):
        # Not sure how to handle a list of DRSs
        drs = ''
    files = []
    for filename in filenames:
        file = LocalFile(filename)
        file.facets.update(_path2facets(file, drs))
        files.append(file)

    if 'version' not in facets:
        files = _select_latest_version(files)

    if debug:
        return files, globs
    return files


class LocalFile(type(Path())):  # type: ignore
    """File on the local filesystem."""
    @property
    def facets(self) -> Facets:
        """Facets describing the file."""
        if not hasattr(self, '_facets'):
            self._facets: Facets = {}
        return self._facets

    @facets.setter
    def facets(self, value: Facets):
        self._facets = value
