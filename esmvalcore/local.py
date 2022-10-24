import itertools
import re
from pathlib import Path
from typing import Union

from ._config import Session
from ._data_finder import (
    _select_drs,
    get_input_filelist,
    get_timerange,
    select_by_time,
)
from .types import FacetValue


class LocalFile(type(Path())):  # type: ignore

    @property
    def facets(self) -> dict[str, str]:
        if not hasattr(self, '_facets'):
            self._facets: dict[str, str] = {}
        return self._facets

    @facets.setter
    def facets(self, value: dict[str, str]):
        self._facets = value

    @classmethod
    def _from_path(
        cls,
        path: Union[Path, str],
        drs: str,
        try_timerange: bool,
    ) -> 'LocalFile':
        """Create an instance from a path and populate its facets."""
        file = cls(path)

        # Get facet values from path using a DRS like '{facet1}/{facet2}'.
        keys = []
        for key in re.findall(r"{(.*?)}", drs):
            key = key.split('.')[0]  # Remove trailing .lower and .upper
            keys.append(key)
        start, end = -len(keys) - 1, -1
        values = file.parts[start:end]
        file.facets = {key: values[idx] for idx, key in enumerate(keys)}

        if try_timerange:
            # Try to set the timerange facet.
            timerange = get_timerange(file.name)
            if timerange is not None:
                file.facets['timerange'] = timerange

        return file


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
    filenames, globs = get_input_filelist(
        facets,
        rootpath=session['rootpath'],
        drs=session['drs'],
    )
    drs = _select_drs('input_dir', session['drs'], facets['project'])
    files = []
    for filename in filenames:
        file = LocalFile._from_path(filename, drs, 'timerange' in facets)
        files.append(file)

    if 'timerange' in facets:
        files = select_by_time(files, facets['timerange'])

    if 'version' not in facets:
        files = _select_latest_version(files)

    if debug:
        return files, globs
    return files
