"""Apply automatic fixes for known errors in cmorized data.

All functions in this module will work even if no fixes are available
for the given dataset. Therefore is recommended to apply them to all
variables to be sure that all known errors are fixed.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor._utils import _get_single_cube
from esmvalcore.io._woodpecker import is_fixed_by_woodpecker
from esmvalcore.io.local import LocalFile

if TYPE_CHECKING:
    from esmvalcore.config import Session

logger = logging.getLogger(__name__)


def fix_file(  # noqa: PLR0913, PLR0917
    file: Path,
    short_name: str,
    project: str,
    dataset: str,
    mip: str,
    output_dir: Path,
    add_unique_suffix: bool = False,
    session: Session | None = None,
    frequency: str | None = None,
    **extra_facets: Any,
) -> Path | Sequence[Cube]:
    """Fix files before loading them into a :class:`~iris.cube.CubeList`.

    This is mainly intended to fix errors that prevent loading the data with
    Iris (e.g., those related to ``missing_value`` or ``_FillValue``) or
    operations that are more efficient with other packages (e.g., loading files
    with lots of variables is much faster with Xarray than Iris).

    If a `woodpecker <https://github.com/roocs/woodpecker>`__ recipe matches
    the file, it is returned unchanged, because woodpecker fixes the file
    when it is loaded.

    Parameters
    ----------
    file:
        Path to the original file. Original files are not overwritten.
    short_name:
        Variable's short name.
    project:
        Project of the dataset.
    dataset:
        Name of the dataset.
    mip:
        Variable's MIP.
    output_dir:
        Output directory for fixed files.
    add_unique_suffix:
        Adds a unique suffix to ``output_dir`` for thread safety.
    session:
        Current session which includes configuration and directory information.
    frequency:
        Variable's data frequency, if available.
    **extra_facets:
        Extra facets. For details, see :ref:`config-extra-facets`.

    Returns
    -------
    :
        Fixed data or a path to them.

    """
    if not isinstance(file, Path):
        # Skip this function for `esmvalcore.io.DataElement` that is not a path
        # to a file.
        return file

    local_file = file if isinstance(file, LocalFile) else LocalFile(file)
    if local_file._woodpecker_recipe() is not None:  # noqa: SLF001
        logger.debug(
            "Not applying ESMValCore fixes to %s, because a woodpecker recipe "
            "matches it",
            file,
        )
        return file

    # Update extra_facets with variable information given as regular arguments
    # to this function
    extra_facets.update(
        {
            "short_name": short_name,
            "project": project,
            "dataset": dataset,
            "mip": mip,
            "frequency": frequency,
        },
    )

    result: Path | Sequence[Cube] = Path(file)
    for fix in Fix.get_fixes(
        project=project,
        dataset=dataset,
        mip=mip,
        short_name=short_name,
        extra_facets=extra_facets,
        session=session,
        frequency=frequency,
    ):
        result = fix.fix_file(
            result,
            output_dir,
            add_unique_suffix=add_unique_suffix,
        )

    if isinstance(file, LocalFile):
        # This happens when this function is called from
        # `esmvalcore.dataset.Dataset.load`.
        if isinstance(result, Path):
            if result == file:
                # No fixes have been applied, return the original file.
                result = file
            else:
                # The file has been fixed and the result is a path to the fixed
                # file. The result needs to be loaded to read the global
                # attributes for recording provenance.
                fixed_file = LocalFile(result)
                fixed_file.facets = file.facets
                fixed_file.ignore_warnings = file.ignore_warnings
                result = fixed_file.to_iris()

        if isinstance(result, Sequence) and isinstance(result[0], Cube):
            # Set the attributes for recording provenance here because
            # to_iris will not be called on the original file.
            file.attributes = result[0].attributes.globals.copy()

    return result


def fix_metadata(
    cubes: Sequence[Cube],
    short_name: str,
    project: str,
    dataset: str,
    mip: str,
    frequency: str | None = None,
    session: Session | None = None,
    **extra_facets: Any,
) -> CubeList:
    """Fix cube metadata if fixes are required.

    This method collects all the relevant fixes (including generic ones) for a
    given variable and applies them.

    Cubes from files that have already been fixed by
    `woodpecker <https://github.com/roocs/woodpecker>`__, i.e. that have the
    global attribute ``woodpecker_provenance``, are not fixed.

    Parameters
    ----------
    cubes:
        Cubes to fix.
    short_name:
        Variable's short name.
    project:
        Project of the dataset.
    dataset:
        Name of the dataset.
    mip:
        Variable's MIP.
    frequency:
        Variable's data frequency, if available.
    session:
        Current session which includes configuration and directory information.
    **extra_facets:
        Extra facets. For details, see :ref:`config-extra-facets`.

    Returns
    -------
    iris.cube.CubeList
        Fixed cubes.

    """
    # Update extra_facets with variable information given as regular arguments
    # to this function
    extra_facets.update(
        {
            "short_name": short_name,
            "project": project,
            "dataset": dataset,
            "mip": mip,
            "frequency": frequency,
        },
    )

    fixes = Fix.get_fixes(
        project=project,
        dataset=dataset,
        mip=mip,
        short_name=short_name,
        extra_facets=extra_facets,
        session=session,
        frequency=frequency,
    )
    fixed_cubes = CubeList()

    # Group cubes by input file and apply all fixes to each group element
    # (i.e., each file) individually
    by_file = defaultdict(list)
    for cube in cubes:
        by_file[cube.attributes.get("source_file", "")].append(cube)

    fixed_by_woodpecker = {
        source_file: any(is_fixed_by_woodpecker(cube) for cube in group)
        for source_file, group in by_file.items()
    }
    if 0 < sum(fixed_by_woodpecker.values()) < len(fixed_by_woodpecker):
        logger.warning(
            "Only some files of %s:%s %s %s have been fixed by woodpecker: "
            "%s. After concatenation, the data of all files is treated as "
            "fixed by woodpecker, so ESMValCore does not fix the data of the "
            "other files.",
            project,
            dataset,
            mip,
            short_name,
            ", ".join(f for f, fixed in fixed_by_woodpecker.items() if fixed),
        )

    for source_file, group in by_file.items():
        cube_list = CubeList(group)
        if fixed_by_woodpecker[source_file]:
            cube = _get_single_cube(
                cube_list,
                short_name,
                dataset_str=f"{project}:{dataset}",
            )
        else:
            for fix in fixes:
                cube_list = fix.fix_metadata(cube_list)

            # The final fix is always GenericFix, whose fix_metadata method
            # always returns a single cube
            cube = cube_list[0]

        cube.attributes.pop("source_file", None)
        fixed_cubes.append(cube)

    return fixed_cubes


def fix_data(
    cube: Cube,
    short_name: str,
    project: str,
    dataset: str,
    mip: str,
    frequency: str | None = None,
    session: Session | None = None,
    **extra_facets: Any,
) -> Cube:
    """Fix cube data if fixes are required.

    This method assumes that metadata is already fixed and checked.

    This method collects all the relevant fixes (including generic ones) for a
    given variable and applies them.

    A cube that has already been fixed by
    `woodpecker <https://github.com/roocs/woodpecker>`__, i.e. that has the
    global attribute ``woodpecker_provenance``, is returned unchanged.

    Parameters
    ----------
    cube:
        Cube to fix.
    short_name:
        Variable's short name.
    project:
        Project of the dataset.
    dataset:
        Name of the dataset.
    mip:
        Variable's MIP.
    frequency:
        Variable's data frequency, if available.
    session:
        Current session which includes configuration and directory information.
    **extra_facets:
        Extra facets. For details, see :ref:`config-extra-facets`.

    Returns
    -------
    iris.cube.Cube
        Fixed cube.

    """
    if is_fixed_by_woodpecker(cube):
        return cube

    # Update extra_facets with variable information given as regular arguments
    # to this function
    extra_facets.update(
        {
            "short_name": short_name,
            "project": project,
            "dataset": dataset,
            "mip": mip,
            "frequency": frequency,
        },
    )

    for fix in Fix.get_fixes(
        project=project,
        dataset=dataset,
        mip=mip,
        short_name=short_name,
        extra_facets=extra_facets,
        session=session,
        frequency=frequency,
    ):
        cube = fix.fix_data(cube)

    return cube
