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
from esmvalcore.io.local import LocalFile, _get_start_end_date

if TYPE_CHECKING:
    from collections.abc import Iterable

    from esmvalcore.config import Session

logger = logging.getLogger(__name__)


def fix_file(  # noqa: PLR0913
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


def _group_cubes(fixes: Iterable[Fix], cubes: CubeList) -> dict[Any, CubeList]:
    """Group cubes for fix_metadata; each group is processed individually."""
    grouped_cubes: dict[Any, CubeList] = defaultdict(CubeList)

    # Group by date
    if any(fix.GROUP_CUBES_BY_DATE for fix in fixes):
        for cube in cubes:
            if "source_file" in cube.attributes:
                dates = _get_start_end_date(cube.attributes["source_file"])
            else:
                dates = None
            grouped_cubes[dates].append(cube)

    # Group by file name
    else:
        for cube in cubes:
            grouped_cubes[cube.attributes.get("source_file", "")].append(cube)

    return grouped_cubes


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

    # Group cubes and apply all fixes to each group element individually. There
    # are two options for grouping:
    # (1) By input file name (default).
    # (2) By time range (can be enabled by setting the attribute
    #     GROUP_CUBES_BY_DATE=True for the fix class; see
    #     _fixes.native6.era5.Rsut for an example).
    grouped_cubes = _group_cubes(fixes, cubes)
    for group in grouped_cubes.values():
        for fix in fixes:
            cube_list = fix.fix_metadata(group)

        # The final fix is always GenericFix, whose fix_metadata method always
        # returns a single cube
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
