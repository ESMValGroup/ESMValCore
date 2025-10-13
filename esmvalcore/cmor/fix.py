"""Apply automatic fixes for known errors in cmorized data.

All functions in this module will work even if no fixes are available
for the given dataset. Therefore is recommended to apply them to all
variables to be sure that all known errors are fixed.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.fix import Fix

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import ncdata
    import xarray as xr

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
    **extra_facets,
) -> str | Path | xr.Dataset | ncdata.NcData:
    """Fix files before loading them into a :class:`~iris.cube.CubeList`.

    This is mainly intended to fix errors that prevent loading the data with
    Iris (e.g., those related to ``missing_value`` or ``_FillValue``) or
    operations that are more efficient with other packages (e.g., loading files
    with lots of variables is much faster with Xarray than Iris).

    Warning
    -------
    A path should only be returned if it points to the original (unchanged)
    file (i.e., a fix was not necessary). If a fix is necessary, this function
    should return a :class:`~ncdata.NcData` or :class:`~xarray.Dataset` object.
    Under no circumstances a copy of the input data should be created (this is
    very inefficient).

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
    str | pathlib.Path | xr.Dataset | ncdata.NcData:
        Fixed data or a path to them.

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
        file = fix.fix_file(
            file,
            output_dir,
            add_unique_suffix=add_unique_suffix,
        )
    return file


def fix_metadata(
    cubes: Sequence[Cube],
    short_name: str,
    project: str,
    dataset: str,
    mip: str,
    frequency: str | None = None,
    session: Session | None = None,
    **extra_facets,
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

    # Group cubes by input file and apply all fixes to each group element
    # (i.e., each file) individually
    by_file = defaultdict(list)
    for cube in cubes:
        by_file[cube.attributes.get("source_file", "")].append(cube)

    for group in by_file.values():
        cube_list = CubeList(group)
        for fix in fixes:
            cube_list = fix.fix_metadata(cube_list)

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
    **extra_facets,
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
