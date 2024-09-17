"""Apply automatic fixes for known errors in cmorized data.

All functions in this module will work even if no fixes are available
for the given dataset. Therefore is recommended to apply them to all
variables to be sure that all known errors are fixed.
"""
from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor.check import CheckLevels, _get_cmor_checker
from esmvalcore.exceptions import ESMValCoreDeprecationWarning

if TYPE_CHECKING:
    from ..config import Session

logger = logging.getLogger(__name__)


def fix_file(
    file: Path,
    short_name: str,
    project: str,
    dataset: str,
    mip: str,
    output_dir: Path,
    add_unique_suffix: bool = False,
    session: Optional[Session] = None,
    frequency: Optional[str] = None,
    **extra_facets,
) -> str | Path:
    """Fix files before ESMValTool can load them.

    These fixes are only for issues that prevent iris from loading the cube or
    that cannot be fixed after the cube is loaded.

    Original files are not overwritten.

    Parameters
    ----------
    file:
        Path to the original file.
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
        Adds a unique suffix to `output_dir` for thread safety.
    session:
        Current session which includes configuration and directory information.
    frequency:
        Variable's data frequency, if available.
    **extra_facets:
        Extra facets are mainly used for data outside of the big projects like
        CMIP, CORDEX, obs4MIPs. For details, see :ref:`extra_facets`.

    Returns
    -------
    str or pathlib.Path
        Path to the fixed file.

    """
    # Update extra_facets with variable information given as regular arguments
    # to this function
    extra_facets.update({
        'short_name': short_name,
        'project': project,
        'dataset': dataset,
        'mip': mip,
        'frequency': frequency,
    })

    for fix in Fix.get_fixes(project=project,
                             dataset=dataset,
                             mip=mip,
                             short_name=short_name,
                             extra_facets=extra_facets,
                             session=session,
                             frequency=frequency):
        file = fix.fix_file(
            file, output_dir, add_unique_suffix=add_unique_suffix
        )
    return file


def fix_metadata(
    cubes: Sequence[Cube],
    short_name: str,
    project: str,
    dataset: str,
    mip: str,
    frequency: Optional[str] = None,
    check_level: CheckLevels = CheckLevels.DEFAULT,
    session: Optional[Session] = None,
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
    check_level:
        Level of strictness of the checks.

        .. deprecated:: 2.10.0
            This option has been deprecated in ESMValCore version 2.10.0 and is
            scheduled for removal in version 2.12.0. Please use the functions
            :func:`~esmvalcore.preprocessor.cmor_check_metadata`,
            :func:`~esmvalcore.preprocessor.cmor_check_data`, or
            :meth:`~esmvalcore.cmor.check.cmor_check` instead. This function
            will no longer perform CMOR checks. Fixes and CMOR checks have been
            clearly separated in ESMValCore version 2.10.0.
    session:
        Current session which includes configuration and directory information.
    **extra_facets:
        Extra facets are mainly used for data outside of the big projects like
        CMIP, CORDEX, obs4MIPs. For details, see :ref:`extra_facets`.

    Returns
    -------
    iris.cube.CubeList
        Fixed cubes.

    """
    # Deprecate CMOR checks (remove in v2.12)
    if check_level != CheckLevels.DEFAULT:
        msg = (
            "The option `check_level` has been deprecated in ESMValCore "
            "version 2.10.0 and is scheduled for removal in version 2.12.0. "
            "Please use the functions "
            "esmvalcore.preprocessor.cmor_check_metadata, "
            "esmvalcore.preprocessor.cmor_check_data, or "
            "esmvalcore.cmor.check.cmor_check instead. This function will no "
            "longer perform CMOR checks. Fixes and CMOR checks have been "
            "clearly separated in ESMValCore version 2.10.0."
        )
        warnings.warn(msg, ESMValCoreDeprecationWarning)

    # Update extra_facets with variable information given as regular arguments
    # to this function
    extra_facets.update({
        'short_name': short_name,
        'project': project,
        'dataset': dataset,
        'mip': mip,
        'frequency': frequency,
    })

    fixes = Fix.get_fixes(project=project,
                          dataset=dataset,
                          mip=mip,
                          short_name=short_name,
                          extra_facets=extra_facets,
                          session=session,
                          frequency=frequency)
    fixed_cubes = CubeList()

    # Group cubes by input file and apply all fixes to each group element
    # (i.e., each file) individually
    by_file = defaultdict(list)
    for cube in cubes:
        by_file[cube.attributes.get('source_file', '')].append(cube)

    for cube_list in by_file.values():
        cube_list = CubeList(cube_list)
        for fix in fixes:
            cube_list = fix.fix_metadata(cube_list)

        # The final fix is always GenericFix, whose fix_metadata method always
        # returns a single cube
        cube = cube_list[0]

        # Perform CMOR checks
        # TODO: remove in v2.12
        checker = _get_cmor_checker(
            project,
            mip,
            short_name,
            frequency,
            fail_on_error=False,
            check_level=check_level,
        )
        cube = checker(cube).check_metadata()

        cube.attributes.pop('source_file', None)
        fixed_cubes.append(cube)

    return fixed_cubes


def fix_data(
    cube: Cube,
    short_name: str,
    project: str,
    dataset: str,
    mip: str,
    frequency: Optional[str] = None,
    check_level: CheckLevels = CheckLevels.DEFAULT,
    session: Optional[Session] = None,
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
    check_level:
        Level of strictness of the checks.

        .. deprecated:: 2.10.0
            This option has been deprecated in ESMValCore version 2.10.0 and is
            scheduled for removal in version 2.12.0. Please use the functions
            :func:`~esmvalcore.preprocessor.cmor_check_metadata`,
            :func:`~esmvalcore.preprocessor.cmor_check_data`, or
            :meth:`~esmvalcore.cmor.check.cmor_check` instead. This function
            will no longer perform CMOR checks. Fixes and CMOR checks have been
            clearly separated in ESMValCore version 2.10.0.
    session:
        Current session which includes configuration and directory information.
    **extra_facets:
        Extra facets are mainly used for data outside of the big projects like
        CMIP, CORDEX, obs4MIPs. For details, see :ref:`extra_facets`.

    Returns
    -------
    iris.cube.Cube
        Fixed cube.

    """
    # Deprecate CMOR checks (remove in v2.12)
    if check_level != CheckLevels.DEFAULT:
        msg = (
            "The option `check_level` has been deprecated in ESMValCore "
            "version 2.10.0 and is scheduled for removal in version 2.12.0. "
            "Please use the functions "
            "esmvalcore.preprocessor.cmor_check_metadata, "
            "esmvalcore.preprocessor.cmor_check_data, or "
            "esmvalcore.cmor.check.cmor_check instead. This function will no "
            "longer perform CMOR checks. Fixes and CMOR checks have been "
            "clearly separated in ESMValCore version 2.10.0."
        )
        warnings.warn(msg, ESMValCoreDeprecationWarning)

    # Update extra_facets with variable information given as regular arguments
    # to this function
    extra_facets.update({
        'short_name': short_name,
        'project': project,
        'dataset': dataset,
        'mip': mip,
        'frequency': frequency,
    })

    for fix in Fix.get_fixes(project=project,
                             dataset=dataset,
                             mip=mip,
                             short_name=short_name,
                             extra_facets=extra_facets,
                             session=session,
                             frequency=frequency):
        cube = fix.fix_data(cube)

    # Perform CMOR checks
    # TODO: remove in v2.12
    checker = _get_cmor_checker(
        project,
        mip,
        short_name,
        frequency,
        fail_on_error=False,
        check_level=check_level,
    )
    cube = checker(cube).check_data()

    return cube
