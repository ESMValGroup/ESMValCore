"""Apply automatic fixes for known errors in cmorized data.

All functions in this module will work even if no fixes are available
for the given dataset. Therefore is recommended to apply them to all
variables to be sure that all known errors are fixed.
"""
import logging
from collections import defaultdict

from iris.cube import CubeList

from ._fixes.fix import Fix
from .check import CheckLevels, _get_cmor_checker

logger = logging.getLogger(__name__)


def fix_file(file, short_name, project, dataset, mip, output_dir,
             **extra_facets):
    """Fix files before ESMValTool can load them.

    This fixes are only for issues that prevent iris from loading the cube or
    that cannot be fixed after the cube is loaded.

    Original files are not overwritten.

    Parameters
    ----------
    file: str
        Path to the original file
    short_name: str
        Variable's short name
    project: str
    dataset:str
    output_dir: str
        Output directory for fixed files
    **extra_facets: dict, optional
        Extra facets are mainly used for data outside of the big projects like
        CMIP, CORDEX, obs4MIPs. For details, see :ref:`extra_facets`.

    Returns
    -------
    str:
        Path to the fixed file
    """
    for fix in Fix.get_fixes(project=project,
                             dataset=dataset,
                             mip=mip,
                             short_name=short_name,
                             extra_facets=extra_facets):
        file = fix.fix_file(file, output_dir)
    return file


def fix_metadata(cubes,
                 short_name,
                 project,
                 dataset,
                 mip,
                 frequency=None,
                 check_level=CheckLevels.DEFAULT,
                 **extra_facets):
    """Fix cube metadata if fixes are required and check it anyway.

    This method collects all the relevant fixes for a given variable, applies
    them and checks the resulting cube (or the original if no fixes were
    needed) metadata to ensure that it complies with the standards of its
    project CMOR tables.

    Parameters
    ----------
    cubes: iris.cube.CubeList
        Cubes to fix
    short_name: str
        Variable's short name
    project: str

    dataset: str

    mip: str
        Variable's MIP

    frequency: str, optional
        Variable's data frequency, if available
    check_level: CheckLevels
        Level of strictness of the checks. Set to default.
    **extra_facets: dict, optional
        Extra facets are mainly used for data outside of the big projects like
        CMIP, CORDEX, obs4MIPs. For details, see :ref:`extra_facets`.

    Returns
    -------
    iris.cube.Cube:
        Fixed and checked cube

    Raises
    ------
    CMORCheckError
        If the checker detects errors in the metadata that it can not fix.
    """
    fixes = Fix.get_fixes(project=project,
                          dataset=dataset,
                          mip=mip,
                          short_name=short_name,
                          extra_facets=extra_facets)
    fixed_cubes = []
    by_file = defaultdict(list)
    for cube in cubes:
        by_file[cube.attributes.get('source_file', '')].append(cube)

    for cube_list in by_file.values():
        cube_list = CubeList(cube_list)
        for fix in fixes:
            cube_list = fix.fix_metadata(cube_list)

        cube = _get_single_cube(cube_list, short_name, project, dataset)
        checker = _get_cmor_checker(frequency=frequency,
                                    table=project,
                                    mip=mip,
                                    short_name=short_name,
                                    check_level=check_level,
                                    fail_on_error=False,
                                    automatic_fixes=True)
        cube = checker(cube).check_metadata()
        cube.attributes.pop('source_file', None)
        fixed_cubes.append(cube)
    return fixed_cubes


def _get_single_cube(cube_list, short_name, project, dataset):
    if len(cube_list) == 1:
        return cube_list[0]
    cube = None
    for raw_cube in cube_list:
        if raw_cube.var_name == short_name:
            cube = raw_cube
            break
    if not cube:
        raise ValueError(
            'More than one cube found for variable %s in %s:%s but '
            'none of their var_names match the expected. \n'
            'Full list of cubes encountered: %s' %
            (short_name, project, dataset, cube_list))
    logger.warning(
        'Found variable %s in %s:%s, but there were other present in '
        'the file. Those extra variables are usually metadata '
        '(cell area, latitude descriptions) that was not saved '
        'according to CF-conventions. It is possible that errors appear '
        'further on because of this. \nFull list of cubes encountered: %s',
        short_name, project, dataset, cube_list)
    return cube


def fix_data(cube,
             short_name,
             project,
             dataset,
             mip,
             frequency=None,
             check_level=CheckLevels.DEFAULT,
             **extra_facets):
    """Fix cube data if fixes add present and check it anyway.

    This method assumes that metadata is already fixed and checked.

    This method collects all the relevant fixes for a given variable, applies
    them and checks resulting cube (or the original if no fixes were
    needed) metadata to ensure that it complies with the standards of its
    project CMOR tables.

    Parameters
    ----------
    cube: iris.cube.Cube
        Cube to fix
    short_name: str
        Variable's short name
    project: str
    dataset: str
    mip: str
        Variable's MIP
    frequency: str, optional
        Variable's data frequency, if available
    check_level: CheckLevels
        Level of strictness of the checks. Set to default.
    **extra_facets: dict, optional
        Extra facets are mainly used for data outside of the big projects like
        CMIP, CORDEX, obs4MIPs. For details, see :ref:`extra_facets`.

    Returns
    -------
    iris.cube.Cube:
        Fixed and checked cube

    Raises
    ------
    CMORCheckError
        If the checker detects errors in the data that it can not fix.
    """
    for fix in Fix.get_fixes(project=project,
                             dataset=dataset,
                             mip=mip,
                             short_name=short_name,
                             extra_facets=extra_facets):
        cube = fix.fix_data(cube)
    checker = _get_cmor_checker(frequency=frequency,
                                table=project,
                                mip=mip,
                                short_name=short_name,
                                fail_on_error=False,
                                automatic_fixes=True,
                                check_level=check_level)
    cube = checker(cube).check_data()
    return cube
