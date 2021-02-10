"""
Preprocessor functions that do not fit into any of the categories.
"""

import logging
import iris

import dask.array as da

logger = logging.getLogger(__name__)


def clip(cube, minimum=None, maximum=None):
    """
    Clip values at a specified minimum and/or maximum value

    Values lower than minimum are set to minimum and values
    higher than maximum are set to maximum.

    Parameters
    ----------
    cube: iris.cube.Cube
        iris cube to be clipped
    minimum: float
        lower threshold to be applied on input cube data.
    maximum: float
        upper threshold to be applied on input cube data.

    Returns
    -------
    iris.cube.Cube
        clipped cube.
    """
    if minimum is None and maximum is None:
        raise ValueError("Either minimum, maximum or both have to be\
                          specified.")
    elif minimum is not None and maximum is not None:
        if maximum < minimum:
            raise ValueError("Maximum should be equal or larger than minimum.")
    cube.data = da.clip(cube.core_data(), minimum, maximum)
    return cube


def add_cell_measure(cube, fx_variables, project, dataset, check_level):
    """
    Load requested fx files, check with CMOR standards and add the
    fx variables as cell measures in the cube containing the data.

    Parameters
    ----------
    cube: iris.cube.Cube
        Iris cube with input data.
    fx_variables: dict
        Path to the needed fx_files.
    project: str

    dataset: str

    check_level: CheckLevels
        Level of strictness of the checks.


    Returns
    -------
    iris.cube.Cube
        Cube with added cell measures.
    """
    from esmvalcore.preprocessor._io import concatenate
    from esmvalcore.cmor.fix import fix_metadata, fix_data
    from esmvalcore.cmor.check import cmor_check_metadata, cmor_check_data

    if not fx_variables:
        return cube
    fx_cubes = iris.cube.CubeList()
    for fx_files in fx_variables.values():
        if isinstance(fx_files, str):
            fx_files = [fx_files]
        if not fx_files:
            continue
        for fx_file in fx_files:
            loaded_cube = iris.load(fx_file)
            short_name = loaded_cube[0].var_name
            mip = loaded_cube[0].attributes['table_id']
            freq = loaded_cube[0].attributes['frequency']

            loaded_cube = fix_metadata(loaded_cube, short_name=short_name,
                                       project=project, dataset=dataset,
                                       mip=mip, frequency=freq,
                                       check_level=check_level)
            fx_cubes.append(loaded_cube[0])

        fx_cube = concatenate(fx_cubes)

        fx_cube = cmor_check_metadata(fx_cube, cmor_table=project, mip=mip,
                                      short_name=short_name, frequency=freq,
                                      check_level=check_level)

        fx_cube = fix_data(fx_cube, short_name=short_name, project=project,
                           dataset=dataset, mip=mip, frequency=freq,
                           check_level=check_level)

        fx_cube = cmor_check_data(fx_cube, cmor_table=project, mip=mip,
                                  short_name=fx_cube.var_name, frequency=freq,
                                  check_level=check_level)

        measure_name = {
            'areacella': 'area',
            'areacello': 'area',
            'volcello': 'volume'
            }

        if fx_cube.var_name in measure_name.keys():
            try:
                fx_data = da.broadcast_to(
                    fx_cube.core_data(), cube.shape)
            except ValueError:
                raise ValueError(
                    f"Frequencies of {cube.var_name} and "
                    f"{fx_cube.var_name} cubes do not match."
                )
            measure = iris.coords.CellMeasure(
                fx_data,
                standard_name=fx_cube.standard_name,
                units=fx_cube.units,
                measure=measure_name[fx_cube.var_name],
                var_name=fx_cube.var_name,
                attributes=fx_cube.attributes)
            cube.add_cell_measure(measure, range(0, measure.ndim))
            logger.info(f'Added {fx_cube.var_name} '
                        f'as cell measure in cube of {cube.var_name}.')
        else:
            logger.info(f'Fx variable {fx_cube.var_name} '
                        'cannot be added as a cell measure '
                        f'in cube of {cube.var_name}.')
    return cube
