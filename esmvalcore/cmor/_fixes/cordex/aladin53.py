"""Fixes for DMI-HIRHAM5 model."""
import iris

from ..common import ClFixHybridPressureCoord
from ..fix import Fix
from ..shared import add_scalar_height_coord

# import IPython
# from traitlets.config import get_config
# c = get_config()
# c.InteractiveShellEmbed.colors = "Linux"

from esmvalcore.preprocessor._shared import guess_bounds
import numpy as np


class AllVars(Fix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        fixed_cubes = iris.cube.CubeList()

        # IPython.embed(config=c)
        for cube in cubes:
            if cube.attributes['CORDEX_domain'] == 'EUR-11':
                if len(cube.dim_coords) < 2 and len(cube.shape) > 1:
                    for cname, cindex in zip(['projection_x_coordinate',
                                              'projection_y_coordinate'],
                                             [2, 1]):
                        coord = cube.coord(cname)
                        coord.points = coord.points.astype('float32')
                        coord.points = np.linspace(coord.points[0],
                                                   coord.points[-1],
                                                   coord.shape[0])
                        newcoord = iris.coords.DimCoord(coord.points.data,
                                            var_name=coord.var_name,
                                            standard_name=coord.standard_name,
                                            long_name=coord.long_name,
                                            units=coord.units,
                                            coord_system=coord.coord_system)
                        cube.remove_coord(cname)
                        cube.add_dim_coord(newcoord, cindex)
                cube = guess_bounds(cube, ['projection_x_coordinate',
                                           'projection_y_coordinate'])
            fixed_cubes.append(cube)
        return fixed_cubes

    # def fix_metadata(self, cubes):
    #     """Fix metadata."""
    #     fixed_cubes = iris.cube.CubeList()
    #     for cube in cubes:
    #         cube.data = cube.core_data().astype('float32')
    #         fixed_cubes.append(cube)

    #     return fixed_cubes


        # IPython.embed(config=c)
        #      # No idea, but there is only one cube
        # for cube in cubes:
        #     if len(cube.dim_coords) < 2 and len(cube.shape) > 1:
        #         print('here')
        #         IPython.embed(config=c)

#        return fixed_cubes


    # def fix_file(file, short_name, project, dataset, mip, output_dir):
    #     """
    #     Fix files before ESMValTool can load them.

    #     This fixes are only for issues that prevent iris from loading the cube or
    #     that cannot be fixed after the cube is loaded.

    #     Original files are not overwritten.

    #     Parameters
    #     ----------
    #     file: str
    #         Path to the original file
    #     short_name: str
    #         Variable's short name
    #     project: str
    #     dataset:str
    #     output_dir: str
    #         Output directory for fixed files

    #     Returns
    #     -------
    #     str:
    #         Path to the fixed file

    #     """
    #     for fix in Fix.get_fixes(
    #             project=project, dataset=dataset, mip=mip, short_name=short_name):
    #         file = fix.fix_file(file, output_dir)
    #     return file
