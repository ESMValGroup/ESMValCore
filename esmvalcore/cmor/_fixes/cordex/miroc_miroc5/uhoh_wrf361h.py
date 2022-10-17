from esmvalcore.cmor.fix import Fix

import iris

class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Add height (2m) coordinate.

        Fix also done for prw.
        Fix latitude_bounds and longitude_bounds data type and round to 4 d.p.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        squeezed = []
        for cube in cubes:
            height = cube.coord('height')
            if isinstance(height, iris.coords.DimCoord):
                squeezed.append(iris.util.squeeze(cube))
        
        if squeezed:
            return squeezed
                    
        return cubes

# for tas / pr rcp85, lat and lon do not exist


# esmvalcore.cmor.check.CMORCheckError: There were errors in variable pr:
#  longitude: does not exist
#  latitude: does not exist
#  pr: does not match coordinate rank
# in cube:
# precipitation_flux / (kg m-2 s-1)     (time: 60; -- : 412; -- : 424)
#     Dimension coordinates:
#         time                               x        -         -
#     Cell methods:
#         mean                          time
#     Attributes:
#         CDO                           'Climate Data Operators version 1.9.4 (http://mpimet.mpg.de/cdo)'
#         CORDEX_domain                 'EUR-11'
#         Conventions                   'CF-1.4'
#         NCO                           '20180925'
#         c3s_disclaimer                'This data has been produced in the context of PRINCIPLE/CORDEX4CDS project...
#         contact                       'viktoria.mohr@uni-hohenheim.de'
#         driving_experiment            'MIROC-MIROC5, rcp85, r1i1p1'
#         driving_experiment_name       'rcp85'
#         driving_model_ensemble_member 'r1i1p1'
#         driving_model_id              'MIROC-MIROC5'
#         experiment                    'rcp85'
#         experiment_id                 'rcp85'
#         frequency                     'mon'
#         institute_id                  'UHOH'
#         institute_run_id              'hoh'
#         institution                   'Institute of Physics an Meteorology University of Hohenheim (UHOH), G...
#         model_id                      'UHOH-WRF361H'
#         nco_openmp_thread_number      1
#         product                       'output'
#         project_id                    'CORDEX'
#         rcm_version_id                'v1'
#         references                    'https://www120.uni-hohenheim.de'
#         source_file                   '/work/ik1017/C3SCORDEX/data/c3s-cordex/output/EUR-11/UHOH/MIROC-MIROC...
# loaded from file /work/ik1017/C3SCORDEX/data/c3s-cordex/output/EUR-11/UHOH/MIROC-MIROC5/rcp85/r1i1p1/UHOH-WRF361H/v1/mon/pr/v20180717/pr_EUR-11_MIROC-MIROC5_rcp85_r1i1p1_UHOH-WRF361H_v1_mon_200601-201012.nc
# 2022-10-11 10:04:10,988 UTC [3437717] INFO    
# If you have a question or need help, please start a new discussion on https://github.com/ESMValGroup/ESMValTool/discussions
# If you suspect this is a bug, please open an issue on https://github.com/ESMValGroup/ESMValTool/issues
# To make it easier to find out what the problem is, please consider attaching the files run/recipe_*.yml and run/main_log_debug.txt from the output directory.
