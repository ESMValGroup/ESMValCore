"""Derivation of variable `lwp`."""

import logging

from iris import NameConstraint

from ._baseclass import DerivedVariableBase

logger = logging.getLogger(__name__)


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `lwp`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'clwvi'
            },
            {
                'short_name': 'clivi'
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute liquid water path.

        Note
        ----
        Some datasets output the variable `clwvi` which only contains `lwp`. In
        these cases, the input `clwvi` cube is just returned.

        """
        # CMIP5 and CMIP6 names are slightly different, so use
        # variable name instead to extract cubes
        clwvi_cube = cubes.extract_cube(NameConstraint(var_name='clwvi'))
        clivi_cube = cubes.extract_cube(NameConstraint(var_name='clivi'))

        # CMIP5 and CMIP6 have different global attributes that we use
        # to determine model name and project name:
        #   - CMIP5: model_id and project_id
        #   - CMIP6: source_id and mip_era
        project = clwvi_cube.attributes.get('project_id')
        if project:
            dataset = clwvi_cube.attributes.get('model_id')
            # some CMIP6 models define both, project_id and source_id but
            # no model_id --> also try source_id to find model name
            if not dataset:
                dataset = clwvi_cube.attributes.get('source_id')
        else:
            project = clwvi_cube.attributes.get('mip_era')
            dataset = clwvi_cube.attributes.get('source_id')

        # Should we check that the model_id/project_id are the same on both
        # cubes?

        bad_datasets = [
            'CCSM4',           # CMIP5 models
            'CESM1-CAM5-1-FV2',
            'CESM1-CAM5',
            'CMCC-CESM',
            'CMCC-CM',
            'CMCC-CMS',
            'CSIRO-Mk3-6-0',
            'GISS-E2-1-G',
            'GISS-E2-1-H',
            'IPSL-CM5A-MR',
            'IPSL-CM5A-LR',
            'IPSL-CM5B-LR',
            'IPSL-CM5A-MR',
            'MIROC-ESM',
            'MIROC-ESM-CHEM',
            'MIROC-ESM',
            'MPI-ESM-LR',
            'MPI-ESM-MR',
            'MPI-ESM-P',
            'AWI-ESM-1-1-LR',   # CMIP6 models
            'CAMS-CSM1-0',
            'FGOALS-f3-L',
            'IPSL-CM6A-LR',
            'MPI-ESM-1-2-HAM',
            'MPI-ESM1-2-HR',
            'MPI-ESM1-2-LR',
            'SAM0-UNICON'
        ]
        affected_projects = ["CMIP5", "CMIP5_ETHZ", "CMIP6"]
        if (project in affected_projects and dataset in bad_datasets):
            logger.info(
                "Assuming that variable clwvi from %s dataset %s "
                "contains only liquid water", project, dataset)
            lwp_cube = clwvi_cube
        else:
            lwp_cube = clwvi_cube - clivi_cube

        return lwp_cube
