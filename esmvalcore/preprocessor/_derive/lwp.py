"""Derivation of variable `lwp`."""

import logging

from ._baseclass import DerivedVariableBase
from ._shared import _var_name_constraint

logger = logging.getLogger(__name__)


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `lwp`."""

    # Required variables
    required = [
        {
            'short_name': 'clwvi'
        },
        {
            'short_name': 'clivi'
        },
    ]

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
        clwvi_cube = cubes.extract_strict(_var_name_constraint('clwvi'))
        clivi_cube = cubes.extract_strict(_var_name_constraint('clivi'))

        dataset = clwvi_cube.attributes.get('model_id')
        project = clwvi_cube.attributes.get('project_id')
        # Should we check that the model_id/project_id are the same on both
        # cubes?

        bad_datasets = [
            'CESM1-CAM5-1-FV2',
            'CESM1-CAM5',
            'CMCC-CESM',
            'CMCC-CM',
            'CMCC-CMS',
            'IPSL-CM5A-MR',
            'IPSL-CM5A-LR',
            'IPSL-CM5B-LR',
            'CCSM4',
            'IPSL-CM5A-MR',
            'MIROC-ESM',
            'MIROC-ESM-CHEM',
            'MIROC-ESM',
            'CSIRO-Mk3-6-0',
            'MPI-ESM-MR',
            'MPI-ESM-LR',
            'MPI-ESM-P',
        ]
        if (project in ["CMIP5", "CMIP5_ETHZ"] and dataset in bad_datasets):
            logger.info(
                "Assuming that variable clwvi from %s dataset %s "
                "contains only liquid water", project, dataset)
            lwp_cube = clwvi_cube
        else:
            lwp_cube = clwvi_cube - clivi_cube

        return lwp_cube
