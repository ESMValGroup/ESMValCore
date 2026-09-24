"""Derivation of variable `lwp`."""

import logging

from iris import NameConstraint

from ._baseclass import DerivedVariableBase

logger = logging.getLogger(__name__)


LWP_ONLY_DATASETS = [
    # CMIP5 models
    {"project_id": "CMIP5", "model_id": "CCSM4"},
    {"project_id": "CMIP5", "model_id": "CESM1-CAM5-1-FV2"},
    {"project_id": "CMIP5", "model_id": "CESM1-CAM5"},
    {"project_id": "CMIP5", "model_id": "CMCC-CESM"},
    {"project_id": "CMIP5", "model_id": "CMCC-CM"},
    {"project_id": "CMIP5", "model_id": "CMCC-CMS"},
    {"project_id": "CMIP5", "model_id": "CSIRO-Mk3-6-0"},
    {"project_id": "CMIP5", "model_id": "GISS-E2-1-G"},
    {"project_id": "CMIP5", "model_id": "GISS-E2-1-H"},
    {"project_id": "CMIP5", "model_id": "IPSL-CM5A-MR"},
    {"project_id": "CMIP5", "model_id": "IPSL-CM5A-LR"},
    {"project_id": "CMIP5", "model_id": "IPSL-CM5B-LR"},
    {"project_id": "CMIP5", "model_id": "IPSL-CM5A-MR"},
    {"project_id": "CMIP5", "model_id": "MIROC-ESM"},
    {"project_id": "CMIP5", "model_id": "MIROC-ESM-CHEM"},
    {"project_id": "CMIP5", "model_id": "MIROC-ESM"},
    {"project_id": "CMIP5", "model_id": "MPI-ESM-LR"},
    {"project_id": "CMIP5", "model_id": "MPI-ESM-MR"},
    {"project_id": "CMIP5", "model_id": "MPI-ESM-P"},
    # CMIP6 models
    {"mip_era": "CMIP6", "source_id": "AWI-ESM-1-1-LR"},
    {"mip_era": "CMIP6", "source_id": "CAMS-CSM1-0"},
    {"mip_era": "CMIP6", "source_id": "FGOALS-f3-L"},
    {"mip_era": "CMIP6", "source_id": "IPSL-CM6A-LR"},
    {"mip_era": "CMIP6", "source_id": "MPI-ESM-1-2-HAM"},
    {"mip_era": "CMIP6", "source_id": "MPI-ESM1-2-HR"},
    {"mip_era": "CMIP6", "source_id": "MPI-ESM1-2-LR"},
    {"mip_era": "CMIP6", "source_id": "SAM0-UNICON"},
    # CORDEX-CMIP5 models
    {"project_id": "CORDEX", "model_id": "SMHI-RCA4"},
    {
        "project_id": "CORDEX",
        "model_id": "CLMcom-CCLM4-8-17",
        "driving_model_id": "MOHC-HadGEM2-ES",
    },
]


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `lwp`."""

    @staticmethod
    def required(project):  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "clwvi"},
            {"short_name": "clivi"},
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
        clwvi_cube = cubes.extract_cube(NameConstraint(var_name="clwvi"))
        clivi_cube = cubes.extract_cube(NameConstraint(var_name="clivi"))

        # Should we check that the model_id/project_id are the same on both
        # cubes?

        for dataset in LWP_ONLY_DATASETS:
            if all(
                clwvi_cube.attributes.get(k) == v for k, v in dataset.items()
            ):
                logger.info(
                    "Assuming that variable clwvi from %s contains only liquid water",
                    ", ".join(f"{k}={v}" for k, v in dataset.items()),
                )
                lwp_cube = clwvi_cube
                break
        else:
            lwp_cube = clwvi_cube - clivi_cube

        return lwp_cube
