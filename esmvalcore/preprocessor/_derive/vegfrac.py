"""Derivation of variable `vegFrac`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import iris
from iris import NameConstraint

from esmvalcore.preprocessor._regrid import regrid

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from iris.cube import Cube, CubeList

    from esmvalcore.typing import Facets


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `vegFrac`."""

    @staticmethod
    def required(facets: Facets) -> list[Facets]:
        """Declare the variables needed for derivation."""
        project = facets["project"]
        sftlf: Facets = {"short_name": "sftlf", "mip": "fx"}
        if project == "CMIP5":
            sftlf["ensemble"] = "r0i0p0"
            return [
                {"short_name": "baresoilFrac"},
                {"short_name": "residualFrac"},
                sftlf,
            ]
        elif project == "CMIP6":
            return [
                {"short_name": "baresoilFrac"},
                {"short_name": "residualFrac"},
                sftlf,
            ]
        elif project == "CMIP7":
            return [
                {
                    "short_name": "baresoilFrac",
                    "branding_suffix": "tavg-u-hxy-u",
                    "mip": "land",
                },
                {
                    "short_name": "residualFrac",
                    "branding_suffix": "tavg-u-hxy-u",
                    "mip": "land",
                },
                {
                    "short_name": "sftlf",
                    "branding_suffix": "ti-u-hxy-u",
                    "frequency": "fx",
                    "mip": "atmos",
                },
            ]
        else:
            msg = (
                f"Project {project} cannot be used for derivation of 'vegfrac'"
            )
            raise ValueError(msg)
        

    @staticmethod
    def calculate(cubes: CubeList) -> Cube:
        """Compute vegetation fraction from bare soil fraction."""
        baresoilfrac_cube = cubes.extract_cube(
            NameConstraint(var_name="baresoilFrac"),
        )
        residualfrac_cube = cubes.extract_cube(
            NameConstraint(var_name="residualFrac"),
        )
        sftlf_cube = cubes.extract_cube(NameConstraint(var_name="sftlf"))

        # Add time dimension to sftlf
        target_shape_sftlf = (baresoilfrac_cube.shape[0], *sftlf_cube.shape)
        sftlf_data = iris.util.broadcast_to_shape(
            sftlf_cube.data,
            target_shape_sftlf,
            (1, 2),
        )
        sftlf_cube = baresoilfrac_cube.copy(sftlf_data)
        sftlf_cube.data = sftlf_cube.lazy_data()

        # Regrid sftlf if necessary and adapt mask
        if sftlf_cube.shape != baresoilfrac_cube.shape:
            sftlf_cube = regrid(sftlf_cube, baresoilfrac_cube, "linear")
        sftlf_cube.data = da.ma.masked_array(
            sftlf_cube.core_data(),
            mask=da.ma.getmaskarray(baresoilfrac_cube.core_data()),
        )

        # Calculate vegetation fraction
        baresoilfrac_cube.data = (
            sftlf_cube.core_data()
            - baresoilfrac_cube.core_data()
            - residualfrac_cube.core_data()
        )
        return baresoilfrac_cube
