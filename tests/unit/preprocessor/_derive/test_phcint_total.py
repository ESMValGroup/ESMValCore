"""Test derivation of ``phcint_total``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import CubeList

from esmvalcore.preprocessor._derive import derive, get_required

if TYPE_CHECKING:
    from iris.cube import Cube


@pytest.fixture
def cubes(realistic_4d_cube: Cube) -> CubeList:
    depth_coord = DimCoord(
        [500.0],
        bounds=[[0.0, 1000.0]],
        standard_name="depth",
        units="m",
        attributes={"positive": "down"},
    )
    realistic_4d_cube.remove_coord("air_pressure")
    realistic_4d_cube.add_dim_coord(depth_coord, 1)
    realistic_4d_cube.var_name = "thetao"
    return CubeList([realistic_4d_cube])


@pytest.mark.parametrize("project", ["CMIP3", "CMIP5", "CMIP6", "CMIP7"])
def test_get_required(project: str) -> None:
    assert get_required("phcint_total", project) == [{"short_name": "thetao"}]


def test_derive(cubes: CubeList) -> None:
    short_name = "phcint_total"
    long_name = (
        "Total Column Integrated Ocean Heat Content from Potential Temperature"
    )
    units = "J m-2"
    standard_name = "integral_wrt_depth_of_sea_water_potential_temperature_expressed_as_heat_content"
    derived_cube = derive(
        cubes,
        short_name=short_name,
        long_name=long_name,
        units=units,
        standard_name=standard_name,
    )

    assert derived_cube.standard_name == standard_name
    assert derived_cube.long_name == long_name
    assert derived_cube.var_name == short_name
    assert derived_cube.units == units

    expected_data = np.ma.masked_invalid(
        [
            [
                [0.0, np.nan, np.nan],
                [np.nan, 16366760000.0, 20458450000.0],
            ],
            [
                [24550140000.0, 28641830000.0, 32733520000.0],
                [36825210000.0, 40916900000.0, 45008590000.0],
            ],
        ],
    )
    np.testing.assert_allclose(derived_cube.data, expected_data)
