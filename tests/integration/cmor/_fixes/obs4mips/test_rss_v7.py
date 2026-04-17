"""Test RSS-v7 fixes."""

import iris.coords
import iris.cube
import numpy as np

from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor._fixes.obs4mips.rss_v7 import Prw
from esmvalcore.cmor.fix import Fix


class TestPrw:
    """Test prw fixes."""

    def test_get(self):
        """Test fix get."""
        assert Fix.get_fixes("obs4MIPs", "RSS-v7", "Amon", "prw") == [
            Prw(None),
            GenericFix(None),
        ]

    def test_fix_coords(self):
        """Test fix of coordinates."""
        cube = iris.cube.Cube(
            np.zeros((1, 1, 1)),
            var_name="prw",
            standard_name="atmosphere_water_vapor_content",
            units="kg m-2",
            dim_coords_and_dims=[
                (iris.coords.DimCoord([0], standard_name="time"), 0),
                (
                    iris.coords.DimCoord(
                        [0],
                        var_name="latitude",
                        standard_name="latitude",
                    ),
                    1,
                ),
                (
                    iris.coords.DimCoord(
                        [0],
                        var_name="longitude",
                        standard_name="longitude",
                    ),
                    2,
                ),
            ],
        )
        fixed_cube = Prw(None).fix_metadata([cube])[0]
        assert fixed_cube.coord("latitude").var_name == "lat"
        assert fixed_cube.coord("longitude").var_name == "lon"
