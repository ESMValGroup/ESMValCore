"""Tests for `esmvalcore.preprocessor._regrid_iris_esmf_regrid`."""

import esmf_regrid
import iris.cube
import numpy as np
import pytest

from esmvalcore.preprocessor.regrid_schemes import IrisESMFRegrid


class TestIrisESMFRegrid:
    def test_repr(self):
        scheme = IrisESMFRegrid(method="bilinear")
        expected = (
            "IrisESMFRegrid(method='bilinear', use_src_mask=True, "
            "use_tgt_mask=True, collapse_src_mask_along=('Z',), "
            "collapse_tgt_mask_along=('Z',), tgt_location=None, "
            "mdtol=None)"
        )
        assert repr(scheme) == expected

    def test_invalid_method_raises(self):
        msg = (
            "`method` should be one of 'bilinear', 'conservative', or "
            "'nearest'"
        )
        with pytest.raises(ValueError, match=msg):
            IrisESMFRegrid(method="x")

    def test_unused_mdtol_raises(self):
        msg = (
            "`mdol` can only be specified when `method='bilinear'` "
            "or `method='conservative'`"
        )
        with pytest.raises(TypeError, match=msg):
            IrisESMFRegrid(method="nearest", mdtol=1)

    def test_unused_src_resolution_raises(self):
        msg = (
            "`src_resolution` can only be specified when "
            "`method='conservative'`"
        )
        with pytest.raises(TypeError, match=msg):
            IrisESMFRegrid(method="nearest", src_resolution=100)

    def test_unused_tgt_resolution_raises(self):
        msg = (
            "`tgt_resolution` can only be specified when "
            "`method='conservative'`"
        )
        with pytest.raises(TypeError, match=msg):
            IrisESMFRegrid(method="nearest", tgt_resolution=100)

    def test_get_mask_2d(self):
        cube = iris.cube.Cube(
            np.ma.masked_array(np.arange(4), mask=[1, 0, 1, 0]).reshape(
                (2, 2),
            ),
            dim_coords_and_dims=(
                [
                    iris.coords.DimCoord(
                        np.arange(2),
                        standard_name="latitude",
                    ),
                    0,
                ],
                [
                    iris.coords.DimCoord(
                        np.arange(2),
                        standard_name="longitude",
                    ),
                    1,
                ],
            ),
        )
        mask = IrisESMFRegrid._get_mask(cube, ("Z",))
        np.testing.assert_array_equal(mask, cube.data.mask)

    def test_get_mask_3d(self):
        cube = iris.cube.Cube(
            np.ma.masked_array(np.arange(4), mask=[1, 0, 1, 1]).reshape(
                (2, 1, 2),
            ),
            dim_coords_and_dims=(
                [
                    iris.coords.DimCoord(
                        np.arange(2),
                        standard_name="air_pressure",
                    ),
                    0,
                ],
                [
                    iris.coords.DimCoord(
                        np.arange(1),
                        standard_name="latitude",
                    ),
                    1,
                ],
                [
                    iris.coords.DimCoord(
                        np.arange(2),
                        standard_name="longitude",
                    ),
                    2,
                ],
            ),
        )
        mask = IrisESMFRegrid._get_mask(cube, ("Z",))
        np.testing.assert_array_equal(mask, np.array([[1, 0]], dtype=bool))

    def test_get_mask_3d_odd_dim_order(self):
        cube = iris.cube.Cube(
            np.ma.masked_array(np.arange(4), mask=[1, 0, 1, 1]).reshape(
                (1, 2, 2),
            ),
            dim_coords_and_dims=(
                [
                    iris.coords.DimCoord(
                        np.arange(1),
                        standard_name="latitude",
                    ),
                    0,
                ],
                [
                    iris.coords.DimCoord(
                        np.arange(2),
                        standard_name="air_pressure",
                    ),
                    1,
                ],
                [
                    iris.coords.DimCoord(
                        np.arange(2),
                        standard_name="longitude",
                    ),
                    2,
                ],
            ),
        )
        mask = IrisESMFRegrid._get_mask(cube, ["air_pressure"])
        np.testing.assert_array_equal(mask, np.array([[1, 0]], dtype=bool))

    @pytest.mark.parametrize(
        "scheme",
        [
            ("bilinear", esmf_regrid.ESMFBilinearRegridder),
            ("conservative", esmf_regrid.ESMFAreaWeightedRegridder),
            ("nearest", esmf_regrid.ESMFNearestRegridder),
        ],
    )
    def test_regrid(self, scheme):
        method, scheme_cls = scheme
        cube = iris.cube.Cube(
            np.ma.arange(4).reshape((2, 2)),
            dim_coords_and_dims=(
                [
                    iris.coords.DimCoord(
                        np.arange(2),
                        standard_name="latitude",
                        units="degrees",
                    ),
                    0,
                ],
                [
                    iris.coords.DimCoord(
                        np.arange(2),
                        standard_name="longitude",
                        units="degrees",
                    ),
                    1,
                ],
            ),
        )

        scheme = IrisESMFRegrid(method=method)
        regridder = scheme.regridder(cube, cube)
        assert isinstance(regridder, scheme_cls)
