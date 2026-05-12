"""Integration tests for :func:`esmvalcore.preprocessor.regrid`."""

import iris
import numpy as np
import pytest
from numpy import ma

from esmvalcore.dataset import Dataset
from esmvalcore.preprocessor import regrid
from tests import assert_array_equal
from tests.unit.preprocessor._regrid import _make_cube


class Test:
    @pytest.fixture(autouse=True)
    def setUp(self):
        """Prepare tests."""
        shape = (3, 2, 2)
        data = np.arange(np.prod(shape, dtype=float)).reshape(shape)
        self.cube = _make_cube(data)
        self.cs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)

        # Setup grid for linear regridding
        lons = iris.coords.DimCoord(
            [1.5],
            standard_name="longitude",
            bounds=[[1, 2]],
            units="degrees_east",
            coord_system=self.cs,
        )
        lats = iris.coords.DimCoord(
            [1.5],
            standard_name="latitude",
            bounds=[[1, 2]],
            units="degrees_north",
            coord_system=self.cs,
        )
        coords_spec = [(lats, 0), (lons, 1)]
        self.grid_for_linear = iris.cube.Cube(
            np.empty((1, 1)),
            dim_coords_and_dims=coords_spec,
        )

        # Setup cube with multiple horizontal dimensions
        self.multidim_cube = _make_cube(data, grid="rotated", aux_coord=False)
        lats, lons = np.meshgrid(
            np.arange(1, data.shape[-2] + 1),
            np.arange(1, data.shape[-1] + 1),
        )
        self.multidim_cube.add_aux_coord(
            iris.coords.AuxCoord(
                lats,
                var_name="lat",
                standard_name="latitude",
                units="degrees",
            ),
            (1, 2),
        )
        self.multidim_cube.add_aux_coord(
            iris.coords.AuxCoord(
                lons,
                var_name="lon",
                standard_name="longitude",
                units="degrees",
            ),
            (1, 2),
        )

        # Setup mesh cube
        self.mesh_cube = _make_cube(data, grid="mesh")

        # Setup unstructured cube and grid
        lons = iris.coords.DimCoord(
            [1.6],
            standard_name="longitude",
            bounds=[[1, 2]],
            units="degrees_east",
            coord_system=self.cs,
        )
        lats = iris.coords.DimCoord(
            [1.6],
            standard_name="latitude",
            bounds=[[1, 2]],
            units="degrees_north",
            coord_system=self.cs,
        )
        coords_spec = [(lats, 0), (lons, 1)]
        self.tgt_grid_for_unstructured = iris.cube.Cube(
            np.zeros((1, 1)),
            dim_coords_and_dims=coords_spec,
        )

        lons = self.cube.coord("longitude")
        lats = self.cube.coord("latitude")
        x, y = np.meshgrid(lons.points, lats.points)

        lats = iris.coords.AuxCoord(
            y.ravel(),
            standard_name=lats.metadata.standard_name,
            long_name=lats.metadata.long_name,
            var_name=lats.metadata.var_name,
            units=lats.metadata.units,
            attributes=lats.metadata.attributes,
            coord_system=lats.metadata.coord_system,
            climatological=lats.metadata.climatological,
        )

        lons = iris.coords.AuxCoord(
            x.ravel(),
            standard_name=lons.metadata.standard_name,
            long_name=lons.metadata.long_name,
            var_name=lons.metadata.var_name,
            units=lons.metadata.units,
            attributes=lons.metadata.attributes,
            coord_system=lons.metadata.coord_system,
            climatological=lons.metadata.climatological,
        )

        unstructured_data = np.ma.masked_less(
            self.cube.data.reshape(3, -1).astype(np.float32),
            3.5,
        )

        self.unstructured_grid_cube = iris.cube.Cube(
            unstructured_data,
            dim_coords_and_dims=[(self.cube.coord("air_pressure"), 0)],
            aux_coords_and_dims=[(lats, 1), (lons, 1)],
        )
        self.unstructured_grid_cube.metadata = self.cube.metadata

        # Setup irregular cube and grid
        lons_2d = iris.coords.AuxCoord(
            [[0, 1]],
            standard_name="longitude",
            units="degrees_east",
        )
        lats_2d = iris.coords.AuxCoord(
            [[0, 1]],
            standard_name="latitude",
            units="degrees_north",
        )
        self.irregular_grid = iris.cube.Cube(
            [[1, 1]],
            aux_coords_and_dims=[(lats_2d, (0, 1)), (lons_2d, (0, 1))],
        )

    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_regrid__linear(self, cache_weights):
        result = regrid(
            self.cube,
            self.grid_for_linear,
            "linear",
            cache_weights=cache_weights,
        )
        expected = np.array([[[1.5]], [[5.5]], [[9.5]]])
        assert_array_equal(result.data, expected)

    def test_regrid__linear_multidim(self):
        """Test that latitude/longitude are used by default."""
        result = regrid(
            self.multidim_cube,
            self.grid_for_linear,
            "linear",
        )
        expected = np.ma.masked_array([[[1.5]], [[5.5]], [[9.5]]], mask=False)
        result.data = np.round(result.data, 1)
        assert_array_equal(result.data, expected)

    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_regrid__linear_file(self, tmp_path, cache_weights):
        file = tmp_path / "file.nc"
        iris.save(self.grid_for_linear, target=file)
        result = regrid(self.cube, file, "linear", cache_weights=cache_weights)
        expected = np.array([[[1.5]], [[5.5]], [[9.5]]])
        assert_array_equal(result.data, expected)

    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_regrid__linear_dataset(self, monkeypatch, cache_weights):
        monkeypatch.setattr(Dataset, "files", ["file.nc"])

        def load(_):
            return self.grid_for_linear

        monkeypatch.setattr(Dataset, "load", load)
        dataset = Dataset(
            short_name="tas",
        )
        result = regrid(
            self.cube,
            dataset,
            "linear",
            cache_weights=cache_weights,
        )
        expected = np.array([[[1.5]], [[5.5]], [[9.5]]])
        assert_array_equal(result.data, expected)

    @pytest.mark.parametrize(
        "scheme",
        [
            {
                "reference": "esmf_regrid.schemes:regrid_rectilinear_to_rectilinear",
            },
            {
                "reference": "esmvalcore.preprocessor.regrid_schemes:IrisESMFRegrid",
                "method": "bilinear",
            },
        ],
    )
    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_regrid__esmf_rectilinear(self, scheme, cache_weights):
        result = regrid(
            self.cube,
            self.grid_for_linear,
            scheme,
            cache_weights=cache_weights,
        )
        expected = np.array([[[1.5]], [[5.5]], [[9.5]]])
        np.testing.assert_array_almost_equal(result.data, expected, decimal=1)

    def test_regrid__esmf_mesh_to_regular(self):
        result = regrid(self.mesh_cube, self.grid_for_linear, "linear")
        expected = np.array([[[1.5]], [[5.5]], [[9.5]]])
        np.testing.assert_array_almost_equal(result.data, expected, decimal=1)

    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_regrid__regular_coordinates(self, cache_weights):
        data = np.ones((1, 1))
        lons = iris.coords.DimCoord(
            [1.50000000000001],
            standard_name="longitude",
            bounds=[[1, 2]],
            units="degrees_east",
            coord_system=self.cs,
        )
        lats = iris.coords.DimCoord(
            [1.50000000000001],
            standard_name="latitude",
            bounds=[[1, 2]],
            units="degrees_north",
            coord_system=self.cs,
        )
        coords_spec = [(lats, 0), (lons, 1)]
        regular_grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)
        result = regrid(
            regular_grid,
            self.grid_for_linear,
            "linear",
            cache_weights=cache_weights,
        )
        iris.common.resolve.Resolve(result, self.grid_for_linear)

    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_regrid__linear_do_not_preserve_dtype(self, cache_weights):
        self.cube.data = self.cube.data.astype(int)
        result = regrid(
            self.cube,
            self.grid_for_linear,
            "linear",
            cache_weights=cache_weights,
        )
        expected = np.array([[[1.5]], [[5.5]], [[9.5]]])
        assert_array_equal(result.data, expected)
        assert np.issubdtype(self.cube.dtype, np.integer)
        assert np.issubdtype(result.dtype, np.floating)

    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_regrid__linear_with_extrapolation(self, cache_weights):
        data = np.empty((3, 3))
        lons = iris.coords.DimCoord(
            [0, 1.5, 3],
            standard_name="longitude",
            bounds=[[0, 1], [1, 2], [2, 3]],
            units="degrees_east",
            coord_system=self.cs,
        )
        lats = iris.coords.DimCoord(
            [0, 1.5, 3],
            standard_name="latitude",
            bounds=[[0, 1], [1, 2], [2, 3]],
            units="degrees_north",
            coord_system=self.cs,
        )
        coords_spec = [(lats, 0), (lons, 1)]
        grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)
        scheme = {
            "reference": "iris.analysis:Linear",
            "extrapolation_mode": "extrapolate",
        }
        result = regrid(self.cube, grid, scheme, cache_weights=cache_weights)
        expected = [
            [[-3.0, -1.5, 0.0], [0.0, 1.5, 3.0], [3.0, 4.5, 6.0]],
            [[1.0, 2.5, 4.0], [4.0, 5.5, 7.0], [7.0, 8.5, 10.0]],
            [[5.0, 6.5, 8.0], [8.0, 9.5, 11.0], [11.0, 12.5, 14.0]],
        ]
        assert_array_equal(result.data, expected)

    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_regrid__linear_with_mask(self, cache_weights):
        data = np.empty((3, 3))
        grid = iris.cube.Cube(data)
        lons = iris.coords.DimCoord(
            [0, 1.5, 3],
            standard_name="longitude",
            bounds=[[0, 1], [1, 2], [2, 3]],
            units="degrees_east",
            coord_system=self.cs,
        )
        lats = iris.coords.DimCoord(
            [0, 1.5, 3],
            standard_name="latitude",
            bounds=[[0, 1], [1, 2], [2, 3]],
            units="degrees_north",
            coord_system=self.cs,
        )
        coords_spec = [(lats, 0), (lons, 1)]
        grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)
        result = regrid(self.cube, grid, "linear", cache_weights=cache_weights)
        expected = ma.empty((3, 3, 3))
        expected.mask = ma.masked
        expected[:, 1, 1] = np.array([1.5, 5.5, 9.5])
        assert_array_equal(result.data, expected)

    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_regrid__nearest(self, cache_weights):
        data = np.empty((1, 1))
        lons = iris.coords.DimCoord(
            [1.6],
            standard_name="longitude",
            bounds=[[1, 2]],
            units="degrees_east",
            coord_system=self.cs,
        )
        lats = iris.coords.DimCoord(
            [1.6],
            standard_name="latitude",
            bounds=[[1, 2]],
            units="degrees_north",
            coord_system=self.cs,
        )
        coords_spec = [(lats, 0), (lons, 1)]
        grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)
        result = regrid(
            self.cube,
            grid,
            "nearest",
            cache_weights=cache_weights,
        )
        expected = np.array([[[3]], [[7]], [[11]]])
        assert_array_equal(result.data, expected)

    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_regrid__nearest_extrapolate_with_mask(self, cache_weights):
        data = np.empty((3, 3))
        lons = iris.coords.DimCoord(
            [0, 1.6, 3],
            standard_name="longitude",
            bounds=[[0, 1], [1, 2], [2, 3]],
            units="degrees_east",
            coord_system=self.cs,
        )
        lats = iris.coords.DimCoord(
            [0, 1.6, 3],
            standard_name="latitude",
            bounds=[[0, 1], [1, 2], [2, 3]],
            units="degrees_north",
            coord_system=self.cs,
        )
        coords_spec = [(lats, 0), (lons, 1)]
        grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)
        result = regrid(
            self.cube,
            grid,
            "nearest",
            cache_weights=cache_weights,
        )
        expected = ma.empty((3, 3, 3))
        expected.mask = ma.masked
        expected[:, 1, 1] = np.array([3, 7, 11])
        assert_array_equal(result.data, expected)

    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_regrid__area_weighted(self, cache_weights):
        data = np.empty((1, 1))
        lons = iris.coords.DimCoord(
            [1.6],
            standard_name="longitude",
            bounds=[[1, 2]],
            units="degrees_east",
            coord_system=self.cs,
        )
        lats = iris.coords.DimCoord(
            [1.6],
            standard_name="latitude",
            bounds=[[1, 2]],
            units="degrees_north",
            coord_system=self.cs,
        )
        coords_spec = [(lats, 0), (lons, 1)]
        grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)
        result = regrid(
            self.cube,
            grid,
            "area_weighted",
            cache_weights=cache_weights,
        )
        expected = np.array([1.499886, 5.499886, 9.499886])
        np.testing.assert_array_almost_equal(result.data, expected, decimal=6)

    @pytest.mark.parametrize(
        "scheme",
        [
            {"reference": "esmf_regrid.schemes:ESMFAreaWeighted"},
            {
                "reference": "esmvalcore.preprocessor.regrid_schemes:IrisESMFRegrid",
                "method": "conservative",
            },
        ],
    )
    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_regrid__esmf_area_weighted(self, scheme, cache_weights):
        data = np.empty((1, 1))
        lons = iris.coords.DimCoord(
            [1.6],
            standard_name="longitude",
            bounds=[[1, 2]],
            units="degrees_east",
            coord_system=self.cs,
        )
        lats = iris.coords.DimCoord(
            [1.6],
            standard_name="latitude",
            bounds=[[1, 2]],
            units="degrees_north",
            coord_system=self.cs,
        )
        coords_spec = [(lats, 0), (lons, 1)]
        grid = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)
        result = regrid(self.cube, grid, scheme, cache_weights=cache_weights)
        expected = np.array([[[1.499886]], [[5.499886]], [[9.499886]]])
        np.testing.assert_array_almost_equal(result.data, expected, decimal=6)

    @pytest.mark.parametrize("scheme", ["linear", "nearest"])
    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_regrid_unstructured_grid_float(self, cache_weights, scheme):
        """Test regridding with unstructured cube of floats."""
        result = regrid(
            self.unstructured_grid_cube,
            self.tgt_grid_for_unstructured,
            scheme,
            cache_weights=cache_weights,
        )
        assert self.unstructured_grid_cube.dtype == np.float32
        assert result.dtype == np.float32

    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_regrid_nearest_unstructured_grid_int(self, cache_weights):
        """Test nearest-neighbor regridding with unstructured cube of ints."""
        self.unstructured_grid_cube.data = np.ones((3, 4), dtype=int)
        result = regrid(
            self.unstructured_grid_cube,
            self.tgt_grid_for_unstructured,
            "nearest",
            cache_weights=cache_weights,
        )
        assert self.unstructured_grid_cube.dtype == int
        assert result.dtype == int

    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_regrid_linear_unstructured_grid_int(self, cache_weights):
        """Test linear regridding with unstructured cube of ints."""
        self.unstructured_grid_cube.data = np.ones((3, 4), dtype=int)
        result = regrid(
            self.unstructured_grid_cube,
            self.tgt_grid_for_unstructured,
            "linear",
            cache_weights=cache_weights,
        )
        assert self.unstructured_grid_cube.dtype == int
        assert result.dtype == np.float64

    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_invalid_scheme_for_unstructured_grid(self, cache_weights):
        """Test invalid scheme for unstructured cube."""
        msg = (
            "Regridding scheme 'invalid' not available for unstructured data, "
            "expected one of: linear, nearest"
        )
        with pytest.raises(ValueError, match=msg):
            regrid(
                self.unstructured_grid_cube,
                self.tgt_grid_for_unstructured,
                "invalid",
                cache_weights=cache_weights,
            )

    @pytest.mark.parametrize("cache_weights", [True, False])
    def test_invalid_scheme_for_irregular_grid(self, cache_weights):
        """Test invalid scheme for irregular cube."""
        msg = (
            "Regridding scheme 'invalid' not available for irregular data, "
            "expected one of: area_weighted, linear, nearest"
        )
        with pytest.raises(ValueError, match=msg):
            regrid(
                self.irregular_grid,
                self.tgt_grid_for_unstructured,
                "invalid",
                cache_weights=cache_weights,
            )
