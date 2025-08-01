"""Unit tests for the :func:`esmvalcore.preprocessor._other` module."""

import unittest

import dask.array as da
import iris.coords
import numpy as np
import pytest
from cf_units import Unit
from iris.aux_factory import AtmosphereSigmaFactory
from iris.coords import (
    AncillaryVariable,
    AuxCoord,
    CellMeasure,
    CellMethod,
    DimCoord,
)
from iris.cube import Cube
from iris.exceptions import CoordinateMultiDimError
from numpy.testing import assert_array_equal

from esmvalcore.preprocessor import (
    align_metadata,
    clip,
    cumulative_sum,
    histogram,
)
from tests.unit.preprocessor._compare_with_refs.test_compare_with_refs import (
    get_3d_cube,
)


def test_align_metadata():
    cube = Cube(0.0, units="celsius")

    new_cube = align_metadata(cube, "CMIP6", "Amon", "tas")

    assert new_cube is not cube

    assert new_cube.long_name == "Near-Surface Air Temperature"
    assert new_cube.standard_name == "air_temperature"
    assert new_cube.units == "K"
    assert new_cube.var_name == "tas"
    np.testing.assert_allclose(new_cube.data, 273.15)

    assert cube.long_name is None
    assert cube.standard_name is None
    assert cube.units == "celsius"
    assert cube.var_name is None
    np.testing.assert_allclose(cube.data, 0.0)


@pytest.mark.parametrize("short_name", ["sic", "siconc"])
def test_align_metadata_alt_names(short_name):
    cube = Cube(0.0, units="%")

    new_cube = align_metadata(cube, "CMIP6", "SImon", short_name)

    assert new_cube is not cube

    assert new_cube.long_name == "Sea-Ice Area Percentage (Ocean Grid)"
    assert new_cube.standard_name == "sea_ice_area_fraction"
    assert new_cube.units == "%"
    assert new_cube.var_name == "siconc"
    np.testing.assert_allclose(new_cube.data, 0.0)

    assert cube.long_name is None
    assert cube.standard_name is None
    assert cube.units == "%"
    assert cube.var_name is None
    np.testing.assert_allclose(cube.data, 0.0)


def test_align_metadata_invalid_project():
    cube = Cube(0.0, units="1")
    msg = "No CMOR tables available for project 'ZZZ'"
    with pytest.raises(KeyError, match=msg):
        align_metadata(cube, "ZZZ", "Amon", "tas")


def test_align_metadata_invalid_short_name_strict():
    cube = Cube(0.0, units="1")
    msg = "Variable 'zzz' not available for table 'Amon' of project 'CMIP6'"
    with pytest.raises(ValueError, match=msg):
        align_metadata(cube, "CMIP6", "Amon", "zzz")


def test_align_metadata_invalid_short_name_not_strict():
    cube = Cube(0.0, units="1")
    new_cube = align_metadata(cube, "CMIP6", "Amon", "zzz", strict=False)
    assert new_cube is not cube
    assert new_cube == cube
    assert new_cube.long_name is None
    assert new_cube.standard_name is None
    assert new_cube.units == "1"
    assert new_cube.var_name is None
    np.testing.assert_allclose(cube.data, 0.0)


class TestOther(unittest.TestCase):
    """Test class for _other."""

    def test_clip(self):
        """Test clip function."""
        cube = Cube(np.array([-10, 0, 10]))
        cube.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(3),
                standard_name="time",
                units=Unit(
                    "days since 1950-01-01 00:00:00",
                    calendar="gregorian",
                ),
            ),
            0,
        )
        # Cube needs to be copied, since it is modified in-place and test cube
        # should not change.
        assert_array_equal(
            clip(cube.copy(), 0, None).data,
            np.array([0, 0, 10]),
        )
        assert_array_equal(
            clip(cube.copy(), None, 0).data,
            np.array([-10, 0, 0]),
        )
        assert_array_equal(clip(cube.copy(), -1, 2).data, np.array([-1, 0, 2]))
        # Masked cube TODO
        # No parameters specified
        with self.assertRaises(ValueError):
            clip(cube, None, None)
        # Maximum lower than minimum
        with self.assertRaises(ValueError):
            clip(cube, 10, 8)


@pytest.fixture
def cube():
    """Regular cube."""
    cube_data = np.ma.masked_inside(
        np.arange(8.0, dtype=np.float32).reshape(2, 2, 2),
        1.5,
        3.5,
    )
    cube_data = np.swapaxes(cube_data, 0, -1)
    return get_3d_cube(
        cube_data,
        standard_name="air_temperature",
        var_name="tas",
        units="K",
    )


def assert_metadata(cube, normalization=None):
    """Assert correct metadata."""
    assert cube.standard_name is None
    if normalization == "sum":
        assert cube.long_name == "Relative Frequency"
        assert cube.var_name == "relative_frequency_tas"
        assert cube.units == "1"
    elif normalization == "integral":
        assert cube.long_name == "Density"
        assert cube.var_name == "density_tas"
        assert cube.units == "K-1"
    else:
        assert cube.long_name == "Frequency"
        assert cube.var_name == "frequency_tas"
        assert cube.units == "1"
    assert cube.attributes == {}
    assert cube.cell_methods == ()
    assert cube.coords("air_temperature")
    bin_coord = cube.coord("air_temperature")
    assert bin_coord.standard_name == "air_temperature"
    assert bin_coord.var_name == "tas"
    assert bin_coord.long_name is None
    assert bin_coord.units == "K"
    assert bin_coord.attributes == {}


@pytest.mark.parametrize("lazy", [False, True])
def test_histogram_defaults(cube, lazy):
    """Test `histogram`."""
    if lazy:
        cube.data = cube.lazy_data()
    input_cube = cube.copy()

    result = histogram(input_cube)

    assert input_cube == cube
    assert_metadata(result)
    assert result.shape == (10,)
    if lazy:
        assert result.has_lazy_data()
    else:
        assert not result.has_lazy_data()
    assert result.dtype == np.float32
    np.testing.assert_allclose(
        result.data,
        [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(result.data.mask, [False] * 10)
    bin_coord = result.coord("air_temperature")
    assert bin_coord.shape == (10,)
    assert bin_coord.dtype == np.float64
    assert bin_coord.bounds_dtype == np.float64
    np.testing.assert_allclose(
        bin_coord.points,
        [0.35, 1.05, 1.75, 2.45, 3.15, 3.85, 4.55, 5.25, 5.95, 6.65],
    )
    np.testing.assert_allclose(
        bin_coord.bounds,
        [
            [0.0, 0.7],
            [0.7, 1.4],
            [1.4, 2.1],
            [2.1, 2.8],
            [2.8, 3.5],
            [3.5, 4.2],
            [4.2, 4.9],
            [4.9, 5.6],
            [5.6, 6.3],
            [6.3, 7.0],
        ],
    )


@pytest.mark.parametrize("normalization", [None, "sum", "integral"])
@pytest.mark.parametrize("weights", [False, None])
@pytest.mark.parametrize("lazy", [False, True])
def test_histogram_over_time(cube, lazy, weights, normalization):
    """Test `histogram`."""
    if lazy:
        cube.data = cube.lazy_data()
    input_cube = cube.copy()

    result = histogram(
        input_cube,
        coords=["time"],
        bins=[4.5, 6.5, 8.5, 10.5],
        bin_range=(4.5, 10.5),
        weights=weights,
        normalization=normalization,
    )

    assert input_cube == cube
    assert_metadata(result, normalization=normalization)
    assert result.coord("latitude") == input_cube.coord("latitude")
    assert result.coord("longitude") == input_cube.coord("longitude")
    assert result.shape == (2, 2, 3)
    if lazy:
        assert result.has_lazy_data()
    else:
        assert not result.has_lazy_data()
    assert result.dtype == np.float32
    if normalization == "integral":
        expected_data = np.ma.masked_invalid(
            [
                [[np.nan, np.nan, np.nan], [0.5, 0.0, 0.0]],
                [[np.nan, np.nan, np.nan], [0.25, 0.25, 0.0]],
            ],
        )
    elif normalization == "sum":
        expected_data = np.ma.masked_invalid(
            [
                [[np.nan, np.nan, np.nan], [1.0, 0.0, 0.0]],
                [[np.nan, np.nan, np.nan], [0.5, 0.5, 0.0]],
            ],
        )
    else:
        expected_data = np.ma.masked_invalid(
            [
                [[np.nan, np.nan, np.nan], [1.0, 0.0, 0.0]],
                [[np.nan, np.nan, np.nan], [1.0, 1.0, 0.0]],
            ],
        )
    np.testing.assert_allclose(result.data, expected_data)
    np.testing.assert_allclose(result.data.mask, expected_data.mask)
    bin_coord = result.coord("air_temperature")
    assert bin_coord.shape == (3,)
    assert bin_coord.dtype == np.float64
    assert bin_coord.bounds_dtype == np.float64
    np.testing.assert_allclose(bin_coord.points, [5.5, 7.5, 9.5])
    np.testing.assert_allclose(
        bin_coord.bounds,
        [[4.5, 6.5], [6.5, 8.5], [8.5, 10.5]],
    )


@pytest.mark.parametrize("normalization", [None, "sum", "integral"])
@pytest.mark.parametrize("lazy", [False, True])
def test_histogram_fully_masked(cube, lazy, normalization):
    """Test `histogram`."""
    cube.data = np.ma.masked_all((2, 2, 2), dtype=np.float32)
    if lazy:
        cube.data = cube.lazy_data()

    result = histogram(cube, bin_range=(0, 10), normalization=normalization)

    assert_metadata(result, normalization=normalization)
    assert result.shape == (10,)
    if lazy:
        assert result.has_lazy_data()
    else:
        assert not result.has_lazy_data()
    assert result.dtype == np.float32
    np.testing.assert_allclose(
        result.data,
        np.ma.masked_all(
            10,
        ),
    )
    np.testing.assert_equal(result.data.mask, [True] * 10)
    bin_coord = result.coord("air_temperature")
    assert bin_coord.shape == (10,)
    assert bin_coord.dtype == np.float64
    assert bin_coord.bounds_dtype == np.float64
    np.testing.assert_allclose(
        bin_coord.points,
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
    )
    np.testing.assert_allclose(
        bin_coord.bounds,
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
        ],
    )


@pytest.mark.parametrize("normalization", [None, "sum", "integral"])
@pytest.mark.parametrize(
    "weights",
    [
        True,
        np.array([[[6, 6], [6, 6]], [[2, 2], [2, 2]]]),
        da.array([[[6, 6], [6, 6]], [[2, 2], [2, 2]]]),
    ],
)
@pytest.mark.parametrize("lazy", [False, True])
def test_histogram_weights(cube, lazy, weights, normalization):
    """Test `histogram`."""
    if lazy:
        cube.data = cube.lazy_data()
    input_cube = cube.copy()

    result = histogram(
        input_cube,
        coords=["time", "longitude"],
        bins=[0.0, 2.0, 4.0, 8.0],
        weights=weights,
        normalization=normalization,
    )

    assert input_cube == cube
    assert_metadata(result, normalization=normalization)
    assert result.coord("latitude") == input_cube.coord("latitude")
    assert result.shape == (2, 3)
    if lazy:
        assert result.has_lazy_data()
    else:
        assert not result.has_lazy_data()
    assert result.dtype == np.float32
    if normalization == "integral":
        expected_data = np.ma.masked_invalid(
            [[0.25, 0.0, 0.125], [0.0, 0.0, 0.25]],
        )
    elif normalization == "sum":
        expected_data = np.ma.masked_invalid(
            [[0.5, 0.0, 0.5], [0.0, 0.0, 1.0]],
        )
    else:
        expected_data = np.ma.masked_invalid(
            [[8.0, 0.0, 8.0], [0.0, 0.0, 8.0]],
        )
    np.testing.assert_allclose(result.data, expected_data)
    np.testing.assert_allclose(result.data.mask, expected_data.mask)
    bin_coord = result.coord("air_temperature")
    assert bin_coord.shape == (3,)
    assert bin_coord.dtype == np.float64
    assert bin_coord.bounds_dtype == np.float64
    np.testing.assert_allclose(bin_coord.points, [1.0, 3.0, 6.0])
    np.testing.assert_allclose(
        bin_coord.bounds,
        [[0.0, 2.0], [2.0, 4.0], [4.0, 8.0]],
    )


@pytest.fixture
def cube_with_rich_metadata():
    """Cube with rich metadata."""
    time = DimCoord([0], bounds=[[-1, 1]], var_name="time", units="s")
    sigma = DimCoord([0], var_name="sigma", units="1")
    lat = DimCoord([0], var_name="lat", units="degrees")
    lon = DimCoord([0], var_name="lon", units="degrees")
    ptop = AuxCoord(0, var_name="ptop", units="Pa")
    psur = AuxCoord([[0]], var_name="ps", units="Pa")
    sigma_factory = AtmosphereSigmaFactory(ptop, sigma, psur)
    cell_area = CellMeasure([[1]], var_name="area", units="m2", measure="area")
    ancillary = AncillaryVariable([0], var_name="ancillary")
    return Cube(
        np.ones((1, 1, 1, 1), dtype=np.float32),
        standard_name=None,
        long_name="Air Temperature",
        var_name=None,
        units="K",
        attributes={"test": "1"},
        cell_methods=(CellMethod("point", "sigma"),),
        dim_coords_and_dims=[(time, 0), (sigma, 1), (lat, 2), (lon, 3)],
        aux_coords_and_dims=[(ptop, ()), (psur, (2, 3))],
        aux_factories=[sigma_factory],
        ancillary_variables_and_dims=[(ancillary, 1)],
        cell_measures_and_dims=[(cell_area, (2, 3))],
    )


@pytest.mark.parametrize("normalization", [None, "sum", "integral"])
@pytest.mark.parametrize("weights", [True, False, None])
@pytest.mark.parametrize("lazy", [False, True])
def test_histogram_metadata(
    cube_with_rich_metadata,
    lazy,
    weights,
    normalization,
):
    """Test `histogram`."""
    if lazy:
        cube_with_rich_metadata.data = cube_with_rich_metadata.lazy_data()
    input_cube = cube_with_rich_metadata.copy()

    result = histogram(
        input_cube,
        coords=["time"],
        bins=[0.0, 1.0, 2.0],
        bin_range=(0.0, 2.0),
        weights=weights,
        normalization=normalization,
    )

    assert input_cube == cube_with_rich_metadata
    assert result.shape == (1, 1, 1, 2)

    assert result.standard_name is None
    if normalization == "sum":
        assert result.long_name == "Relative Frequency of Air Temperature"
        assert result.var_name == "relative_frequency"
        assert result.units == "1"
    elif normalization == "integral":
        assert result.long_name == "Density of Air Temperature"
        assert result.var_name == "density"
        assert result.units == "K-1"
    else:
        assert result.long_name == "Frequency of Air Temperature"
        assert result.var_name == "frequency"
        assert result.units == "1"
    assert result.attributes == {"test": "1"}
    assert result.cell_methods == (CellMethod("point", "sigma"),)

    assert not result.coords("time", dim_coords=True)
    for dim_coord in ("sigma", "lat", "lon"):
        assert result.coord(dim_coord, dim_coords=True) == input_cube.coord(
            dim_coord,
            dim_coords=True,
        )
        assert result.coord_dims(dim_coord) == (
            input_cube.coord_dims(dim_coord)[0] - 1,
        )
    assert result.coords("Air Temperature", dim_coords=True)
    bin_coord = result.coord("Air Temperature")
    assert result.coord_dims(bin_coord) == (3,)
    assert bin_coord.standard_name is None
    assert bin_coord.long_name == "Air Temperature"
    assert bin_coord.var_name is None
    assert bin_coord.units == "K"
    assert bin_coord.attributes == {}

    assert result.coords("time", dim_coords=False)
    assert result.coord_dims("time") == ()
    assert result.coord("ptop") == input_cube.coord("ptop")
    assert result.coord("ps") == input_cube.coord("ps")
    assert len(result.aux_factories) == 1
    assert isinstance(result.aux_factories[0], AtmosphereSigmaFactory)
    assert result.ancillary_variables() == input_cube.ancillary_variables()
    assert result.cell_measures() == input_cube.cell_measures()


@pytest.mark.parametrize("lazy", [False, True])
def test_histogram_fully_masked_no_bin_range(cube, lazy):
    """Test `histogram`."""
    cube.data = np.ma.masked_all((2, 2, 2), dtype=np.float32)
    if lazy:
        cube.data = cube.lazy_data()

    msg = (
        r"Cannot calculate histogram for bin_range=\(masked, masked\) \(or "
        r"for fully masked data when `bin_range` is not given\)"
    )
    with pytest.raises(ValueError, match=msg):
        histogram(cube)


def test_histogram_invalid_bins(cube):
    """Test `histogram`."""
    msg = (
        r"bins cannot be a str \(got 'auto'\), must be int or Sequence of int"
    )
    with pytest.raises(TypeError, match=msg):
        histogram(cube, bins="auto")


def test_histogram_invalid_normalization(cube):
    """Test `histogram`."""
    msg = (
        r"Expected one of \(None, 'sum', 'integral'\) for normalization, got "
        r"'invalid'"
    )
    with pytest.raises(ValueError, match=msg):
        histogram(cube, normalization="invalid")


@pytest.mark.parametrize("lazy", [True, False])
def test_cumulative_sum_time(cube, lazy):
    """Test `cumulative_sum`."""
    if lazy:
        cube.data = cube.lazy_data()

    result = cumulative_sum(cube, "time")

    assert result is not cube
    assert result.standard_name == "air_temperature"
    assert result.var_name == "cumulative_tas"
    assert result.long_name is None
    assert result.units == "K"
    assert result.shape == cube.shape
    assert result.dtype == cube.dtype
    assert result.has_lazy_data() is lazy
    expected_data = np.ma.masked_invalid(
        [[[0.0, 4.0], [np.nan, 6.0]], [[1.0, 9.0], [np.nan, 13.0]]],
    )
    np.testing.assert_allclose(result.data, expected_data)
    np.testing.assert_allclose(result.data.mask, expected_data.mask)


@pytest.mark.parametrize("lazy", [True, False])
def test_cumulative_sum_time_weighted(cube, lazy):
    """Test `cumulative_sum`."""
    cube.var_name = None
    cube.long_name = "Air Temperature"
    if lazy:
        cube.data = cube.lazy_data()

    result = cumulative_sum(cube, "time", weights=True)

    assert result is not cube
    assert result.standard_name is None
    assert result.var_name is None
    assert result.long_name == "Cumulative Air Temperature"
    assert result.units == "K.d"
    assert result.shape == cube.shape
    assert result.dtype == cube.dtype
    assert result.has_lazy_data() is lazy
    expected_data = np.ma.masked_invalid(
        [[[0.0, 24.0], [np.nan, 36.0]], [[2.0, 34.0], [np.nan, 50.0]]],
    )
    np.testing.assert_allclose(result.data, expected_data)
    np.testing.assert_allclose(result.data.mask, expected_data.mask)


@pytest.mark.parametrize("weights", [False, None])
@pytest.mark.parametrize("lazy", [True, False])
def test_cumulative_sum_latitude(cube, lazy, weights):
    """Test `cumulative_sum`."""
    if lazy:
        cube.data = cube.lazy_data()

    result = cumulative_sum(cube, "latitude", weights=weights)

    assert result is not cube
    assert result.standard_name == "air_temperature"
    assert result.var_name == "cumulative_tas"
    assert result.long_name is None
    assert result.units == "K"
    assert result.shape == cube.shape
    assert result.dtype == cube.dtype
    assert result.has_lazy_data() is lazy
    expected_data = np.ma.masked_invalid(
        [[[0.0, 4.0], [np.nan, 10.0]], [[1.0, 5.0], [np.nan, 12.0]]],
    )
    np.testing.assert_allclose(result.data, expected_data)
    np.testing.assert_allclose(result.data.mask, expected_data.mask)


@pytest.mark.parametrize("lazy", [True, False])
def test_cumulative_sum_latitude_weighted(cube, lazy):
    """Test `cumulative_sum`."""
    cube.var_name = None
    cube.long_name = "Air Temperature"
    if lazy:
        cube.data = cube.lazy_data()

    result = cumulative_sum(cube, "latitude", weights=cube.core_data())

    assert result is not cube
    assert result.standard_name is None
    assert result.var_name is None
    assert result.long_name == "Cumulative Air Temperature"
    assert result.units == "K.degrees_north"
    assert result.shape == cube.shape
    assert result.dtype == cube.dtype
    assert result.has_lazy_data() is lazy
    expected_data = np.ma.masked_invalid(
        [[[0.0, 16.0], [np.nan, 52.0]], [[1.0, 25.0], [np.nan, 74.0]]],
    )
    np.testing.assert_allclose(result.data, expected_data)
    np.testing.assert_allclose(result.data.mask, expected_data.mask)


@pytest.mark.parametrize("lazy", [True, False])
def test_cumulative_sum_scalar_longitude(cube, lazy):
    """Test `cumulative_sum`."""
    cube = cube.collapsed("longitude", iris.analysis.SUM)
    if lazy:
        cube.data = cube.lazy_data()

    result = cumulative_sum(cube, "longitude")

    assert result is not cube
    assert result.standard_name == "air_temperature"
    assert result.var_name == "cumulative_tas"
    assert result.long_name is None
    assert result.units == "K"
    assert result.shape == cube.shape
    assert result.dtype == cube.dtype
    assert result.has_lazy_data() is lazy
    np.testing.assert_allclose(result.data, [[4.0, 6.0], [6.0, 7.0]])


@pytest.mark.parametrize("lazy", [True, False])
def test_cumulative_sum_scalar_longitude_weighted(cube, lazy):
    """Test `cumulative_sum`."""
    cube = cube.collapsed("longitude", iris.analysis.SUM)
    if lazy:
        cube.data = cube.lazy_data()

    result = cumulative_sum(cube, "longitude", weights=True)

    assert result is not cube
    assert result.standard_name is None
    assert result.var_name == "cumulative_tas"
    assert result.long_name is None
    assert result.units == "K.degrees_east"
    assert result.shape == cube.shape
    assert result.dtype == cube.dtype
    assert result.has_lazy_data() is lazy
    np.testing.assert_allclose(result.data, [[40.0, 60.0], [60.0, 70.0]])


def test_cumulative_sum_invalid_coordinate(cube):
    """Test `cumulative_sum`."""
    aux_coord = AuxCoord(np.ones((2, 2)), var_name="aux_2d")
    cube.add_aux_coord(aux_coord, (0, 1))
    msg = r"Multi-dimensional coordinate not supported: 'aux_2d'"

    with pytest.raises(CoordinateMultiDimError, match=msg):
        cumulative_sum(cube, coord="aux_2d")

    with pytest.raises(CoordinateMultiDimError, match=msg):
        cumulative_sum(cube, coord=aux_coord)


if __name__ == "__main__":
    unittest.main()
