"""Test derivation of ``toz``."""

import dask.array as da
import iris
import numpy as np
import pytest

from esmvalcore.preprocessor._derive import toz

from .test_co2s import get_coord_spec, get_ps_cube


def get_masked_o3_cube():
    """Get masked ``o3`` cube."""
    coord_spec = get_coord_spec()
    o3_data = da.ma.masked_less(
        [
            [
                [[0.0, -1.0], [-1.0, -1.0]],
                [[1.0, 2.0], [3.0, -1.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ],
        ],
        0.0,
    )
    return iris.cube.Cube(
        o3_data,
        var_name="o3",
        standard_name="mole_fraction_of_ozone_in_air",
        units="1e-9",
        dim_coords_and_dims=coord_spec,
    )


def get_masked_o3_hybrid_plevs_cube():
    """Get masked ``o3`` cube with hybrid pressure levels."""
    o3_cube = get_masked_o3_cube()
    o3_cube.remove_coord("air_pressure")

    ap_coord = iris.coords.AuxCoord(
        [0.0, 10000.0, 0.0],
        var_name="ap",
        units="Pa",
    )
    b_coord = iris.coords.AuxCoord([0.95, 0.8, 0.7], var_name="b", units="1")
    ps_coord = iris.coords.AuxCoord(
        [[[100000.0, 100000.0], [100000.0, 100000.0]]],
        var_name="ps",
        units="Pa",
    )
    z_coord = iris.coords.DimCoord(
        [0.95, 0.9, 0.7],
        var_name="lev",
        units="1",
        attributes={"positive": "down"},
    )
    o3_cube.add_aux_coord(ap_coord, 1)
    o3_cube.add_aux_coord(b_coord, 1)
    o3_cube.add_aux_coord(ps_coord, (0, 2, 3))
    o3_cube.add_dim_coord(z_coord, 1)

    aux_factory = iris.aux_factory.HybridPressureFactory(
        delta=ap_coord,
        sigma=b_coord,
        surface_air_pressure=ps_coord,
    )
    o3_cube.add_aux_factory(aux_factory)

    return o3_cube


@pytest.fixture
def masked_cubes():
    """Masked O3 cube."""
    o3_cube = get_masked_o3_cube()
    ps_cube = get_ps_cube()
    return iris.cube.CubeList([o3_cube, ps_cube])


@pytest.fixture
def masked_cubes_no_lon():
    """Masked zonal mean O3 cube."""
    o3_cube = get_masked_o3_cube()
    o3_cube = o3_cube.collapsed("longitude", iris.analysis.MEAN)
    o3_cube.remove_coord("longitude")
    ps_cube = get_ps_cube()
    ps_cube.data = [[[101300.0, 101300.0], [101300.0, 101300.0]]]
    return iris.cube.CubeList([o3_cube, ps_cube])


@pytest.fixture
def masked_cubes_hybrid_plevs():
    """Masked zonal mean O3 cube on hybrid levels."""
    o3_cube = get_masked_o3_hybrid_plevs_cube()
    ps_cube = get_ps_cube()
    ps_cube.data = [[[101300.0, 101300.0], [101300.0, 101300.0]]]
    return iris.cube.CubeList([o3_cube, ps_cube])


@pytest.fixture
def unmasked_cubes():
    """Unmasked O3 cube."""
    coord_spec = get_coord_spec()
    o3_data = da.array(
        [
            [
                [[2.0, 1.0], [0.8, 1.0]],
                [[1.5, 0.8], [2.0, 3.0]],
                [[4.0, 1.0], [3.0, 2.0]],
            ],
        ],
    )
    o3_cube = iris.cube.Cube(
        o3_data,
        var_name="o3",
        standard_name="mole_fraction_of_ozone_in_air",
        units="1e-9",
        dim_coords_and_dims=coord_spec,
    )
    ps_cube = get_ps_cube()
    return iris.cube.CubeList([o3_cube, ps_cube])


def test_toz_calculate_masked_cubes(masked_cubes):
    """Test function ``calculate`` with masked cube."""
    derived_var = toz.DerivedVariable()

    out_cube = derived_var.calculate(masked_cubes)

    assert out_cube.units == "m"
    assert not np.ma.is_masked(out_cube.data)
    np.testing.assert_allclose(
        out_cube.data,
        [
            [
                [1.2988646378902597e-5, 0.7871906896304607e-5],
                [1.6924599827054907e-5, 0.9446288275565529e-5],
            ],
        ],
    )


def test_toz_calculate_masked_cubes_no_lon(masked_cubes_no_lon):
    """Test function ``calculate`` with zonal mean masked cube."""
    derived_var = toz.DerivedVariable()

    out_cube = derived_var.calculate(masked_cubes_no_lon)

    assert out_cube.units == "m"
    assert not np.ma.is_masked(out_cube.data)
    np.testing.assert_allclose(
        out_cube.data,
        [[[1.3972634740940675e-5], [1.6924599827054907e-5]]],
    )


def test_toz_calculate_masked_cubes_hybrid_plevs(masked_cubes_hybrid_plevs):
    """Test function ``calculate`` with zonal mean masked cube."""
    derived_var = toz.DerivedVariable()

    out_cube = derived_var.calculate(masked_cubes_hybrid_plevs)

    assert out_cube.units == "m"
    assert not np.ma.is_masked(out_cube.data)
    np.testing.assert_allclose(
        out_cube.data,
        [
            [
                [0.33701601399804104e-5, 0.3739155775744688e-5],
                [0.440334792012039e-5, 0.19679767240761517e-5],
            ],
        ],
    )


def test_toz_calculate_unmasked_cubes(unmasked_cubes):
    """Test function ``calculate`` with unmasked cube."""
    derived_var = toz.DerivedVariable()

    out_cube = derived_var.calculate(unmasked_cubes)

    assert out_cube.units == "m"
    assert not np.ma.is_masked(out_cube.data)
    np.testing.assert_allclose(
        out_cube.data,
        [[[2.65676858e-5, 0.39359534e-5], [2.04669579e-5, 0.94462883e-5]]],
    )


@pytest.mark.parametrize(
    ("project", "out"),
    [
        ("CMIP5", [{"short_name": "tro3"}, {"short_name": "ps"}]),
        ("TEST", [{"short_name": "tro3"}, {"short_name": "ps"}]),
        ("CMIP6", [{"short_name": "o3"}, {"short_name": "ps", "mip": "Amon"}]),
    ],
)
def test_toz_required(project, out):
    """Test function ``required``."""
    derived_var = toz.DerivedVariable()
    output = derived_var.required(project)
    assert output == out
