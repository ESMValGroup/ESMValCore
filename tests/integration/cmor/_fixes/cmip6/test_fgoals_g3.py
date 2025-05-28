"""Tests for the fixes of FGOALS-g3."""

from unittest import mock

import iris
import numpy as np
import pandas as pd
import pytest

from esmvalcore.cmor._fixes.cmip6.fgoals_g3 import Mrsos, Siconc, Tas, Tos
from esmvalcore.cmor._fixes.common import OceanFixGrid
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


def test_get_tos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "BCC-CSM2-MR", "Omon", "tos")
    assert fix == [Tos(None), GenericFix(None)]


def test_tos_fix():
    """Test fix for ``tos``."""
    assert issubclass(Tos, OceanFixGrid)


@mock.patch(
    "esmvalcore.cmor._fixes.cmip6.fgoals_g3.OceanFixGrid.fix_metadata",
    autospec=True,
)
def test_tos_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``tos``."""
    mock_base_fix_metadata.side_effect = lambda x, y: y

    # Create test cube
    lat_coord = iris.coords.AuxCoord(
        [3.14, 1200.0, 6.28],
        var_name="lat",
        standard_name="latitude",
    )
    lon_coord = iris.coords.AuxCoord(
        [1.0, 2.0, 1e30],
        var_name="lon",
        standard_name="longitude",
    )
    cube = iris.cube.Cube(
        [1.0, 2.0, 3.0],
        var_name="tos",
        standard_name="sea_surface_temperature",
        aux_coords_and_dims=[(lat_coord, 0), (lon_coord, 0)],
    )
    cubes = iris.cube.CubeList([cube])

    # Apply fix
    vardef = get_var_info("CMIP6", "Omon", "tos")
    fix = Tos(vardef)
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes[0]
    np.testing.assert_allclose(
        fixed_cube.coord("latitude").points,
        [3.14, 0.0, 6.28],
    )
    np.testing.assert_allclose(
        fixed_cube.coord("longitude").points,
        [1.0, 2.0, 0.0],
    )
    mock_base_fix_metadata.assert_called_once_with(fix, cubes)


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "BCC-CSM2-MR", "SImon", "siconc")
    assert fix == [Siconc(None), GenericFix(None)]


def test_siconc_fix():
    """Test fix for ``siconc``."""
    assert Siconc is Tos


def test_get_mrsos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("CMIP6", "FGOALS-g3", "Lmon", "mrsos")
    assert fix == [Mrsos(None), GenericFix(None)]


def test_mrsos_fix():
    """Test fix for ``mrsos``."""
    assert issubclass(Mrsos, Fix)


@mock.patch(
    "esmvalcore.cmor._fixes.cmip6.fgoals_g3.Fix.fix_metadata",
    autospec=True,
)
def test_mrsos_fix_metadata(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``mrsos``."""
    mock_base_fix_metadata.side_effect = lambda x, y: y

    # Create test cube
    lat_coord = iris.coords.AuxCoord(
        [1.0, 2.0, 3.0],
        var_name="lat",
        standard_name="latitude",
    )
    lat_coord.bounds = [[0.5, 1.5], [-0.5, 0.5], [2.5, 3.5]]
    lon_coord = iris.coords.AuxCoord(
        [1.0, 2.0, 3.0],
        var_name="lon",
        standard_name="longitude",
    )
    lon_coord.bounds = [[0.5, 1.5], [-0.5, 0.5], [2.5, 3.5]]
    cube = iris.cube.Cube(
        [1.0, 2.0, 3.0],
        var_name="mrsos",
        standard_name="mass_content_of_water_in_soil_layer",
        aux_coords_and_dims=[(lat_coord, 0), (lon_coord, 0)],
    )
    cubes = iris.cube.CubeList([cube])

    # Apply fix
    vardef = get_var_info("CMIP6", "Lmon", "mrsos")
    fix = Mrsos(vardef)
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes[0]
    np.testing.assert_allclose(
        fixed_cube.coord("latitude").bounds,
        [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]],
    )
    np.testing.assert_allclose(
        fixed_cube.coord("longitude").bounds,
        [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]],
    )
    mock_base_fix_metadata.assert_called_once_with(fix, cubes)


@mock.patch(
    "esmvalcore.cmor._fixes.cmip6.fgoals_g3.Fix.fix_metadata",
    autospec=True,
)
def test_mrsos_fix_metadata_2(mock_base_fix_metadata):
    """Test ``fix_metadata`` for ``mrsos`` if no fix is necessary."""
    mock_base_fix_metadata.side_effect = lambda x, y: y

    # Create test cube
    lat_coord = iris.coords.AuxCoord(
        [1.0, 2.0, 3.0],
        var_name="lat",
        standard_name="latitude",
    )
    lat_coord.bounds = [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]]
    lon_coord = iris.coords.AuxCoord(
        [1.0, 2.0, 3.0],
        var_name="lon",
        standard_name="longitude",
    )
    lon_coord.bounds = [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]]
    cube = iris.cube.Cube(
        [1.0, 2.0, 3.0],
        var_name="mrsos",
        standard_name="mass_content_of_water_in_soil_layer",
        aux_coords_and_dims=[(lat_coord, 0), (lon_coord, 0)],
    )
    cubes = iris.cube.CubeList([cube])

    # Apply fix
    vardef = get_var_info("CMIP6", "Lmon", "mrsos")
    fix = Mrsos(vardef)
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes[0]
    np.testing.assert_allclose(
        fixed_cube.coord("latitude").bounds,
        [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]],
    )
    np.testing.assert_allclose(
        fixed_cube.coord("longitude").bounds,
        [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]],
    )
    mock_base_fix_metadata.assert_called_once_with(fix, cubes)


@pytest.fixture
def tas_cubes():
    correct_time_coord = iris.coords.DimCoord(
        points=[1.0, 2.0, 3.0, 4.0, 5.0],
        var_name="time",
        standard_name="time",
        units="days since 1850-01-01",
    )

    lat_coord = iris.coords.DimCoord(
        [0.0],
        var_name="lat",
        standard_name="latitude",
    )

    lon_coord = iris.coords.DimCoord(
        [0.0],
        var_name="lon",
        standard_name="longitude",
    )

    correct_coord_specs = [
        (correct_time_coord, 0),
        (lat_coord, 1),
        (lon_coord, 2),
    ]

    correct_tas_cube = iris.cube.Cube(
        np.ones((5, 1, 1)),
        var_name="tas",
        units="K",
        dim_coords_and_dims=correct_coord_specs,
    )

    scalar_cube = iris.cube.Cube(0.0, var_name="ps")

    return iris.cube.CubeList([correct_tas_cube, scalar_cube])


def test_get_tas_fix():
    """Test tas fix."""
    fix = Fix.get_fixes("CMIP6", "FGOALS-g3", "day", "tas")
    assert fix == [Tas(None), GenericFix(None)]


def test_tas_fix_metadata(tas_cubes):
    """Test metadata fix."""
    vardef = get_var_info("CMIP6", "day", "tas")
    fix = Tas(vardef)

    out_cubes = fix.fix_metadata(tas_cubes)
    assert out_cubes[0].var_name == "tas"
    coord = out_cubes[0].coord("time")
    assert pd.Series(coord.points).is_monotonic_increasing

    # de-monotonize time points
    for cube in tas_cubes:
        if cube.var_name == "tas":
            time = cube.coord("time")
            points = np.array(time.points)
            points[-1] = points[0]
            dims = cube.coord_dims(time)
            cube.remove_coord(time)
            time = iris.coords.AuxCoord.from_coord(time)
            cube.add_aux_coord(time.copy(points), dims)

    out_cubes = fix.fix_metadata(tas_cubes)
    for cube in out_cubes:
        if cube.var_name == "tas":
            assert cube.coord("time").is_monotonic()
