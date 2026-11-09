from __future__ import annotations

from typing import TYPE_CHECKING

import iris.cube
import numpy as np
import pytest
from iris.aux_factory import HybridPressureFactory
from iris.coords import AuxCoord, DimCoord

from esmvalcore.io.local import LocalFile, _get_attr_from_field_coord

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


@pytest.fixture
def local_file(tmp_path: Path) -> LocalFile:
    cube = iris.cube.Cube([0])
    cube.attributes.globals["attribute"] = "value"
    file = tmp_path / "test.nc"
    iris.save(cube, file)
    return LocalFile(file)


def test_to_iris(local_file: LocalFile) -> None:
    cubes = local_file.to_iris()
    assert len(cubes) == 1


def test_attributes(local_file: LocalFile) -> None:
    local_file.to_iris()  # Load the file to populate attributes
    attrs = local_file.attributes
    assert attrs["attribute"] == "value"


def test_attributes_without_loading(local_file: LocalFile) -> None:
    """Test that accessing attributes without loading the file first raises."""
    with pytest.raises(
        ValueError,
        match=r"Attributes have not been read yet.*",
    ):
        local_file.attributes  # noqa: B018


def test_get_attr_from_field_coord_none(mocker: MockerFixture) -> None:
    """Test ``_get_attr_from_field_coord``."""
    attr = _get_attr_from_field_coord(mocker.sentinel.ncfield, None, "attr")
    assert attr is None


def test_to_iris_removes_formula_term_cubes(tmp_path: Path) -> None:
    """Test that cubes only needed for derived coordinates are removed."""
    lev = DimCoord(
        [0.5, 0.9],
        var_name="lev",
        standard_name="atmosphere_hybrid_sigma_pressure_coordinate",
        units="1",
    )
    ap = AuxCoord([1.0, 2.0], var_name="ap", units="Pa")
    b = AuxCoord([0.0, 1.0], var_name="b", units="1")
    ps = AuxCoord(
        np.ones((2, 2)),
        var_name="ps",
        standard_name="surface_air_pressure",
        units="Pa",
    )
    cube = iris.cube.Cube(
        np.zeros((2, 2, 2)),
        var_name="cli",
        dim_coords_and_dims=[(lev, 0)],
        aux_coords_and_dims=[(ap, 0), (b, 0), (ps, (1, 2))],
    )
    cube.add_aux_factory(
        HybridPressureFactory(delta=ap, sigma=b, surface_air_pressure=ps),
    )
    file = tmp_path / "test.nc"
    iris.save(cube, file)
    assert len(iris.load_raw(file)) == 2

    cubes = LocalFile(file).to_iris()

    assert len(cubes) == 1
    assert cubes[0].var_name == "cli"
    assert cubes[0].coords("air_pressure")
