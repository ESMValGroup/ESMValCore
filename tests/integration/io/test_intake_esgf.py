"""Integration tests for :mod:`esmvalcore.io.intake_esgf`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import intake_esgf
import intake_esgf.config
import iris.coords
import iris.cube
import numpy as np
import pytest
import xarray as xr

from esmvalcore.io.intake_esgf import IntakeESGFDataset
from esmvalcore.io.local import LocalFile

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


@pytest.fixture
def scalar_coord_file(tmp_path: Path) -> Path:
    """Create a file with a dimensionless coordinate variable."""
    path = tmp_path / "tas.nc"
    time = iris.coords.DimCoord(
        [0.0, 1.0],
        standard_name="time",
        var_name="time",
        units="days since 1850-01-01",
    )
    height = iris.coords.AuxCoord(
        2.0,
        standard_name="height",
        var_name="height",
        units="m",
        attributes={"positive": "up"},
    )
    cube = iris.cube.Cube(
        np.arange(2, dtype=np.float32),
        standard_name="air_temperature",
        var_name="tas",
        units="K",
        dim_coords_and_dims=[(time, 0)],
        aux_coords_and_dims=[(height, ())],
    )
    iris.save(cube, path)
    return path


def test_scalar_coord_intake_esgf_same_as_local(
    mocker: MockerFixture,
    scalar_coord_file: Path,
) -> None:
    """Test that a scalar coordinate loads the same from both data sources."""
    local_cube = (
        LocalFile(scalar_coord_file).to_iris().extract_cube("air_temperature")
    )

    cat = intake_esgf.ESGFCatalog()
    key = "my.dataset.1"
    mocker.patch.object(
        cat,
        "to_path_dict",
        return_value={key: [scalar_coord_file]},
    )
    mocker.patch.object(
        cat,
        "to_dataset_dict",
        return_value={
            key: xr.open_mfdataset(
                [scalar_coord_file],
                **intake_esgf.config.defaults["default_open_kwargs"],
            ),
        },
    )
    dataset = IntakeESGFDataset(name=key, facets={}, catalog=cat)
    esgf_cube = dataset.to_iris().extract_cube("air_temperature")

    # The source_file attribute differs in formatting between data sources.
    for cube in (local_cube, esgf_cube):
        cube.attributes.globals.pop("source_file", None)
        cube.attributes.locals.pop("source_file", None)
    assert esgf_cube == local_cube
    # Cube equality does not distinguish between DimCoord and AuxCoord, but
    # other operations do, potentially causing a failure in multi-model
    # statistics if cubes come from different data sources.
    for coord in esgf_cube.coords():
        assert isinstance(coord, type(local_cube.coord(coord.name())))
