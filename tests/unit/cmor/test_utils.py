"""Unit tests for :mod:`esmvalcore.cmor._utils`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube

from esmvalcore.cmor._utils import (
    _get_alternative_generic_lev_coord,
    _get_single_cube,
)
from esmvalcore.cmor.table import get_tables

if TYPE_CHECKING:
    from esmvalcore.config import Session


@pytest.mark.parametrize(
    "cubes",
    [[Cube(0)], [Cube(0, var_name="x")], [Cube(0, var_name="y")]],
)
def test_get_single_cube_one_cube(cubes, caplog):
    """Test ``_get_single_cube``."""
    single_cube = _get_single_cube(cubes, "x")
    assert single_cube == cubes[0]
    assert not caplog.records


@pytest.mark.parametrize(
    ("dataset_str", "msg"),
    [
        (None, "Found variable x, but"),
        ("XYZ", "Found variable x in XYZ, but"),
    ],
)
@pytest.mark.parametrize(
    "cubes",
    [
        [Cube(0), Cube(0, var_name="x")],
        [Cube(0, var_name="x"), Cube(0)],
        [Cube(0, var_name="x"), Cube(0, var_name="x")],
        [Cube(0), Cube(0), Cube(0, var_name="x")],
    ],
)
def test_get_single_cube_multiple_cubes(cubes, dataset_str, msg, caplog):
    """Test ``_get_single_cube``."""
    single_cube = _get_single_cube(cubes, "x", dataset_str=dataset_str)
    assert single_cube == Cube(0, var_name="x")
    assert len(caplog.records) == 1
    log = caplog.records[0]
    assert log.levelname == "WARNING"
    assert msg in log.message


@pytest.mark.parametrize(
    ("dataset_str", "msg"),
    [
        (None, "More than one cube found for variable x but"),
        ("XYZ", "More than one cube found for variable x in XYZ but"),
    ],
)
@pytest.mark.parametrize(
    "cubes",
    [
        [Cube(0), Cube(0)],
        [Cube(0, var_name="y"), Cube(0)],
        [Cube(0, var_name="y"), Cube(0, var_name="z")],
        [Cube(0), Cube(0), Cube(0, var_name="z")],
    ],
)
def test_get_single_cube_no_cubes_fail(cubes, dataset_str, msg):
    """Test ``_get_single_cube``."""
    with pytest.raises(ValueError, match=msg):
        _get_single_cube(cubes, "x", dataset_str=dataset_str)


def _get_level_cube(
    points: list[float],
    var_name: str,
    standard_name: str = "air_pressure",
    units: str = "Pa",
) -> Cube:
    """Create a cube with a vertical coordinate."""
    coord = AuxCoord(
        points,
        var_name=var_name,
        standard_name=standard_name,
        units=units,
    )
    return Cube(np.zeros(len(points)), aux_coords_and_dims=[(coord, 0)])


@pytest.mark.parametrize(
    ("project", "coord_name"),
    [
        ("CMIP3", "zlevel"),
        ("CMIP5", "alevel"),
        ("CMIP6", "alevel"),
        ("CMIP6", "alevhalf"),
        ("CMIP7", "alevel"),
        ("obs4MIPs", "alevel"),
    ],
)
def test_get_alternative_generic_lev_coord_requested(
    session: Session,
    project: str,
    coord_name: str,
) -> None:
    """Test that the coordinate matching the requested values is selected."""
    tables = get_tables(session, project)
    candidates = [
        coord
        for coord in tables.coords.values()
        if coord.standard_name in ("air_pressure", "altitude")
        and coord.axis == "Z"
        and not coord.value
        and coord.requested
    ]
    assert len(candidates) >= 2
    for expected in candidates:
        cube = _get_level_cube(
            [float(v) for v in expected.requested],
            var_name=expected.out_name,
            standard_name=expected.standard_name,
            units=expected.units,
        )
        cmor_coord, cube_coord = _get_alternative_generic_lev_coord(
            cube,
            coord_name,
            project,
            session,
        )
        assert cmor_coord.name == expected.name
        assert cube_coord is cube.coord(var_name=expected.out_name)


def test_get_alternative_generic_lev_coord_units(session: Session) -> None:
    """Test that requested values are compared after unit conversion."""
    tables = get_tables(session, "CMIP6")
    points = [float(v) / 100 for v in tables.coords["plev7h"].requested]
    cube = _get_level_cube(points, var_name="plev", units="hPa")
    cmor_coord, _ = _get_alternative_generic_lev_coord(
        cube,
        "alevel",
        "CMIP6",
        session,
    )
    assert cmor_coord.name == "plev7h"


def test_get_alternative_generic_lev_coord_closest_length(
    session: Session,
) -> None:
    """Test that the closest number of levels is selected."""
    cube = _get_level_cube(list(np.linspace(100000.0, 100.0, 20)), "plev")
    cmor_coord, _ = _get_alternative_generic_lev_coord(
        cube,
        "alevel",
        "CMIP6",
        session,
    )
    assert cmor_coord.name == "plev19"


def test_get_alternative_generic_lev_coord_no_requested(
    session: Session,
) -> None:
    """Test that a coordinate without requested values is preferred."""
    cube = _get_level_cube([100000.0, 50000.0, 10000.0, 1000.0], "plev")
    cmor_coord, _ = _get_alternative_generic_lev_coord(
        cube,
        "zlevel",
        "CMIP3",
        session,
    )
    assert cmor_coord.name == "pressure2"


def test_get_alternative_generic_lev_coord_no_single_level(
    session: Session,
) -> None:
    """Test that single level coordinates are not selected."""
    cube = _get_level_cube([85000.0], "plev")
    cmor_coord, _ = _get_alternative_generic_lev_coord(
        cube,
        "alevel",
        "CMIP6",
        session,
    )
    assert not cmor_coord.value
    assert len(cmor_coord.requested) > 1


def test_get_alternative_generic_lev_coord_not_found(
    session: Session,
) -> None:
    """Test that an error is raised if no alternative is found."""
    cube = _get_level_cube([100000.0, 50000.0], "pressure_levels")
    msg = (
        "Found no valid alternative coordinate for generic level coordinate "
        "'alevel'"
    )
    with pytest.raises(ValueError, match=msg):
        _get_alternative_generic_lev_coord(cube, "alevel", "CMIP6", session)


def test_get_alternative_generic_lev_coord_unknown_project(
    session: Session,
) -> None:
    """Test that an error is raised for an unknown project."""
    cube = _get_level_cube([100000.0, 50000.0], "plev")
    with pytest.raises(ValueError, match="Unknown project 'unknown'"):
        _get_alternative_generic_lev_coord(cube, "alevel", "unknown", session)
