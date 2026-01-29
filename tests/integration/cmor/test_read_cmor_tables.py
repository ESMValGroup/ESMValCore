from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest
import yaml

import esmvalcore.cmor.table
from esmvalcore.cmor.table import (
    CMOR_TABLES,
    VariableInfo,
    get_tables,
    read_cmor_tables,
)
from esmvalcore.cmor.table import __file__ as root

if TYPE_CHECKING:
    from esmvalcore.config import Session


def test_read_cmor_tables_raiser():
    """Test func raiser."""
    cfg_file = {"cow": "moo"}
    with pytest.raises(TypeError) as exc:
        read_cmor_tables(cfg_file)
    assert "cow" in str(exc)


def test_read_cmor_tables():
    """Test that the function `read_cmor_tables` loads the tables correctly."""
    table_path = Path(root).parent / "tables"

    for project in "CMIP5", "CMIP6":
        table = CMOR_TABLES[project]
        assert table.paths == (
            table_path / project.lower() / "Tables",
            table_path / f"{project.lower()}-custom",
        )
        assert table.strict is True

    project = "OBS"
    table = CMOR_TABLES[project]
    assert table.paths == (
        table_path / "cmip5" / "Tables",
        table_path / "cmip5-custom",
    )
    assert table.strict is False

    project = "OBS6"
    table = CMOR_TABLES[project]
    assert table.paths == (
        table_path / "cmip6" / "Tables",
        table_path / "cmip6-custom",
    )
    assert table.strict is False

    project = "obs4MIPs"
    table = CMOR_TABLES[project]
    assert table.paths == (
        table_path / "obs4mips" / "Tables",
        table_path / "cmip6-custom",
    )
    assert table.strict is False


@pytest.mark.parametrize(
    (
        "project",
        "mip",
        "short_name",
        "branding_suffix",
    ),
    [
        ("CMIP7", "atmos", "tas", "tavg-h2m-hxy-u"),
        ("CMIP7", "Amon", "alb", None),  # custom derived variable
        ("CMIP6", "Amon", "tas", None),
        ("CMIP6", "Amon", "alb", None),  # custom derived variable
        ("CMIP6", "Amon", "ch4", "Clim"),  # table entry != short_name
        ("CMIP5", "Amon", "tas", None),
        ("CMIP5", "Amon", "alb", None),  # custom derived variable
        ("CMIP3", "A1", "tas", None),
        ("CMIP3", "A1", "alb", None),  # custom derived variable
        ("CORDEX", "mon", "tas", None),
        ("CORDEX", "mon", "alb", None),  # custom derived variable
        ("obs4MIPs", "Amon", "tas", None),
        ("obs4MIPs", "Amon", "agb", None),  # custom variable
        ("obs4MIPs", "Amon", "alb", None),  # custom derived variable
        ("ana4MIPs", "Amon", "tas", None),
        ("native6", "Amon", "tas", None),
        ("native6", "Amon", "agb", None),  # custom variable
        ("native6", "Amon", "alb", None),  # custom derived variable
        ("ACCESS", "Amon", "tas", None),
        ("CESM", "Amon", "tas", None),
        ("EMAC", "Amon", "tas", None),
        ("ICON", "Amon", "tas", None),
        ("IPSLCM", "Amon", "tas", None),
        ("OBS6", "Amon", "tas", None),
        ("OBS6", "Amon", "agb", None),  # custom variable
        ("OBS6", "Amon", "alb", None),  # custom derived variable
        ("OBS", "Amon", "tas", None),
        ("OBS", "Amon", "agb", None),  # custom variable
        ("OBS", "Amon", "alb", None),  # custom derived variable
    ],
)
def test_get_tables(
    session: Session,
    project: str,
    mip: str,
    short_name: str,
    branding_suffix: str | None,
) -> None:
    info = get_tables(session, project)
    assert info.tables
    vardef = info.get_variable(
        mip,
        short_name,
        branding_suffix=branding_suffix,
        derived=short_name == "alb",
    )
    assert isinstance(vardef, VariableInfo)
    assert vardef.short_name
    assert vardef.units


def test_clear_table_cache(session: Session) -> None:
    assert esmvalcore.cmor.table._TABLE_CACHE
    esmvalcore.cmor.table.clear_table_cache()
    assert not esmvalcore.cmor.table._TABLE_CACHE


CMOR_NEWVAR_ENTRY = dedent(
    """
    !============
    variable_entry:    newvarfortesting
    !============
    modeling_realm:    atmos
    !----------------------------------
    ! Variable attributes:
    !----------------------------------
    standard_name:
    units:             kg s m A
    cell_methods:      time: mean
    cell_measures:     area: areacella
    long_name:         Custom Variable for Testing
    !----------------------------------
    ! Additional variable information:
    !----------------------------------
    dimensions:        longitude latitude time
    type:              real
    positive:          up
    !----------------------------------
    !
    """,
)
CMOR_NETCRE_ENTRY = dedent(
    """
    !============
    variable_entry:    netcre
    !============
    modeling_realm:    atmos
    !----------------------------------
    ! Variable attributes:
    !----------------------------------
    standard_name:     air_temperature  ! for testing
    units:             K                ! for testing
    cell_methods:      time: mean
    cell_measures:     area: areacella
    long_name:         This is New      ! for testing
    !----------------------------------
    ! Additional variable information:
    !----------------------------------
    dimensions:        longitude latitude time
    type:              real
    positive:          up
    !----------------------------------
    !
    """,
)
CMOR_NEWCOORD_ENTRY = dedent(
    """
    !============
    axis_entry: newcoordfortesting
    !============
    !----------------------------------
    ! Axis attributes:
    !----------------------------------
    standard_name:
    units:            kg
    axis:             Y             ! X, Y, Z, T (default: undeclared)
    long_name:        Custom Coordinate for Testing
    !----------------------------------
    ! Additional axis information:
    !----------------------------------
    out_name:         newcoordfortesting
    valid_min:        -90.0
    valid_max:        90.0
    stored_direction: increasing
    type:             double
    must_have_bounds: yes
    !----------------------------------
    !
    """,
)


def test_read_custom_cmor_tables_config_developer(tmp_path):
    """Test reading of custom CMOR tables."""
    (tmp_path / "CMOR_newvarfortesting.dat").write_text(CMOR_NEWVAR_ENTRY)
    (tmp_path / "CMOR_netcre.dat").write_text(CMOR_NETCRE_ENTRY)
    (tmp_path / "CMOR_coordinates.dat").write_text(CMOR_NEWCOORD_ENTRY)

    custom_cfg_developer = {
        "custom": {"cmor_path": str(tmp_path)},
        "CMIP6": {
            "cmor_strict": True,
            "input_dir": {"default": "/"},
            "input_file": "*.nc",
            "output_file": "out.nc",
            "cmor_type": "CMIP6",
        },
    }
    cfg_file = tmp_path / "config-developer.yml"
    with cfg_file.open("w", encoding="utf-8") as file:
        yaml.safe_dump(custom_cfg_developer, file)

    read_cmor_tables(cfg_file)

    assert len(CMOR_TABLES) == 2
    assert "CMIP6" in CMOR_TABLES
    assert "custom" in CMOR_TABLES

    custom_table = CMOR_TABLES["custom"]
    assert custom_table.paths == (
        Path(root).parent / "tables" / "cmip5-custom",
        tmp_path,
    )

    # Make sure that default tables have been read
    assert "alb" in custom_table.tables["custom"]
    assert "latitude" in custom_table.coords

    # Make sure that custom tables have been read
    assert "newvarfortesting" in custom_table.tables["custom"]
    assert "newcoordfortesting" in custom_table.coords
    netcre = custom_table.get_variable("custom", "netcre")
    assert netcre.standard_name == "air_temperature"
    assert netcre.units == "K"
    assert netcre.long_name == "This is New"

    cmip6_table = CMOR_TABLES["CMIP6"]
    assert cmip6_table.default is custom_table

    # Restore default tables
    read_cmor_tables()
