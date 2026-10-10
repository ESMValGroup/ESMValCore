from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from esmvalcore.cmor.table import (
    _TABLE_CACHE,
    VariableInfo,
    clear_table_cache,
    get_tables,
)
from esmvalcore.exceptions import InvalidConfigParameter

if TYPE_CHECKING:
    from esmvalcore.config import Session


@pytest.mark.parametrize(
    (
        "project",
        "mip",
        "short_name",
        "branding_suffix",
    ),
    [
        ("CMIP7", "atmos", "tas", "tavg-h2m-hxy-u"),
        ("CMIP7", "atmos", "alb", None),  # custom derived variable
        # custom derived variable with branding suffix of input variables:
        ("CMIP7", "atmos", "rtnt", "tavg-u-hxy-u"),
        ("CMIP6", "Amon", "tas", None),
        ("CMIP6", "Amon", "alb", None),  # custom derived variable
        ("CMIP6", "Amon", "ch4", "Clim"),  # table entry != short_name
        ("CMIP5", "Amon", "tas", None),
        ("CMIP5", "Amon", "alb", None),  # custom derived variable
        ("CMIP3", "A1", "tas", None),
        ("CMIP3", "A1", "lwcre", None),  # custom derived variable
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
        derived=short_name in ("alb", "lwcre", "rtnt"),
    )
    assert isinstance(vardef, VariableInfo)
    assert vardef.short_name
    assert vardef.units
    assert info.project == project
    assert vardef.project == project


def test_get_tables_same_configuration(session: Session) -> None:
    """Test that projects with the same table configuration get separate tables."""
    native6_info = get_tables(session, "native6")
    obs6_info = get_tables(session, "OBS6")
    assert native6_info is not obs6_info
    native6_vardef = native6_info.get_variable("Amon", "tas")
    obs6_vardef = obs6_info.get_variable("Amon", "tas")
    assert isinstance(native6_vardef, VariableInfo)
    assert isinstance(obs6_vardef, VariableInfo)
    assert native6_vardef.project == "native6"
    assert obs6_vardef.project == "OBS6"


def test_get_tables_no_info(session: Session) -> None:
    """Test that the project is set for projects without CMOR tables."""
    session["projects"]["test"] = {
        "cmor_table": {"type": "esmvalcore.cmor.table.NoInfo"},
    }
    info = get_tables(session, "test")
    assert info.project == "test"
    vardef = info.get_variable("Amon", "tas")
    assert isinstance(vardef, VariableInfo)
    assert vardef.project == "test"


def test_get_tables_unknown_project(
    session: Session,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"Unknown project 'unknown', please configure it under 'projects'.",
    ):
        get_tables(session, "unknown")


def test_get_tables_no_type(
    session: Session,
) -> None:
    session["projects"]["test"] = {"cmor_table": {}}
    with pytest.raises(
        ValueError,
        match=(
            r"Missing CMOR table 'type' in configuration of project "
            r"test. Current configuration is:\n{}\n"
        ),
    ):
        get_tables(session, "test")


def test_get_tables_non_existent_table_module(
    session: Session,
) -> None:
    session["projects"]["test"] = {
        "cmor_table": {
            "type": "tests.integration.cmor.does_not_exist.DoesNotExist",
        },
    }
    with pytest.raises(
        InvalidConfigParameter,
        match=(
            r"Failed to import module 'tests.integration.cmor.does_not_exist' "
            r"for CMOR table of project 'test'. Please check your configuration. "
        ),
    ):
        get_tables(session, "test")


def test_get_tables_non_existent_table_class(
    session: Session,
) -> None:
    session["projects"]["test"] = {
        "cmor_table": {
            "type": "tests.integration.cmor.test_read_cmor_tables.NonExistentTable",
        },
    }
    with pytest.raises(
        InvalidConfigParameter,
        match=(
            r"Class 'NonExistentTable' for reading CMOR table of project 'test' "
            r"does not exist in module 'tests.integration.cmor.test_read_cmor_tables'. "
            r"Please check your configuration."
        ),
    ):
        get_tables(session, "test")


class InvalidTable:
    pass


def test_get_tables_invalid_type(
    session: Session,
) -> None:
    session["projects"]["test"] = {
        "cmor_table": {
            "type": "tests.integration.cmor.test_read_cmor_tables.InvalidTable",
        },
    }
    with pytest.raises(
        TypeError,
        match=(
            r"`type` should be a subclass `esmvalcore.cmor.table.InfoBase`, "
            r"but your configuration for project 'test' contains "
        ),
    ):
        get_tables(session, "test")


def test_clear_table_cache(session: Session) -> None:
    assert _TABLE_CACHE
    clear_table_cache()
    assert not _TABLE_CACHE
