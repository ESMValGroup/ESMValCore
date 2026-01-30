"""Integration tests for the variable_info module."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

import esmvalcore.cmor
import esmvalcore.cmor.table
from esmvalcore.cmor.table import (
    CMIP3Info,
    CMIP5Info,
    CMIP6Info,
    CustomInfo,
    NoInfo,
    Obs4MIPsInfo,
    VariableInfo,
    _get_branding_suffixes,
    _get_mips,
    _update_cmor_facets,
    get_var_info,
)


def test_update_cmor_facets():
    facets = {
        "project": "CMIP6",
        "mip": "Amon",
        "short_name": "tas",
        "dataset": "CanESM5",
        "exp": "historical",
    }

    _update_cmor_facets(facets)

    expected = {
        "project": "CMIP6",
        "mip": "Amon",
        "short_name": "tas",
        "dataset": "CanESM5",
        "original_short_name": "tas",
        "standard_name": "air_temperature",
        "long_name": "Near-Surface Air Temperature",
        "units": "K",
        "modeling_realm": ["atmos"],
        "frequency": "mon",
        "activity": "CMIP",
        "exp": "historical",
        "institute": [
            "CCCma",
        ],
    }
    assert facets == expected


def test_update_cmor_facets_facet_not_in_table(mocker):
    facets = {
        "project": "CMIP6",
        "mip": "Amon",
        "short_name": "tas",
    }

    mocker.patch.object(
        esmvalcore.cmor.table,
        "getattr",
        create_autospec=True,
        return_value=None,
    )
    _update_cmor_facets(facets)

    expected = {
        "project": "CMIP6",
        "mip": "Amon",
        "short_name": "tas",
        "original_short_name": "tas",
    }
    assert facets == expected


class TestCMIP6Info:
    """Tests for the CMIP6 info class."""

    @pytest.fixture
    def variables_info(self) -> CMIP6Info:
        return CMIP6Info(
            paths=[
                Path("cmip6/Tables"),
                Path("cmip6-custom"),
            ],
            strict=True,
            alt_names=[
                ["sic", "siconc"],
                ["tro3", "o3"],
            ],
        )

    def test_repr(self, variables_info: CMIP6Info) -> None:
        builtin_tables_path = Path(esmvalcore.cmor.__file__).parent / "tables"
        expected_paths = [
            builtin_tables_path / "cmip6" / "Tables",
            builtin_tables_path / "cmip6-custom",
        ]
        result = repr(variables_info)
        assert result.startswith(
            f"CMIP6Info(paths={expected_paths}, strict=True, alt_names=",
        )
        assert result.endswith(")")

    def test_custom_tables_location(self):
        """Test constructor with custom tables location."""
        cmor_path = os.path.dirname(os.path.realpath(esmvalcore.cmor.__file__))
        cmor_tables_path = os.path.join(cmor_path, "tables", "cmip6")
        cmor_tables_path = os.path.abspath(cmor_tables_path)
        CMIP6Info(cmor_tables_path, default=None, strict=False)

    def test_get_table_frequency(self, variables_info):
        """Test get table frequency."""
        assert variables_info.get_table("Amon").frequency == "mon"
        assert variables_info.get_table("day").frequency == "day"

    def test_get_variable_tas(self, variables_info):
        """Get tas variable."""
        var = variables_info.get_variable("Amon", "tas")
        assert var.short_name == "tas"

    def test_get_variable_from_alt_names(self, variables_info):
        """Get a variable from a known alt_names."""
        var = variables_info.get_variable("SImon", "sic")
        assert var.short_name == "siconc"

    def test_get_variable_derived(self, variables_info):
        """Test that derived variable are looked up from other MIP tables."""
        var = variables_info.get_variable("3hr", "sfcWind", derived=True)
        assert var.short_name == "sfcWind"

    def test_get_variable_from_custom(self, variables_info):
        """Get a variable from default."""
        variables_info.strict = False
        var = variables_info.get_variable("Amon", "swcre")
        assert var.short_name == "swcre"
        assert var.frequency == "mon"

        var = variables_info.get_variable("day", "swcre")
        assert var.short_name == "swcre"
        assert var.frequency == "day"

    def test_get_bad_variable(self, variables_info):
        """Get none if a variable is not in the given table."""
        assert variables_info.get_variable("Omon", "ta") is None

    def test_omon_ta_fail_if_strict(self, variables_info):
        """Get ta fails with Omon if strict."""
        assert variables_info.get_variable("Omon", "ta") is None

    def test_omon_ta_succes_if_strict(self, variables_info):
        """Get ta does not fail with AERMonZ if not strict."""
        variables_info.strict = False
        var = variables_info.get_variable("Omon", "ta")
        assert var.short_name == "ta"
        assert var.frequency == "mon"

    def test_omon_toz_succes_if_strict(self, variables_info):
        """Get toz does not fail with Omon if not strict."""
        variables_info.strict = False
        var = variables_info.get_variable("Omon", "toz")
        assert var.short_name == "toz"
        assert var.frequency == "mon"

    def test_get_institute_from_source(self, variables_info):
        """Get institution for source ACCESS-CM2."""
        institute = variables_info.institutes["ACCESS-CM2"]
        assert institute == ["CSIRO-ARCCSS"]

    def test_get_activity_from_exp(self, variables_info):
        """Get activity for experiment 1pctCO2."""
        activity = variables_info.activities["1pctCO2"]
        assert activity == ["CMIP"]

    def test_invalid_path(self) -> None:
        path = Path(__file__) / "path" / "does" / "not" / "exist"
        msg = r"CMOR tables not found in"
        with pytest.raises(ValueError, match=msg):
            CMIP6Info(str(path))

    def test_invalid_paths(self) -> None:
        path = Path(__file__) / "path" / "does" / "not" / "exist"
        with pytest.raises(NotADirectoryError, match=str(path)):
            CMIP6Info(paths=[path])

    def test_no_tables_in_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            CMIP6Info(paths=[tmp_path])

    def test_invalid_file(self, tmp_path: Path) -> None:
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("invalid content", encoding="utf-8")
        with pytest.raises(ValueError):
            CMIP6Info(paths=[tmp_path])

    def test_invalid_file_logged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # attach the root logger handler added by caplog
        monkeypatch.setattr(
            esmvalcore.cmor.table.logger,
            "handlers",
            logging.getLogger().handlers,
        )
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("invalid content", encoding="utf-8")
        with pytest.raises(ValueError):
            CMIP6Info(paths=[tmp_path])
        assert (
            f"Exception raised when loading {tmp_path}/invalid.json"
            in caplog.messages
        )


class Testobs4mipsInfo:
    """Tests for the obs4mips info class."""

    @pytest.fixture
    def variables_info(self) -> Obs4MIPsInfo:
        return Obs4MIPsInfo(
            paths=[
                Path("obs4mips/Tables"),
                Path("cmip6-custom"),
            ],
            strict=False,
        )

    def test_get_table_frequency(self, variables_info):
        """Test get table frequency."""
        assert variables_info.get_table("monStderr").frequency == "mon"

    def test_custom_tables_location(self):
        """Test constructor with custom tables location."""
        cmor_path = os.path.dirname(os.path.realpath(esmvalcore.cmor.__file__))
        cmor_tables_path = os.path.join(cmor_path, "tables", "cmip6")
        cmor_tables_path = os.path.abspath(cmor_tables_path)
        CMIP6Info(cmor_tables_path, None, True)

    def test_get_variable_ndvistderr(self, variables_info):
        """Get ndviStderr variable.

        Note table name obs4MIPs_[mip]
        """
        var = variables_info.get_variable(
            "obs4MIPs_monStderr",
            "ndviStderr",
        )
        assert var.short_name == "ndviStderr"
        assert var.frequency == "mon"

    def test_get_variable_hus(self, variables_info):
        """Get hus variable."""
        var = variables_info.get_variable("Amon", "hus")
        assert var.short_name == "hus"
        assert var.frequency == "mon"

    def test_get_variable_hus_default_prefix(self, variables_info):
        """Get hus variable."""
        var = variables_info.get_variable("Amon", "hus")
        assert var.short_name == "hus"
        assert var.frequency == "mon"

    def test_get_variable_from_custom(self, variables_info):
        """Get prStderr variable.

        Note table name obs4MIPs_[mip]
        """
        var = variables_info.get_variable(
            "monStderr",
            "prStderr",
        )
        assert var.short_name == "prStderr"
        assert var.frequency == "mon"

    def test_get_variable_from_custom_deriving(self, variables_info):
        """Get a variable from default."""
        var = variables_info.get_variable(
            "Amon",
            "swcre",
            derived=True,
        )
        assert var.short_name == "swcre"
        assert var.frequency == "mon"

        var = variables_info.get_variable(
            "Aday",
            "swcre",
            derived=True,
        )
        assert var.short_name == "swcre"
        assert var.frequency == "day"

    def test_get_bad_variable(self, variables_info):
        """Get none if a variable is not in the given table."""
        assert variables_info.get_variable("Omon", "tras") is None


class TestCMIP5Info:
    """Tests for the CMIP5 info class."""

    @pytest.fixture
    def variables_info(self) -> CMIP5Info:
        return CMIP5Info(
            paths=[
                Path("cmip5/Tables"),
                Path("cmip5-custom"),
            ],
            strict=True,
        )

    def test_custom_tables_location(self):
        """Test constructor with custom tables location."""
        cmor_path = os.path.dirname(os.path.realpath(esmvalcore.cmor.__file__))
        cmor_tables_path = os.path.join(cmor_path, "tables", "cmip5")
        cmor_tables_path = os.path.abspath(cmor_tables_path)
        CMIP5Info(cmor_tables_path, None, True)

    def test_get_variable_tas(self, variables_info):
        """Get tas variable."""
        var = variables_info.get_variable("Amon", "tas")
        assert var.short_name == "tas"

    def test_get_variable_zg(self, variables_info):
        """Get zg variable."""
        var = variables_info.get_variable("Amon", "zg")
        assert var.short_name == "zg"
        assert var.coordinates["plevs"].requested == [
            "100000.",
            "92500.",
            "85000.",
            "70000.",
            "60000.",
            "50000.",
            "40000.",
            "30000.",
            "25000.",
            "20000.",
            "15000.",
            "10000.",
            "7000.",
            "5000.",
            "3000.",
            "2000.",
            "1000.",
        ]

    def test_get_variable_from_custom(self, variables_info):
        """Get a variable from default."""
        variables_info.strict = False
        var = variables_info.get_variable("Amon", "swcre")
        assert var.short_name == "swcre"
        assert var.frequency == "mon"

        var = variables_info.get_variable("day", "swcre")
        assert var.short_name == "swcre"
        assert var.frequency == "day"

    def test_get_bad_variable(self, variables_info):
        """Get none if a variable is not in the given table."""
        assert variables_info.get_variable("Omon", "tas") is None

    def test_aermon_ta_fail_if_strict(self, variables_info):
        """Get ta fails with AERMonZ if strict."""
        assert variables_info.get_variable("Omon", "ta") is None

    def test_aermon_ta_succes_if_strict(self, variables_info):
        """Get ta does not fail with Omon if not strict."""
        variables_info.strict = False
        var = variables_info.get_variable("Omon", "ta")
        assert var.short_name == "ta"
        assert var.frequency == "mon"

    def test_omon_toz_succes_if_strict(self, variables_info):
        """Get toz does not fail with Omon if not strict."""
        variables_info.strict = False
        var = variables_info.get_variable("Omon", "toz")
        assert var.short_name == "toz"
        assert var.frequency == "mon"

    def test_invalid_file(self, tmp_path: Path) -> None:
        invalid_file = tmp_path / "invalid"
        invalid_file.write_text("invalid content", encoding="utf-8")
        with pytest.raises(ValueError):
            CMIP5Info(paths=[tmp_path])

    def test_invalid_file_logged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # attach the root logger handler added by caplog
        monkeypatch.setattr(
            esmvalcore.cmor.table.logger,
            "handlers",
            logging.getLogger().handlers,
        )
        invalid_file = tmp_path / "invalid"
        invalid_file.write_text("invalid content", encoding="utf-8")
        with pytest.raises(ValueError):
            CMIP5Info(paths=[tmp_path])
        assert (
            f"Exception raised when loading {tmp_path}/invalid"
            in caplog.messages
        )


class TestCMIP3Info:
    """Tests for the CMIP3 info class."""

    @pytest.fixture
    def variables_info(self) -> CMIP3Info:
        return CMIP3Info(
            paths=[
                Path("cmip3/Tables"),
                Path("cmip5-custom"),
            ],
            strict=True,
        )

    def test_custom_tables_location(self):
        """Test constructor with custom tables location."""
        cmor_path = os.path.dirname(os.path.realpath(esmvalcore.cmor.__file__))
        cmor_tables_path = os.path.join(cmor_path, "tables", "cmip3")
        cmor_tables_path = os.path.abspath(cmor_tables_path)
        CMIP3Info(cmor_tables_path, None, True)

    def test_get_variable_tas(self, variables_info):
        """Get tas variable."""
        var = variables_info.get_variable("A1", "tas")
        assert var.short_name == "tas"

    def test_get_variable_zg(self, variables_info):
        """Get zg variable."""
        var = variables_info.get_variable("A1", "zg")
        assert var.short_name == "zg"
        assert var.coordinates["pressure"].requested == [
            "100000.",
            "92500.",
            "85000.",
            "70000.",
            "60000.",
            "50000.",
            "40000.",
            "30000.",
            "25000.",
            "20000.",
            "15000.",
            "10000.",
            "7000.",
            "5000.",
            "3000.",
            "2000.",
            "1000.",
        ]

    def test_get_variable_from_custom(self, variables_info):
        """Get a variable from default."""
        variables_info.strict = False
        var = variables_info.get_variable("A1", "swcre")
        assert var.short_name == "swcre"
        assert var.frequency == ""

        var = variables_info.get_variable("day", "swcre")
        assert var.short_name == "swcre"
        assert var.frequency == ""

    def test_get_bad_variable(self, variables_info):
        """Get none if a variable is not in the given table."""
        assert variables_info.get_variable("O1", "tas") is None

    def test_aermon_ta_fail_if_strict(self, variables_info):
        """Get ta fails with AERMonZ if strict."""
        assert variables_info.get_variable("O1", "ta") is None

    def test_aermon_ta_succes_if_strict(self, variables_info):
        """Get ta does not fail with Omon if not strict."""
        variables_info.strict = False
        var = variables_info.get_variable("O1", "ta")
        assert var.short_name == "ta"
        assert var.frequency == ""

    def test_omon_toz_succes_if_strict(self, variables_info):
        """Get toz does not fail with Omon if not strict."""
        variables_info.strict = False
        var = variables_info.get_variable("O1", "toz")
        assert var.short_name == "toz"
        assert var.frequency == ""


class TestCORDEXInfo:
    """Tests for the CORDEX info class."""

    @pytest.fixture
    def variables_info(self) -> CMIP5Info:
        return CMIP5Info(
            paths=[
                Path("cordex/Tables"),
                Path("cmip5-custom"),
            ],
        )

    def test_custom_tables_location(self):
        """Test constructor with custom tables location."""
        cmor_path = os.path.dirname(os.path.realpath(esmvalcore.cmor.__file__))
        cmor_tables_path = os.path.join(cmor_path, "tables", "cordex")
        CMIP5Info(cmor_tables_path)

    def test_get_variable_tas(self, variables_info):
        """Get tas variable."""
        var = variables_info.get_variable("mon", "tas")
        assert var.short_name == "tas"

    def test_get_bad_variable(self, variables_info):
        """Get none if a variable is not in the given table."""
        assert variables_info.get_variable("Omon", "tas") is None


class TestCustomInfo:
    """Tests for the custom info class."""

    @pytest.fixture
    def variables_info(self) -> CustomInfo:
        return CustomInfo()

    def test_repr(self, variables_info: CustomInfo) -> None:
        builtin_tables_path = Path(esmvalcore.cmor.__file__).parent / "tables"
        expected_paths = [
            builtin_tables_path / "cmip5-custom",
        ]
        result = repr(variables_info)
        assert result == f"CustomInfo(paths={expected_paths})"

    def test_custom_tables_default_location(self, variables_info):
        """Test constructor with default tables location."""
        custom_info = CustomInfo()
        expected_cmor_folder = os.path.join(
            os.path.dirname(esmvalcore.cmor.__file__),
            "tables",
            "cmip5-custom",
        )
        assert custom_info.paths == (Path(expected_cmor_folder),)
        assert custom_info.tables["custom"]
        assert custom_info.coords

    def test_custom_tables_location(self, variables_info):
        """Test constructor with custom tables location."""
        cmor_path = os.path.dirname(os.path.realpath(esmvalcore.cmor.__file__))
        default_cmor_tables_path = os.path.join(
            cmor_path,
            "tables",
            "cmip5-custom",
        )
        cmor_tables_path = os.path.join(cmor_path, "tables", "cmip5")
        cmor_tables_path = os.path.abspath(cmor_tables_path)

        custom_info = CustomInfo(cmor_tables_path)

        assert custom_info.paths == (
            Path(default_cmor_tables_path),
            Path(cmor_tables_path),
        )
        assert custom_info.tables["custom"]
        assert custom_info.coords

    def test_custom_tables_invalid_location(self):
        """Test constructor with invalid custom tables location."""
        with pytest.raises(ValueError):
            CustomInfo("this_file_does_not_exist.dat")

    def test_get_variable_netcre(self, variables_info):
        """Get tas variable."""
        CustomInfo()
        var = variables_info.get_variable("Amon", "netcre")
        assert var.short_name == "netcre"

    def test_get_bad_variable(self, variables_info):
        """Get none if a variable is not in the given table."""
        assert variables_info.get_variable("Omon", "badvar") is None

    def test_get_variable_tasconf5(self, variables_info):
        """Get tas variable."""
        CustomInfo()
        var = variables_info.get_variable("Amon", "tasConf5")
        assert var.short_name == "tasConf5"
        assert (
            var.long_name == "Near-Surface Air Temperature Uncertainty Range"
        )
        assert var.units == "K"

    def test_get_variable_tasconf95(self, variables_info):
        """Get tas variable."""
        CustomInfo()
        var = variables_info.get_variable("Amon", "tasConf95")
        assert var.short_name == "tasConf95"
        assert (
            var.long_name == "Near-Surface Air Temperature Uncertainty Range"
        )
        assert var.units == "K"

    def test_get_variable_tasaga(self, variables_info):
        """Get tas variable."""
        CustomInfo()
        var = variables_info.get_variable("Amon", "tasaga")
        assert var.short_name == "tasaga"
        assert (
            var.long_name == "Global-mean Near-Surface Air Temperature Anomaly"
        )
        assert var.units == "K"

    def test_get_variable_ch4s(self, variables_info):
        """Get ch4s variable."""
        CustomInfo()
        var = variables_info.get_variable("Amon", "ch4s")
        assert var.short_name == "ch4s"
        assert var.long_name == "Atmosphere CH4 surface"
        assert var.units == "1e-09"

    def test_get_variable_tosstderr(self, variables_info):
        """Get tosStderr variable."""
        CustomInfo()
        var = variables_info.get_variable("Omon", "tosStderr")
        assert var.short_name == "tosStderr"
        assert var.long_name == "Sea Surface Temperature Error"
        assert var.units == "K"


class TestNoInfo:
    """Tests for the no info class."""

    @pytest.fixture
    def variables_info(self) -> NoInfo:
        return NoInfo()

    def test_repr(self, variables_info: NoInfo) -> None:
        result = repr(variables_info)
        assert result == "NoInfo()"

    def test_get_variable_tas(self, variables_info: NoInfo) -> None:
        """Get tas variable."""
        var = variables_info.get_variable("Amon", "tas")
        assert isinstance(var, VariableInfo)
        assert var.short_name == "tas"


@pytest.mark.parametrize(
    ("project", "mip", "short_name", "frequency"),
    [
        ("CMIP5", "Amon", "tas", "mon"),
        ("CMIP5", "day", "tas", "day"),
        ("CMIP6", "Amon", "tas", "mon"),
        ("CMIP6", "day", "tas", "day"),
        ("CORDEX", "3hr", "tas", "3hr"),
    ],
)
def test_get_var_info(project, mip, short_name, frequency):
    """Test ``get_var_info``."""
    var_info = get_var_info(project, mip, short_name)

    assert var_info.short_name == short_name
    assert var_info.frequency == frequency


@pytest.mark.parametrize(
    ("mip", "short_name"),
    [
        ("INVALID_MIP", "tas"),
        ("Amon", "INVALID_VAR"),
    ],
)
def test_get_var_info_invalid_mip_short_name(mip, short_name):
    """Test ``get_var_info``."""
    var_info = get_var_info("CMIP6", mip, short_name)

    assert var_info is None


def test_get_var_info_invalid_project():
    """Test ``get_var_info``."""
    with pytest.raises(KeyError):
        get_var_info("INVALID_PROJECT", "Amon", "tas")


def test_get_mips_cmip5() -> None:
    """Test ``_get_mips``."""
    mips = _get_mips(project="CMIP5", short_name="tas")
    expected = {
        "3hr",
        "Amon",
        "cf3hr",
        "cfSites",
        "day",
    }
    assert set(mips) == expected


def test_get_mips_cmip7() -> None:
    """Test ``_get_mips``."""
    mips = _get_mips(project="CMIP7", short_name="tas")
    expected = {"atmos", "land", "landIce"}
    assert set(mips) == expected


def test_get_branding_suffixes() -> None:
    """Test ``_get_branding_suffixes``."""
    suffixes = _get_branding_suffixes(
        project="CMIP7",
        mip="atmos",
        short_name="areacella",
    )
    expected = {"ti-u-hxy-u"}
    assert set(suffixes) == expected
