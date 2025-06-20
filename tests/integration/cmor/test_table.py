"""Integration tests for the variable_info module."""

import os
import unittest

import pytest

import esmvalcore.cmor
from esmvalcore.cmor.table import (
    CMIP3Info,
    CMIP5Info,
    CMIP6Info,
    CustomInfo,
    _update_cmor_facets,
    get_var_info,
)


def test_update_cmor_facets():
    facets = {
        "project": "CMIP6",
        "mip": "Amon",
        "short_name": "tas",
    }

    _update_cmor_facets(facets)

    expected = {
        "project": "CMIP6",
        "mip": "Amon",
        "short_name": "tas",
        "original_short_name": "tas",
        "standard_name": "air_temperature",
        "long_name": "Near-Surface Air Temperature",
        "units": "K",
        "modeling_realm": ["atmos"],
        "frequency": "mon",
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


class TestCMIP6Info(unittest.TestCase):
    """Test for the CMIP6 info class."""

    @classmethod
    def setUpClass(cls):
        """Set up tests.

        We read CMIP6Info once to keep tests times manageable
        """
        cls.variables_info = CMIP6Info(
            "cmip6",
            default=CustomInfo(),
            strict=True,
            alt_names=[
                ["sic", "siconc"],
                ["tro3", "o3"],
            ],
        )

    def setUp(self):
        self.variables_info.strict = True

    def test_custom_tables_location(self):
        """Test constructor with custom tables location."""
        cmor_path = os.path.dirname(os.path.realpath(esmvalcore.cmor.__file__))
        cmor_tables_path = os.path.join(cmor_path, "tables", "cmip6")
        cmor_tables_path = os.path.abspath(cmor_tables_path)
        CMIP6Info(cmor_tables_path, default=None, strict=False)

    def test_get_table_frequency(self):
        """Test get table frequency."""
        self.assertEqual(
            self.variables_info.get_table("Amon").frequency,
            "mon",
        )
        self.assertEqual(self.variables_info.get_table("day").frequency, "day")

    def test_get_variable_tas(self):
        """Get tas variable."""
        var = self.variables_info.get_variable("Amon", "tas")
        self.assertEqual(var.short_name, "tas")

    def test_get_variable_from_alt_names(self):
        """Get a variable from a known alt_names."""
        var = self.variables_info.get_variable("SImon", "sic")
        self.assertEqual(var.short_name, "siconc")

    def test_get_variable_derived(self):
        """Test that derived variable are looked up from other MIP tables."""
        var = self.variables_info.get_variable("3hr", "sfcWind", derived=True)
        self.assertEqual(var.short_name, "sfcWind")

    def test_get_variable_from_custom(self):
        """Get a variable from default."""
        self.variables_info.strict = False
        var = self.variables_info.get_variable("Amon", "swcre")
        self.assertEqual(var.short_name, "swcre")
        self.assertEqual(var.frequency, "mon")

        var = self.variables_info.get_variable("day", "swcre")
        self.assertEqual(var.short_name, "swcre")
        self.assertEqual(var.frequency, "day")

    def test_get_bad_variable(self):
        """Get none if a variable is not in the given table."""
        self.assertIsNone(self.variables_info.get_variable("Omon", "ta"))

    def test_omon_ta_fail_if_strict(self):
        """Get ta fails with Omon if strict."""
        self.assertIsNone(self.variables_info.get_variable("Omon", "ta"))

    def test_omon_ta_succes_if_strict(self):
        """Get ta does not fail with AERMonZ if not strict."""
        self.variables_info.strict = False
        var = self.variables_info.get_variable("Omon", "ta")
        self.assertEqual(var.short_name, "ta")
        self.assertEqual(var.frequency, "mon")

    def test_omon_toz_succes_if_strict(self):
        """Get toz does not fail with Omon if not strict."""
        self.variables_info.strict = False
        var = self.variables_info.get_variable("Omon", "toz")
        self.assertEqual(var.short_name, "toz")
        self.assertEqual(var.frequency, "mon")

    def test_get_institute_from_source(self):
        """Get institution for source ACCESS-CM2."""
        institute = self.variables_info.institutes["ACCESS-CM2"]
        self.assertListEqual(institute, ["CSIRO-ARCCSS"])

    def test_get_activity_from_exp(self):
        """Get activity for experiment 1pctCO2."""
        activity = self.variables_info.activities["1pctCO2"]
        self.assertListEqual(activity, ["CMIP"])


class Testobs4mipsInfo(unittest.TestCase):
    """Test for the obs$mips info class."""

    @classmethod
    def setUpClass(cls):
        """Set up tests.

        We read CMIP6Info once to keep tests times manageable
        """
        cls.variables_info = CMIP6Info(
            cmor_tables_path="obs4mips",
            default=CustomInfo(),
            strict=False,
            default_table_prefix="obs4MIPs_",
        )

    def setUp(self):
        self.variables_info.strict = False

    def test_get_table_frequency(self):
        """Test get table frequency."""
        self.assertEqual(
            self.variables_info.get_table("obs4MIPs_monStderr").frequency,
            "mon",
        )

    def test_custom_tables_location(self):
        """Test constructor with custom tables location."""
        cmor_path = os.path.dirname(os.path.realpath(esmvalcore.cmor.__file__))
        cmor_tables_path = os.path.join(cmor_path, "tables", "cmip6")
        cmor_tables_path = os.path.abspath(cmor_tables_path)
        CMIP6Info(cmor_tables_path, None, True)

    def test_get_variable_ndvistderr(self):
        """Get ndviStderr variable.

        Note table name obs4MIPs_[mip]
        """
        var = self.variables_info.get_variable(
            "obs4MIPs_monStderr",
            "ndviStderr",
        )
        self.assertEqual(var.short_name, "ndviStderr")
        self.assertEqual(var.frequency, "mon")

    def test_get_variable_hus(self):
        """Get hus variable."""
        var = self.variables_info.get_variable("obs4MIPs_Amon", "hus")
        self.assertEqual(var.short_name, "hus")
        self.assertEqual(var.frequency, "mon")

    def test_get_variable_hus_default_prefix(self):
        """Get hus variable."""
        var = self.variables_info.get_variable("Amon", "hus")
        self.assertEqual(var.short_name, "hus")
        self.assertEqual(var.frequency, "mon")

    def test_get_variable_from_custom(self):
        """Get prStderr variable.

        Note table name obs4MIPs_[mip]
        """
        var = self.variables_info.get_variable(
            "obs4MIPs_monStderr",
            "prStderr",
        )
        self.assertEqual(var.short_name, "prStderr")
        self.assertEqual(var.frequency, "mon")

    def test_get_variable_from_custom_deriving(self):
        """Get a variable from default."""
        var = self.variables_info.get_variable(
            "obs4MIPs_Amon",
            "swcre",
            derived=True,
        )
        self.assertEqual(var.short_name, "swcre")
        self.assertEqual(var.frequency, "mon")

        var = self.variables_info.get_variable(
            "obs4MIPs_Aday",
            "swcre",
            derived=True,
        )
        self.assertEqual(var.short_name, "swcre")
        self.assertEqual(var.frequency, "day")

    def test_get_bad_variable(self):
        """Get none if a variable is not in the given table."""
        self.assertIsNone(self.variables_info.get_variable("Omon", "tras"))


class TestCMIP5Info(unittest.TestCase):
    """Test for the CMIP5 info class."""

    @classmethod
    def setUpClass(cls):
        """Set up tests.

        We read CMIP5Info once to keep testing times manageable
        """
        cls.variables_info = CMIP5Info("cmip5", CustomInfo(), strict=True)

    def setUp(self):
        self.variables_info.strict = True

    def test_custom_tables_location(self):
        """Test constructor with custom tables location."""
        cmor_path = os.path.dirname(os.path.realpath(esmvalcore.cmor.__file__))
        cmor_tables_path = os.path.join(cmor_path, "tables", "cmip5")
        cmor_tables_path = os.path.abspath(cmor_tables_path)
        CMIP5Info(cmor_tables_path, None, True)

    def test_get_variable_tas(self):
        """Get tas variable."""
        var = self.variables_info.get_variable("Amon", "tas")
        self.assertEqual(var.short_name, "tas")

    def test_get_variable_zg(self):
        """Get zg variable."""
        var = self.variables_info.get_variable("Amon", "zg")
        self.assertEqual(var.short_name, "zg")
        self.assertEqual(
            var.coordinates["plevs"].requested,
            [
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
            ],
        )

    def test_get_variable_from_custom(self):
        """Get a variable from default."""
        self.variables_info.strict = False
        var = self.variables_info.get_variable("Amon", "swcre")
        self.assertEqual(var.short_name, "swcre")
        self.assertEqual(var.frequency, "mon")

        var = self.variables_info.get_variable("day", "swcre")
        self.assertEqual(var.short_name, "swcre")
        self.assertEqual(var.frequency, "day")

    def test_get_bad_variable(self):
        """Get none if a variable is not in the given table."""
        self.assertIsNone(self.variables_info.get_variable("Omon", "tas"))

    def test_aermon_ta_fail_if_strict(self):
        """Get ta fails with AERMonZ if strict."""
        self.assertIsNone(self.variables_info.get_variable("Omon", "ta"))

    def test_aermon_ta_succes_if_strict(self):
        """Get ta does not fail with Omon if not strict."""
        self.variables_info.strict = False
        var = self.variables_info.get_variable("Omon", "ta")
        self.assertEqual(var.short_name, "ta")
        self.assertEqual(var.frequency, "mon")

    def test_omon_toz_succes_if_strict(self):
        """Get toz does not fail with Omon if not strict."""
        self.variables_info.strict = False
        var = self.variables_info.get_variable("Omon", "toz")
        self.assertEqual(var.short_name, "toz")
        self.assertEqual(var.frequency, "mon")


class TestCMIP3Info(unittest.TestCase):
    """Test for the CMIP5 info class."""

    @classmethod
    def setUpClass(cls):
        """Set up tests.

        We read CMIP5Info once to keep testing times manageable
        """
        cls.variables_info = CMIP3Info("cmip3", CustomInfo(), strict=True)

    def setUp(self):
        self.variables_info.strict = True

    def test_custom_tables_location(self):
        """Test constructor with custom tables location."""
        cmor_path = os.path.dirname(os.path.realpath(esmvalcore.cmor.__file__))
        cmor_tables_path = os.path.join(cmor_path, "tables", "cmip3")
        cmor_tables_path = os.path.abspath(cmor_tables_path)
        CMIP3Info(cmor_tables_path, None, True)

    def test_get_variable_tas(self):
        """Get tas variable."""
        var = self.variables_info.get_variable("A1", "tas")
        self.assertEqual(var.short_name, "tas")

    def test_get_variable_zg(self):
        """Get zg variable."""
        var = self.variables_info.get_variable("A1", "zg")
        self.assertEqual(var.short_name, "zg")
        self.assertEqual(
            var.coordinates["pressure"].requested,
            [
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
            ],
        )

    def test_get_variable_from_custom(self):
        """Get a variable from default."""
        self.variables_info.strict = False
        var = self.variables_info.get_variable("A1", "swcre")
        self.assertEqual(var.short_name, "swcre")
        self.assertEqual(var.frequency, "")

        var = self.variables_info.get_variable("day", "swcre")
        self.assertEqual(var.short_name, "swcre")
        self.assertEqual(var.frequency, "")

    def test_get_bad_variable(self):
        """Get none if a variable is not in the given table."""
        self.assertIsNone(self.variables_info.get_variable("O1", "tas"))

    def test_aermon_ta_fail_if_strict(self):
        """Get ta fails with AERMonZ if strict."""
        self.assertIsNone(self.variables_info.get_variable("O1", "ta"))

    def test_aermon_ta_succes_if_strict(self):
        """Get ta does not fail with Omon if not strict."""
        self.variables_info.strict = False
        var = self.variables_info.get_variable("O1", "ta")
        self.assertEqual(var.short_name, "ta")
        self.assertEqual(var.frequency, "")

    def test_omon_toz_succes_if_strict(self):
        """Get toz does not fail with Omon if not strict."""
        self.variables_info.strict = False
        var = self.variables_info.get_variable("O1", "toz")
        self.assertEqual(var.short_name, "toz")
        self.assertEqual(var.frequency, "")


class TestCORDEXInfo(unittest.TestCase):
    """Test for the CORDEX info class."""

    @classmethod
    def setUpClass(cls):
        """Set up tests.

        We read CORDEX once to keep testing times manageable
        """
        cls.variables_info = CMIP5Info("cordex", default=CustomInfo())

    def test_custom_tables_location(self):
        """Test constructor with custom tables location."""
        cmor_path = os.path.dirname(os.path.realpath(esmvalcore.cmor.__file__))
        cmor_tables_path = os.path.join(cmor_path, "tables", "cordex")
        CMIP5Info(cmor_tables_path)

    def test_get_variable_tas(self):
        """Get tas variable."""
        var = self.variables_info.get_variable("mon", "tas")
        self.assertEqual(var.short_name, "tas")

    def test_get_bad_variable(self):
        """Get none if a variable is not in the given table."""
        self.assertIsNone(self.variables_info.get_variable("Omon", "tas"))


class TestCustomInfo(unittest.TestCase):
    """Test for the custom info class."""

    @classmethod
    def setUpClass(cls):
        """Set up tests.

        We read CMIP5Info once to keep testing times manageable
        """
        cls.variables_info = CustomInfo()

    def test_custom_tables_default_location(self):
        """Test constructor with default tables location."""
        custom_info = CustomInfo()
        expected_cmor_folder = os.path.join(
            os.path.dirname(esmvalcore.cmor.__file__),
            "tables",
            "custom",
        )
        self.assertEqual(custom_info._cmor_folder, expected_cmor_folder)
        self.assertTrue(custom_info.tables["custom"])
        self.assertTrue(custom_info.coords)

    def test_custom_tables_location(self):
        """Test constructor with custom tables location."""
        cmor_path = os.path.dirname(os.path.realpath(esmvalcore.cmor.__file__))
        default_cmor_tables_path = os.path.join(cmor_path, "tables", "custom")
        cmor_tables_path = os.path.join(cmor_path, "tables", "cmip5")
        cmor_tables_path = os.path.abspath(cmor_tables_path)

        custom_info = CustomInfo(cmor_tables_path)

        self.assertEqual(custom_info._cmor_folder, default_cmor_tables_path)
        self.assertEqual(custom_info._user_table_folder, cmor_tables_path)
        self.assertTrue(custom_info.tables["custom"])
        self.assertTrue(custom_info.coords)

    def test_custom_tables_invalid_location(self):
        """Test constructor with invalid custom tables location."""
        with self.assertRaises(ValueError):
            CustomInfo("this_file_does_not_exist.dat")

    def test_get_variable_netcre(self):
        """Get tas variable."""
        CustomInfo()
        var = self.variables_info.get_variable("Amon", "netcre")
        self.assertEqual(var.short_name, "netcre")

    def test_get_bad_variable(self):
        """Get none if a variable is not in the given table."""
        self.assertIsNone(self.variables_info.get_variable("Omon", "badvar"))

    def test_get_variable_tasconf5(self):
        """Get tas variable."""
        CustomInfo()
        var = self.variables_info.get_variable("Amon", "tasConf5")
        self.assertEqual(var.short_name, "tasConf5")
        self.assertEqual(
            var.long_name,
            "Near-Surface Air Temperature Uncertainty Range",
        )
        self.assertEqual(var.units, "K")

    def test_get_variable_tasconf95(self):
        """Get tas variable."""
        CustomInfo()
        var = self.variables_info.get_variable("Amon", "tasConf95")
        self.assertEqual(var.short_name, "tasConf95")
        self.assertEqual(
            var.long_name,
            "Near-Surface Air Temperature Uncertainty Range",
        )
        self.assertEqual(var.units, "K")

    def test_get_variable_tasaga(self):
        """Get tas variable."""
        CustomInfo()
        var = self.variables_info.get_variable("Amon", "tasaga")
        self.assertEqual(var.short_name, "tasaga")
        self.assertEqual(
            var.long_name,
            "Global-mean Near-Surface Air Temperature Anomaly",
        )
        self.assertEqual(var.units, "K")

    def test_get_variable_ch4s(self):
        """Get ch4s variable."""
        CustomInfo()
        var = self.variables_info.get_variable("Amon", "ch4s")
        self.assertEqual(var.short_name, "ch4s")
        self.assertEqual(var.long_name, "Atmosphere CH4 surface")
        self.assertEqual(var.units, "1e-09")

    def test_get_variable_tosstderr(self):
        """Get tosStderr variable."""
        CustomInfo()
        var = self.variables_info.get_variable("Omon", "tosStderr")
        self.assertEqual(var.short_name, "tosStderr")
        self.assertEqual(var.long_name, "Sea Surface Temperature Error")
        self.assertEqual(var.units, "K")


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
