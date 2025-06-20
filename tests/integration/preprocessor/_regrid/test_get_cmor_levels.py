"""Tests for :func:`esmvalcore.preprocessor.regrid.get_cmor_levels`."""

import unittest

from esmvalcore.preprocessor import _regrid


class TestGetCmorLevels(unittest.TestCase):
    def test_cmip6_alt40(self):
        self.assertListEqual(
            _regrid.get_cmor_levels("CMIP6", "alt40"),
            [
                240.0,
                720.0,
                1200.0,
                1680.0,
                2160.0,
                2640.0,
                3120.0,
                3600.0,
                4080.0,
                4560.0,
                5040.0,
                5520.0,
                6000.0,
                6480.0,
                6960.0,
                7440.0,
                7920.0,
                8400.0,
                8880.0,
                9360.0,
                9840.0,
                10320.0,
                10800.0,
                11280.0,
                11760.0,
                12240.0,
                12720.0,
                13200.0,
                13680.0,
                14160.0,
                14640.0,
                15120.0,
                15600.0,
                16080.0,
                16560.0,
                17040.0,
                17520.0,
                18000.0,
                18480.0,
                18960.0,
            ],
        )

    def test_cmip6_p200(self):
        self.assertListEqual(
            _regrid.get_cmor_levels("CMIP6", "p200"),
            [20000.0],
        )

    def test_cmip5_alt40(self):
        self.assertListEqual(
            _regrid.get_cmor_levels("CMIP5", "plevs"),
            [
                100000.0,
                92500.0,
                85000.0,
                70000.0,
                60000.0,
                50000.0,
                40000.0,
                30000.0,
                25000.0,
                20000.0,
                15000.0,
                10000.0,
                7000.0,
                5000.0,
                3000.0,
                2000.0,
                1000.0,
            ],
        )

    def test_cmip5_p500(self):
        self.assertListEqual(_regrid.get_cmor_levels("CMIP5", "p500"), [50000])

    def test_not_values_in_coordinate(self):
        with self.assertRaises(ValueError):
            _regrid.get_cmor_levels("CMIP6", "time")

    def test_bad_table(self):
        with self.assertRaises(ValueError):
            _regrid.get_cmor_levels("CMOCK", "p500")

    def test_bad_coordinate(self):
        with self.assertRaises(ValueError):
            _regrid.get_cmor_levels("CMIP5", "uglycoord")
