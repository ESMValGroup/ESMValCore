"""Unit tests for :func:`esmvalcore._data_finder.regrid._stock_cube`"""

import unittest

import esmvalcore._config
from esmvalcore._data_finder import get_start_end_year


esmvalcore._config.CFG = esmvalcore._config.read_config_developer_file()
esmvalcore._config.CFG['TEST'] = {
    'date_prefix_tags': '[dataset]_[random_tag]_',
}


class TestGetStartEndYear(unittest.TestCase):
    """Tests for get_start_end_year function"""

    def setUp(self):
        """Prepare tests"""
        self.var_cmip5 = {'project': 'CMIP5'}
        self.var_emac =  {'project': 'EMAC'}
        self.var_test =  {
            'project': 'TEST',
            'dataset': 'xyz',
            'random_tag': 'ooo',
        }

    def test_one_year_test(self):
        """Test default style for EMAC."""
        start, end = get_start_end_year('xyz_ooo_198001', self.var_test)
        self.assertEqual(1980, start)
        self.assertEqual(1980, end)

    def test_years_test(self):
        """Test default style for EMAC."""
        start, end = get_start_end_year('xyz_ooo_198001-200001', self.var_test)
        self.assertEqual(1980, start)
        self.assertEqual(2000, end)

    def test_one_year_emac(self):
        """Test default style for EMAC."""
        start, end = get_start_end_year('1234567890_____198001', self.var_emac)
        self.assertEqual(1980, start)
        self.assertEqual(1980, end)

    def test_years_emac(self):
        """Test default style for EMAC."""
        start, end = get_start_end_year('1234567890_____198001-200001', self.var_emac)
        self.assertEqual(1980, start)
        self.assertEqual(2000, end)

    def test_years_at_the_end_cmip5(self):
        """Test parse files with two years at the end"""
        start, end = get_start_end_year('var_whatever_1980-1981', self.var_cmip5)
        self.assertEqual(1980, start)
        self.assertEqual(1981, end)

    def test_one_year_at_the_end_cmip5(self):
        """Test parse files with one year at the end"""
        start, end = get_start_end_year('var_whatever_1980.nc', self.var_cmip5)
        self.assertEqual(1980, start)
        self.assertEqual(1980, end)

    def test_full_dates_at_the_end_cmip5(self):
        """Test parse files with two dates at the end"""
        start, end = get_start_end_year('var_whatever_19800101-19811231.nc', self.var_cmip5)

        self.assertEqual(1980, start)
        self.assertEqual(1981, end)

    def test_one_fulldate_at_the_end_cmip5(self):
        """Test parse files with one date at the end"""
        start, end = get_start_end_year('var_whatever_19800101.nc', self.var_cmip5)
        self.assertEqual(1980, start)
        self.assertEqual(1980, end)

    def test_years_at_the_start_cmip5(self):
        """Test parse files with two years at the start"""
        start, end = get_start_end_year('1980-1981_var_whatever.nc', self.var_cmip5)
        self.assertEqual(1980, start)
        self.assertEqual(1981, end)

    def test_one_year_at_the_start_cmip5(self):
        """Test parse files with one year at the start"""
        start, end = get_start_end_year('1980_var_whatever.nc', self.var_cmip5)
        self.assertEqual(1980, start)
        self.assertEqual(1980, end)

    def test_full_dates_at_the_start_cmip5(self):
        """Test parse files with two dates at the start"""
        start, end = get_start_end_year('19800101-19811231_var_whatever.nc', self.var_cmip5)
        self.assertEqual(1980, start)
        self.assertEqual(1981, end)

    def test_one_fulldate_at_the_start_cmip5(self):
        """Test parse files with one date at the start"""
        start, end = get_start_end_year('19800101_var_whatever.nc', self.var_cmip5)
        self.assertEqual(1980, start)
        self.assertEqual(1980, end)

    def test_start_and_date_in_name_cmip5(self):
        """Test parse one date at the start and one in experiment's name"""
        start, end = get_start_end_year('19800101_var_control-1950_whatever.nc', self.var_cmip5)
        self.assertEqual(1980, start)
        self.assertEqual(1980, end)

    def test_end_and_date_in_name_cmip5(self):
        """Test parse one date at the end and one in experiment's name"""
        start, end = get_start_end_year('var_control-1950_whatever_19800101.nc', self.var_cmip5)
        self.assertEqual(1980, start)
        self.assertEqual(1980, end)

    def test_fails_if_no_date_present_cmip5(self):
        """Test raises if no date is present"""
        with self.assertRaises(ValueError):
            get_start_end_year('var_whatever', self.var_cmip5)
