"""Unit tests for the variable_info module."""

import unittest

from esmvalcore.cmor.table import CoordinateInfo, VariableInfo


class TestVariableInfo(unittest.TestCase):
    """Variable info tests."""

    def setUp(self):
        """Prepare for testing."""
        self.info = VariableInfo('table_type', 'var')
        self.value = 'value'

    def test_constructor(self):
        """Test basic constructor."""
        self.assertEqual('table_type', self.info.table_type)
        self.assertEqual('var', self.info.short_name)

    def test_read_empty_dictionary(self):
        """Test read empty dict."""
        self.info.read_json({}, '')
        self.assertEqual('', self.info.standard_name)

    def test_read_standard_name(self):
        """Test standard_name."""
        self.info.read_json({'standard_name': self.value}, '')
        self.assertEqual(self.info.standard_name, self.value)

    def test_read_long_name(self):
        """Test long_name."""
        self.info.read_json({'long_name': self.value}, '')
        self.assertEqual(self.info.long_name, self.value)

    def test_read_units(self):
        """Test units."""
        self.info.read_json({'units': self.value}, '')
        self.assertEqual(self.info.units, self.value)

    def test_read_valid_min(self):
        """Test valid_min."""
        self.info.read_json({'valid_min': self.value}, '')
        self.assertEqual(self.info.valid_min, self.value)

    def test_read_valid_max(self):
        """Test valid_max."""
        self.info.read_json({'valid_max': self.value}, '')
        self.assertEqual(self.info.valid_max, self.value)

    def test_read_positive(self):
        """Test positive."""
        self.info.read_json({'positive': self.value}, '')
        self.assertEqual(self.info.positive, self.value)

    def test_read_frequency(self):
        """Test frequency."""
        self.info.read_json({'frequency': self.value}, '')
        self.assertEqual(self.info.frequency, self.value)

    def test_read_default_frequency(self):
        """Test frequency."""
        self.info.read_json({}, self.value)
        self.assertEqual(self.info.frequency, self.value)

    def test_has_dimension_that_startswith_empty_dim(self):
        """Test `has_dimension_that_startswith`."""
        assert self.info.has_dimension_that_startswith('time') is False

    def test_has_dimension_that_startswith_false(self):
        """Test `has_dimension_that_startswith`."""
        self.info.dimensions = ['test', 'not_time', 'TIME', '_time_']
        assert self.info.has_dimension_that_startswith('time') is False

    def test_has_dimension_that_startswith_true(self):
        """Test `has_dimension_that_startswith`."""
        self.info.dimensions = ['test', 'time1']
        assert self.info.has_dimension_that_startswith('time') is True

    def test_has_dimension_that_startswith_multiple(self):
        """Test `has_dimension_that_startswith`."""
        self.info.dimensions = ['test', 'time1', 'time', 'time2']
        assert self.info.has_dimension_that_startswith('time') is True


class TestCoordinateInfo(unittest.TestCase):
    """Tests for CoordinataInfo."""

    def setUp(self):
        """Prepare for testing."""
        self.value = 'value'

    def test_constructor(self):
        """Test constructor."""
        info = CoordinateInfo('var')
        self.assertEqual('var', info.name)

    def test_read_empty_dictionary(self):
        """Test empty dict."""
        info = CoordinateInfo('var')
        info.read_json({})
        self.assertEqual('', info.standard_name)

    def test_read_standard_name(self):
        """Test standard_name."""
        info = CoordinateInfo('var')
        info.read_json({'standard_name': self.value})
        self.assertEqual(info.standard_name, self.value)

    def test_read_var_name(self):
        """Test var_name."""
        info = CoordinateInfo('var')
        info.read_json({'var_name': self.value})
        self.assertEqual(info.var_name, self.value)

    def test_read_out_name(self):
        """Test out_name."""
        info = CoordinateInfo('var')
        info.read_json({'out_name': self.value})
        self.assertEqual(info.out_name, self.value)

    def test_read_units(self):
        """Test units."""
        info = CoordinateInfo('var')
        info.read_json({'units': self.value})
        self.assertEqual(info.units, self.value)

    def test_read_valid_min(self):
        """Test valid_min."""
        info = CoordinateInfo('var')
        info.read_json({'valid_min': self.value})
        self.assertEqual(info.valid_min, self.value)

    def test_read_valid_max(self):
        """Test valid_max."""
        info = CoordinateInfo('var')
        info.read_json({'valid_max': self.value})
        self.assertEqual(info.valid_max, self.value)

    def test_read_value(self):
        """Test value."""
        info = CoordinateInfo('var')
        info.read_json({'value': self.value})
        self.assertEqual(info.value, self.value)

    def test_read_requested(self):
        """Test requested."""
        value = ['value1', 'value2']
        info = CoordinateInfo('var')
        info.read_json({'requested': value})
        self.assertEqual(info.requested, value)
