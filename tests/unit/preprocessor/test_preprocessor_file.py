"""Unit tests for :class:`esmvalcore.preprocessor.PreprocessorFile`."""

from unittest import mock

import pytest
from iris.cube import Cube, CubeList

from esmvalcore.preprocessor import PreprocessorFile

ATTRIBUTES = {
    'filename': 'file.nc',
    'standard_name': 'precipitation',
    'long_name': 'Precipitation',
    'short_name': 'pr',
    'units': 'kg m-2 s-1',
    'frequency': 'mon',
}


@pytest.fixture
def product():
    """PreprocessorFile object used for testing."""
    cube = Cube(
        0,
        var_name='tas',
        standard_name='air_temperature',
        long_name='Near-Surface Air Temperature',
        units='K',
        attributes={'frequency': 'day'},
    )
    product = PreprocessorFile(attributes=ATTRIBUTES, settings={})
    product._cubes = CubeList([cube, cube, cube])
    return product


def test_update_attributes_empty_cubes(product):
    """Test ``update_attributes``."""
    product._cubes = CubeList([])
    product.update_attributes()

    assert not product._cubes
    assert product.attributes == ATTRIBUTES


def test_update_attributes(product):
    """Test ``update_attributes``."""
    product.update_attributes()

    assert product.attributes == {
        'filename': 'file.nc',
        'standard_name': 'air_temperature',
        'long_name': 'Near-Surface Air Temperature',
        'short_name': 'tas',
        'units': 'K',
        'frequency': 'day',
    }
    assert isinstance(product.attributes['units'], str)


def test_update_attributes_empty_standard_name(product):
    """Test ``update_attributes``."""
    product._cubes[0].standard_name = None
    product.update_attributes()

    assert product.attributes == {
        'filename': 'file.nc',
        'standard_name': '',
        'long_name': 'Near-Surface Air Temperature',
        'short_name': 'tas',
        'units': 'K',
        'frequency': 'day',
    }
    assert isinstance(product.attributes['units'], str)


def test_update_attributes_empty_long_name(product):
    """Test ``update_attributes``."""
    product._cubes[0].long_name = None
    product.update_attributes()

    assert product.attributes == {
        'filename': 'file.nc',
        'standard_name': 'air_temperature',
        'long_name': '',
        'short_name': 'tas',
        'units': 'K',
        'frequency': 'day',
    }
    assert isinstance(product.attributes['units'], str)


def test_update_attributes_empty_var_name(product):
    """Test ``update_attributes``."""
    product._cubes[0].var_name = None
    product.update_attributes()

    assert product.attributes == {
        'filename': 'file.nc',
        'standard_name': 'air_temperature',
        'long_name': 'Near-Surface Air Temperature',
        'short_name': '',
        'units': 'K',
        'frequency': 'day',
    }
    assert isinstance(product.attributes['units'], str)


def test_update_attributes_empty_frequency(product):
    """Test ``update_attributes``."""
    product._cubes[0].attributes.pop('frequency')
    product.update_attributes()

    assert product.attributes == {
        'filename': 'file.nc',
        'standard_name': 'air_temperature',
        'long_name': 'Near-Surface Air Temperature',
        'short_name': 'tas',
        'units': 'K',
        'frequency': 'mon',
    }
    assert isinstance(product.attributes['units'], str)


def test_update_attributes_no_frequency(product):
    """Test ``update_attributes``."""
    product._cubes[0].attributes.pop('frequency')
    product.attributes.pop('frequency')
    product.update_attributes()

    assert product.attributes == {
        'filename': 'file.nc',
        'standard_name': 'air_temperature',
        'long_name': 'Near-Surface Air Temperature',
        'short_name': 'tas',
        'units': 'K',
    }
    assert isinstance(product.attributes['units'], str)


def test_close_no_cubes():
    """Test ``close``."""
    product = mock.create_autospec(PreprocessorFile, instance=True)
    product._cubes = None

    PreprocessorFile.close(product)

    product.update_attributes.assert_not_called()
    product.save.assert_not_called()
    product.save_provenance.assert_not_called()
    assert product._cubes is None


def test_close():
    """Test ``close``."""
    product = mock.create_autospec(PreprocessorFile, instance=True)
    product._cubes = CubeList([Cube(0)])

    PreprocessorFile.close(product)

    product.update_attributes.assert_called_once_with()
    product.save.assert_called_once_with()
    product.save_provenance.assert_called_once_with()
    assert product._cubes is None
