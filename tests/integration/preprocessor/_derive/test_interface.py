"""Integration tests for derive module."""

from unittest import mock

import pytest
from cf_units import Unit
from iris.cube import Cube, CubeList

from esmvalcore.preprocessor import _derive, derive
from esmvalcore.preprocessor._derive import get_required
from esmvalcore.preprocessor._derive.ohc import DerivedVariable

SHORT_NAME = 'short_name'


@pytest.fixture
def mock_cubes():
    """Fixture for mock version of :class:`iris.cube.CubeList`."""
    return mock.create_autospec(CubeList, spec_set=True, instance=True)


@pytest.fixture
def patched_derive(mocker):
    """Fixture for mocked derivation scripts."""
    mocker.patch('iris.cube.CubeList', side_effect=lambda x: x)
    mocker.patch.object(_derive, 'ALL_DERIVED_VARIABLES', autospec=True)
    mocker.patch.object(_derive, 'logger', autospec=True)


def mock_all_derived_variables(returned_units):
    """Mock the :obj:`dict` containing all derived variables accordingly."""
    cube = mock.create_autospec(Cube, instance=True)
    cube.units = returned_units
    calculate_function = mock.Mock(return_value=cube)
    derived_var = mock.Mock(name='DerivedVariable')
    derived_var.return_value.calculate = calculate_function
    _derive.ALL_DERIVED_VARIABLES.__getitem__.return_value = derived_var


def assert_derived_var_calc_called_once_with(*args):
    """Assert that derivation script of variable has been called."""
    (_derive.ALL_DERIVED_VARIABLES.__getitem__.return_value.return_value.
     calculate.assert_called_once_with(*args))


@pytest.mark.usefixtures("patched_derive")
def test_check_units_none(mock_cubes):
    """Test units after derivation if derivation scripts returns None."""
    mock_all_derived_variables(None)
    cube = derive(mock_cubes, SHORT_NAME, mock.sentinel.long_name,
                  mock.sentinel.units,
                  standard_name=mock.sentinel.standard_name)
    assert_derived_var_calc_called_once_with(mock_cubes)
    assert cube.units == mock.sentinel.units
    assert cube.var_name == SHORT_NAME
    assert cube.long_name == mock.sentinel.long_name
    assert cube.standard_name == mock.sentinel.standard_name
    _derive.logger.warning.assert_not_called()
    cube.convert_units.assert_not_called()


@pytest.mark.usefixtures("patched_derive")
def test_check_units_equal(mock_cubes):
    """Test units after derivation if derivation scripts returns None."""
    mock_all_derived_variables(Unit('kg m2 s-2'))
    cube = derive(mock_cubes, SHORT_NAME, mock.sentinel.long_name, 'J',
                  standard_name=mock.sentinel.standard_name)
    assert_derived_var_calc_called_once_with(mock_cubes)
    assert cube.units == Unit('J')
    assert cube.var_name == SHORT_NAME
    assert cube.long_name == mock.sentinel.long_name
    assert cube.standard_name == mock.sentinel.standard_name
    _derive.logger.warning.assert_not_called()
    cube.convert_units.assert_not_called()


@pytest.mark.usefixtures("patched_derive")
def test_check_units_nounit(mock_cubes):
    """Test units after derivation if derivation scripts returns None."""
    mock_all_derived_variables(Unit('no unit'))
    cube = derive(mock_cubes, SHORT_NAME, mock.sentinel.long_name, 'J',
                  standard_name=mock.sentinel.standard_name)
    assert_derived_var_calc_called_once_with(mock_cubes)
    assert cube.units == Unit('J')
    assert cube.var_name == SHORT_NAME
    assert cube.long_name == mock.sentinel.long_name
    assert cube.standard_name == mock.sentinel.standard_name
    _derive.logger.warning.assert_called_once_with(
        "Units of cube after executing derivation script of '%s' are '%s', "
        "automatically setting them to '%s'. This might lead to incorrect "
        "data", SHORT_NAME, Unit('no_unit'), 'J')
    cube.convert_units.assert_not_called()


@pytest.mark.usefixtures("patched_derive")
def test_check_units_unknown(mock_cubes):
    """Test units after derivation if derivation scripts returns None."""
    mock_all_derived_variables(Unit('unknown'))
    cube = derive(mock_cubes, SHORT_NAME, mock.sentinel.long_name, 'J',
                  standard_name=mock.sentinel.standard_name)
    assert_derived_var_calc_called_once_with(mock_cubes)
    assert cube.units == Unit('J')
    assert cube.var_name == SHORT_NAME
    assert cube.long_name == mock.sentinel.long_name
    assert cube.standard_name == mock.sentinel.standard_name
    _derive.logger.warning.assert_called_once_with(
        "Units of cube after executing derivation script of '%s' are '%s', "
        "automatically setting them to '%s'. This might lead to incorrect "
        "data", SHORT_NAME, Unit('unknown'), 'J')
    cube.convert_units.assert_not_called()


@pytest.mark.usefixtures("patched_derive")
def test_check_units_convertible(mock_cubes):
    """Test units after derivation if derivation scripts returns None."""
    mock_all_derived_variables(Unit('kg s-1'))
    cube = derive(mock_cubes, SHORT_NAME, mock.sentinel.long_name, 'g yr-1',
                  standard_name=mock.sentinel.standard_name)
    assert_derived_var_calc_called_once_with(mock_cubes)
    cube.convert_units.assert_called_once_with('g yr-1')
    assert cube.var_name == SHORT_NAME
    assert cube.long_name == mock.sentinel.long_name
    assert cube.standard_name == mock.sentinel.standard_name
    _derive.logger.warning.assert_not_called()


@pytest.mark.usefixtures("patched_derive")
def test_check_units_fail(mock_cubes):
    """Test units after derivation if derivation scripts returns None."""
    mock_all_derived_variables(Unit('kg'))
    with pytest.raises(ValueError) as err:
        derive(mock_cubes, SHORT_NAME, mock.sentinel.long_name, 'm',
               standard_name=mock.sentinel.standard_name)
    assert str(err.value) == (
        "Units 'kg' after executing derivation script of 'short_name' cannot "
        "be converted to target units 'm'"
    )
    _derive.logger.warning.assert_not_called()


def test_get_required():
    """Test getting required variables for derivation."""
    variables = get_required('alb', 'CMIP5')

    reference = [
        {
            'short_name': 'rsdscs',
        },
        {
            'short_name': 'rsuscs',
        },
    ]

    assert variables == reference


def test_get_required_with_fx():
    """Test getting required variables for derivation with fx variables."""
    variables = get_required('ohc', 'CMIP5')

    reference = [
        {'short_name': 'thetao'},
        {'short_name': 'volcello', 'mip': 'fx'},
    ]

    assert variables == reference


def test_derive_nonstandard_nofx():
    """Test a specific derivation."""
    short_name = 'alb'
    long_name = 'albedo at the surface'
    units = 1
    standard_name = ''

    rsdscs = Cube([2.])
    rsdscs.short_name = 'rsdscs'
    rsdscs.var_name = rsdscs.short_name

    rsuscs = Cube([1.])
    rsuscs.short_name = 'rsuscs'
    rsuscs.var_name = rsuscs.short_name

    cubes = CubeList([rsdscs, rsuscs])

    alb = derive(cubes, short_name, long_name, units, standard_name)

    assert alb.var_name == short_name
    assert alb.long_name == long_name
    assert alb.units == units
    assert alb.data == [0.5]


def test_derive_noop():
    """Test derivation when it is not necessary."""
    alb = Cube([1.])
    alb.var_name = 'alb'
    alb.long_name = 'albedo at the surface'
    alb.units = 1

    cube = derive([alb], alb.var_name, alb.long_name, alb.units)

    assert cube is alb


def test_derive_mixed_case_with_fx(monkeypatch):
    """Test derivation with fx file."""
    short_name = 'ohc'
    long_name = 'Heat content in grid cell'
    units = 'J'

    ohc_cube = Cube([])

    def mock_calculate(_, cubes):
        assert len(cubes) == 1
        assert cubes[0] == ohc_cube
        return Cube([])

    monkeypatch.setattr(DerivedVariable, 'calculate', mock_calculate)

    derive(
        [ohc_cube],
        short_name,
        long_name,
        units,
    )
