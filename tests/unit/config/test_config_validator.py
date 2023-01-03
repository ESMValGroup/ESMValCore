from pathlib import Path

import numpy as np
import pytest

from esmvalcore import __version__ as current_version
from esmvalcore.config._config_validators import (
    _listify_validator,
    deprecate,
    validate_bool,
    validate_check_level,
    validate_diagnostics,
    validate_float,
    validate_int,
    validate_int_or_none,
    validate_int_positive_or_none,
    validate_path,
    validate_path_or_none,
    validate_positive,
    validate_string,
    validate_string_or_none,
)


def generate_validator_testcases(valid):
    # The code for this function was taken from matplotlib (v3.3) and modified
    # to fit the needs of ESMValCore. Matplotlib is licenced under the terms of
    # the the 'Python Software Foundation License'
    # (https://www.python.org/psf/license)

    validation_tests = (
        {
            'validator': validate_bool,
            'success': ((True, True), (False, False)),
            'fail': ((_, ValueError) for _ in ('fail', 2, -1, []))
        },
        {
            'validator': validate_check_level,
            'success': (
                (1, 1),
                (5, 5),
                ('dEBUG', 1),
                ('default', 3),
            ),
            'fail': (
                (6, ValueError),
                (0, ValueError),
                ('fail', ValueError),
            ),
        },
        {
            'validator':
            validate_diagnostics,
            'success': (
                ('/', {'/'}),
                ('a ', {'a/*'}),
                ('/ a ', {'/', 'a/*'}),
                ('/ a a', {'/', 'a/*'}),
                (('/', 'a'), {'/', 'a/*'}),
                ([], set()),
            ),
            'fail': (
                (1, TypeError),
                ([1, 2], TypeError),
            ),
        },
        {
            'validator':
            _listify_validator(validate_float, n_items=2),
            'success':
            ((_, [1.5, 2.5])
             for _ in ('1.5, 2.5', [1.5, 2.5], [1.5, 2.5], (1.5, 2.5),
                       np.array((1.5, 2.5)))),
            'fail': ((_, ValueError) for _ in ('fail', ('a', 1), (1, 2, 3)))
        },
        {
            'validator':
            _listify_validator(validate_float, n_items=2),
            'success':
            ((_, [1.5, 2.5])
             for _ in ('1.5, 2.5', [1.5, 2.5], [1.5, 2.5], (1.5, 2.5),
                       np.array((1.5, 2.5)))),
            'fail': ((_, ValueError) for _ in ('fail', ('a', 1), (1, 2, 3)))
        },
        {
            'validator':
            _listify_validator(validate_int, n_items=2),
            'success':
            ((_, [1, 2])
             for _ in ('1, 2', [1.5, 2.5], [1, 2], (1, 2), np.array((1, 2)))),
            'fail': ((_, ValueError) for _ in ('fail', ('a', 1), (1, 2, 3)))
        },
        {
            'validator': validate_int_or_none,
            'success': ((None, None), ),
            'fail': (),
        },
        {
            'validator': validate_int_positive_or_none,
            'success': ((None, None), ),
            'fail': (),
        },
        {
            'validator':
            validate_path,
            'success': (
                ('a/b/c', Path.cwd() / 'a' / 'b' / 'c'),
                ('/a/b/c/', Path('/', 'a', 'b', 'c')),
                ('~/', Path.home()),
            ),
            'fail': (
                (None, ValueError),
                (123, ValueError),
                (False, ValueError),
                ([], ValueError),
            ),
        },
        {
            'validator': validate_path_or_none,
            'success': ((None, None), ),
            'fail': (),
        },
        {
            'validator': validate_positive,
            'success': (
                (0.1, 0.1),
                (1, 1),
                (1.5, 1.5),
            ),
            'fail': (
                (0, ValueError),
                (-1, ValueError),
                ('fail', TypeError),
            ),
        },
        {
            'validator':
            _listify_validator(validate_string),
            'success': (
                ('', []),
                ('a,b', ['a', 'b']),
                ('abc', ['abc']),
                ('abc, ', ['abc']),
                ('abc, ,', ['abc']),
                (['a', 'b'], ['a', 'b']),
                (('a', 'b'), ['a', 'b']),
                (iter(['a', 'b']), ['a', 'b']),
                (np.array(['a', 'b']), ['a', 'b']),
                ((1, 2), ['1', '2']),
                (np.array([1, 2]), ['1', '2']),
            ),
            'fail': (
                (set(), ValueError),
                (1, ValueError),
            )
        },
        {
            'validator': validate_string_or_none,
            'success': ((None, None), ),
            'fail': (),
        },
    )

    for validator_dict in validation_tests:
        validator = validator_dict['validator']
        if valid:
            for arg, target in validator_dict['success']:
                yield validator, arg, target
        else:
            for arg, error_type in validator_dict['fail']:
                yield validator, arg, error_type


@pytest.mark.parametrize('validator, arg, target',
                         generate_validator_testcases(True))
def test_validator_valid(validator, arg, target):
    res = validator(arg)
    assert res == target


@pytest.mark.parametrize('validator, arg, exception_type',
                         generate_validator_testcases(False))
def test_validator_invalid(validator, arg, exception_type):
    with pytest.raises(exception_type):
        validator(arg)


@pytest.mark.parametrize('version', (current_version, '0.0.1', '9.9.9'))
def test_deprecate(version):
    def test_func():
        pass

    # This always warns
    with pytest.warns(UserWarning):
        f = deprecate(test_func, 'test_var', version)

    assert callable(f)
