import pytest

from esmvalcore.preprocessor import _ancillary_vars


def test_register(monkeypatch):
    """Test registering an ancillary variable."""
    registered = {}
    monkeypatch.setattr(
        _ancillary_vars,
        'PREPROCESSOR_ANCILLARIES',
        registered,
    )

    @_ancillary_vars.register_ancillaries(
        ['areacella'],
        required='require_at_least_one',
    )
    def test_func():
        pass

    assert registered == {
        'test_func': {
            'required': 'require_at_least_one',
            'variables': ['areacella'],
        }
    }


def test_register_invalid_fails():
    """test that registering an invalid requirement fails."""
    with pytest.raises(NotImplementedError):

        @_ancillary_vars.register_ancillaries(
            ['areacella'],
            required='invalid',
        )
        def test_func():
            pass
