import pytest

from esmvalcore.preprocessor import _supplementary_vars


def test_register(monkeypatch):
    """Test registering an supplementary variable."""
    registered = {}
    monkeypatch.setattr(
        _supplementary_vars,
        'PREPROCESSOR_SUPPLEMENTARIES',
        registered,
    )

    @_supplementary_vars.register_supplementaries(
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

        @_supplementary_vars.register_supplementaries(
            ['areacella'],
            required='invalid',
        )
        def test_func():
            pass
